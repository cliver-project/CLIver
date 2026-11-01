"""Context-window seam for the Re-Act loop.

The loop can only grow its history: every tool batch appends an assistant
message plus one tool result per call.  A caller that must stay inside the
model's context window passes a :data:`ContextCompactor` to
:class:`~cliver_core.agent.AgentCore` (or per call).  Core invokes it before
every model call with the run's non-system history; the compactor either
returns a :class:`CompactionPlan` (recommended), returns a full replacement
history (escape hatch), or returns ``None`` to leave the history unchanged.

A plan declares only *what* to compact — ``keep_from``, an optional summary,
and the summarization call's usage — and core materializes the replacement:
it drops everything before ``keep_from``, prepends the summary as the first
user message, preserves tool call/result pairs, re-validates the result, and
records the plan's usage into the run's ``usage_total``.  The structural
invariants therefore live in core, not in every application's compactor.
:meth:`CompactionContext.split_at_user_turn` picks a boundary that cannot cut
a tool call/result pair, and :meth:`CompactionContext.render` formats the
dropped messages for a summarizer prompt.

Core ships no automatic policy: nothing compacts unless the application
supplies a compactor.  :class:`cliver_core.compactors.LlmCompactor` is an
opt-in reference implementation.

A returned history must satisfy :func:`validate_compacted_history`:

* non-empty and free of ``system`` messages — core owns the single system
  prompt and re-attaches it;
* every assistant ``tool_calls`` entry is followed by matching tool
  results, and every tool result follows its assistant tool call;
* the last message is ``user`` or ``tool`` so the model has a turn to
  answer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Awaitable, Callable, Sequence

from cliver_core.messages import CLIverMessage, UsageInfo

COMPACTED_CONTEXT_PREFIX = "[Compacted earlier context]"


@dataclass(slots=True)
class CompactionContext:
    """What a compactor sees before one model call.

    ``messages`` is the run history *excluding* the core-owned system
    message (a fresh list; treat it as read-only).  ``iteration`` is the
    zero-based Re-Act iteration about to call the model.  ``model`` is the
    active binding's model before any failover for this call.  ``last_usage``
    is the previous iteration's usage for this run (``None`` on the first
    iteration).

    ``record_usage`` folds usage from a model call the compactor made — for
    example the summarization call — into this run's ``usage_total``.
    Non-streaming runs return that total on ``CLIverResponse.usage_total``;
    streaming runs expose it as ``usage_total`` on each terminal trace entry.
    """

    messages: list[CLIverMessage]
    iteration: int
    run_id: str | None
    model: str
    last_usage: UsageInfo | None
    record_usage: Callable[[UsageInfo], None]

    def split_at_user_turn(self, keep_turns: int) -> CompactionSplit:
        """Split the history so the last ``keep_turns`` user turns survive.

        The boundary is always a ``user`` message, so no assistant tool call
        is separated from its results.  When the history has fewer than
        ``keep_turns`` user turns, the split keeps everything
        (``keep_from == 0``).
        """
        if not isinstance(keep_turns, int) or isinstance(keep_turns, bool) or keep_turns < 1:
            raise ValueError("keep_turns must be a positive integer")
        positions = [index for index, message in enumerate(self.messages) if message.role == "user"]
        keep_from = positions[-keep_turns] if len(positions) >= keep_turns else 0
        return CompactionSplit(
            keep_from=keep_from,
            older=list(self.messages[:keep_from]),
            recent=list(self.messages[keep_from:]),
        )

    def render(self, messages: Sequence[CLIverMessage]) -> str:
        """Render messages as compact text suitable for a summarizer prompt."""
        return render_messages(messages)


@dataclass(slots=True, frozen=True)
class CompactionPlan:
    """A declarative compaction request that core materializes.

    ``keep_from`` is the index of the first retained message in
    ``CompactionContext.messages`` and must point at a ``user`` message.
    ``summary``, when set, replaces the dropped prefix with one user message
    beginning with :data:`COMPACTED_CONTEXT_PREFIX`.  ``usage`` is the
    summarization call's usage; core folds it into the run's
    ``usage_total`` after the history is accepted.
    """

    keep_from: int
    summary: str | None = None
    usage: UsageInfo | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.keep_from, int) or isinstance(self.keep_from, bool) or self.keep_from < 1:
            raise ValueError("CompactionPlan.keep_from must be a positive integer index into the history")
        if self.summary is not None and (not isinstance(self.summary, str) or not self.summary.strip()):
            raise ValueError("CompactionPlan.summary must be a non-empty string or None")
        if self.usage is not None and not isinstance(self.usage, UsageInfo):
            raise ValueError("CompactionPlan.usage must be a UsageInfo or None")


@dataclass(slots=True)
class CompactionSplit:
    """A safe keep boundary plus the two halves it divides.

    Produce one with :meth:`CompactionContext.split_at_user_turn`, summarize
    ``older`` with :meth:`CompactionContext.render`, then pass the summary to
    :meth:`plan`.  ``keep_from`` is bound to the split, so the plan cannot
    drift from the text the summarizer actually saw.
    """

    keep_from: int
    older: list[CLIverMessage]
    recent: list[CLIverMessage]

    def plan(self, *, summary: str | None = None, usage: UsageInfo | None = None) -> CompactionPlan:
        """Build the matching plan; rejects a split that drops nothing."""
        return CompactionPlan(keep_from=self.keep_from, summary=summary, usage=usage)


CompactorResult = CompactionPlan | list[CLIverMessage] | None

# App seam: sync or async; returns a plan, a replacement history, or ``None``.
ContextCompactor = Callable[[CompactionContext], CompactorResult | Awaitable[CompactorResult]]


def render_messages(messages: Sequence[CLIverMessage]) -> str:
    """Render messages as one line per message for a summarizer prompt."""
    lines: list[str] = []
    for message in messages:
        if message.role == "tool":
            lines.append(f"tool result: {message.content or ''}")
        elif message.role == "assistant" and message.tool_calls:
            calls = ", ".join(
                f"{call.name}({json.dumps(call.args, ensure_ascii=False)})" for call in message.tool_calls
            )
            line = f"assistant -> tools: {calls}"
            if message.content:
                line = f"assistant: {message.content}\n{line}"
            lines.append(line)
        else:
            lines.append(f"{message.role}: {message.content or ''}")
    return "\n".join(lines)


def materialize_compaction(plan: CompactionPlan, context: CompactionContext) -> list[CLIverMessage]:
    """Turn a plan into the replacement history and record its usage.

    Raises ``ValueError`` when ``keep_from`` falls outside the history or does
    not point at a ``user`` message, or when the resulting history violates
    :func:`validate_compacted_history`.  Usage is recorded only after the
    history is accepted.
    """
    messages = context.messages
    if plan.keep_from >= len(messages):
        raise ValueError(
            f"context_compactor plan keep_from={plan.keep_from} must point inside the {len(messages)}-message history"
        )
    if messages[plan.keep_from].role != "user":
        raise ValueError("context_compactor plan keep_from must point at a user message")

    history = list(messages[plan.keep_from :])
    if plan.summary is not None:
        history.insert(0, CLIverMessage(role="user", content=f"{COMPACTED_CONTEXT_PREFIX}\n{plan.summary}"))
    history = validate_compacted_history(history)
    if plan.usage is not None:
        context.record_usage(plan.usage)
    return history


def validate_compacted_history(messages: Sequence[CLIverMessage]) -> list[CLIverMessage]:
    """Validate a compactor's returned history and return it as a list.

    Raises ``ValueError`` for anything the Re-Act loop or the protocols
    cannot safely send — see the module docstring for the rules.
    """
    history = list(messages)
    if not history:
        raise ValueError("context_compactor returned an empty history; at least one non-system message is required")

    pending: set[str] = set()
    for message in history:
        if not isinstance(message, CLIverMessage):
            raise ValueError("context_compactor must return CLIverMessage objects")
        if message.role == "system":
            raise ValueError("context_compactor must not return system messages; core owns the system prompt")
        if message.role == "assistant":
            if pending:
                raise ValueError("context_compactor left assistant tool calls without matching tool results")
            pending = {call.id for call in (message.tool_calls or [])}
        elif message.role == "tool":
            if message.tool_call_id is None or message.tool_call_id not in pending:
                raise ValueError("context_compactor returned a tool result with no matching assistant tool call")
            pending.discard(message.tool_call_id)
        elif pending:
            raise ValueError("context_compactor left assistant tool calls without matching tool results")

    if pending:
        raise ValueError("context_compactor left assistant tool calls without matching tool results")
    if history[-1].role not in ("user", "tool"):
        raise ValueError(
            "context_compactor history must end with a user or tool message so the model has a turn to answer"
        )
    return history
