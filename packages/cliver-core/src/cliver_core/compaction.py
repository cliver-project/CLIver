"""Context-window seam for the Re-Act loop.

The loop can only grow its history: every tool batch appends an assistant
message plus one tool result per call.  A caller that must stay inside the
model's context window passes a :data:`ContextCompactor` to
:class:`~cliver_core.agent.AgentCore` (or per call).  Core invokes it before
every model call with the run's non-system history; the compactor returns a
replacement history, or ``None`` to leave the history unchanged.

Core ships no summarization policy, tokenizer, threshold, or persistence.
The callback is the application's policy: a typical implementation checks
``last_usage.input_tokens`` against the model's known context window and
summarizes older turns with a cheap model.  ``record_usage`` folds the
compaction call's usage into the run's ``usage_total`` so metering stays
complete.

A returned history must satisfy :func:`validate_compacted_history`:

* non-empty and free of ``system`` messages — core owns the single system
  prompt and re-attaches it;
* every assistant ``tool_calls`` entry is followed by matching tool
  results, and every tool result follows its assistant tool call;
* the last message is ``user`` or ``tool`` so the model has a turn to
  answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable, Sequence

from cliver_core.messages import CLIverMessage, UsageInfo


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


CompactorResult = list[CLIverMessage] | None

# App seam: sync or async; returns a replacement history or ``None``.
ContextCompactor = Callable[[CompactionContext], CompactorResult | Awaitable[CompactorResult]]


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
