"""Opt-in compaction policies built on the context-window seam.

Core never installs a compactor.  Pass :class:`LlmCompactor` to
``AgentCore(context_compactor=...)`` to summarize old turns with a model when
the previous iteration's input usage crosses a threshold.  Any object with an
async ``chat(prompt) -> CLIverResponse`` works as the summarizer — typically a
second :class:`~cliver_core.agent.AgentCore` bound to a cheaper model.

The compactor returns a :class:`~cliver_core.compaction.CompactionPlan`, so core
still owns the boundary check, the summary framing, and usage accounting.
"""

from __future__ import annotations

from typing import Any, Protocol

from cliver_core.compaction import CompactionContext, CompactionPlan
from cliver_core.provider import CLIverResponse

DEFAULT_SUMMARY_PROMPT = (
    "Summarize the earlier part of this agent run for the same assistant. "
    "Keep facts, decisions, tool results that still matter, and open tasks. "
    "Be compact and do not invent."
)


class Summarizer(Protocol):
    """Anything with an async ``chat(prompt)`` returning a ``CLIverResponse``."""

    async def chat(self, prompt: str, **kwargs: Any) -> CLIverResponse: ...


class LlmCompactor:
    """Threshold-triggered summarization, driven by the previous call's usage.

    ``threshold_tokens`` compares against ``CompactionContext.last_usage.input_tokens``
    (no tokenizer, no extra dependency).  ``keep_turns`` is the number of most
    recent user turns that survive verbatim; older turns are rendered and
    summarized.  The summarization call's usage is attached to the plan, so
    core folds it into the run's ``usage_total``.

    The compactor is silent below the threshold, on the first iteration, and
    when there is nothing older to drop.  An empty summary leaves the history
    unchanged (its usage is still recorded).
    """

    def __init__(
        self,
        summarizer: Summarizer,
        *,
        threshold_tokens: int,
        keep_turns: int = 2,
        prompt: str = DEFAULT_SUMMARY_PROMPT,
    ) -> None:
        if not callable(getattr(summarizer, "chat", None)):
            raise ValueError("summarizer must provide an async chat(prompt) method")
        if not isinstance(threshold_tokens, int) or isinstance(threshold_tokens, bool) or threshold_tokens < 1:
            raise ValueError("threshold_tokens must be a positive integer")
        if not isinstance(keep_turns, int) or isinstance(keep_turns, bool) or keep_turns < 1:
            raise ValueError("keep_turns must be a positive integer")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        self._summarizer = summarizer
        self._threshold_tokens = threshold_tokens
        self._keep_turns = keep_turns
        self._prompt = prompt

    async def __call__(self, context: CompactionContext) -> CompactionPlan | None:
        used = context.last_usage.input_tokens if context.last_usage is not None else None
        if used is None or used < self._threshold_tokens:
            return None

        split = context.split_at_user_turn(self._keep_turns)
        if not split.older:
            return None

        prompt = f"{self._prompt}\n\n{context.render(split.older)}"
        response = await self._summarizer.chat(prompt)
        content = response.message.content
        if not isinstance(content, str) or not content.strip():
            if response.usage_total is not None:
                context.record_usage(response.usage_total)
            return None
        return split.plan(summary=content.strip(), usage=response.usage_total)
