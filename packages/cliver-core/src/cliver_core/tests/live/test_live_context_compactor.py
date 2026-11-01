"""Live: the context-window seam with the docs' LLM compactor.

A real DeepSeek summarizer compacts a long legacy conversation between Re-Act
iterations; the test proves the replacement history reaches the wire, the
summary model's usage is accounted for, and a per-call ``None`` disables the
constructor compactor.
"""

from __future__ import annotations

import json

import pytest

from cliver_core import AgentCore, CLIverMessage, CompactionContext, ModelBindings, presets
from cliver_core.tests.live.support import (
    LIVE_OFF_REASON,
    drain,
    live_enabled,
    skip_unless_keys,
    stream_trace,
    tool_names,
)
from cliver_core.tool import tool

pytestmark = pytest.mark.skipif(not live_enabled(), reason=LIVE_OFF_REASON)

SYSTEM = "You are an order-status assistant. You must call lookup_order exactly once before answering the user."


@tool(name="lookup_order", description="Return the status of one order id.")
def lookup_order(order_id: str) -> str:
    return f"Order {order_id}: packed, ETA 2 days, carrier Foo Logistics."


class LlmCompactor:
    """The reference compactor from the context-window docs page."""

    def __init__(self, summarizer: AgentCore, *, threshold_tokens: int, keep_turns: int = 2) -> None:
        self.summarizer = summarizer
        self.threshold_tokens = threshold_tokens
        self.keep_turns = keep_turns

    async def __call__(self, context: CompactionContext) -> list[CLIverMessage] | None:
        used = context.last_usage.input_tokens if context.last_usage else 0
        if used < self.threshold_tokens:
            return None
        older, recent = _split_at_turns(context.messages, self.keep_turns)
        if not older:
            return None
        response = await self.summarizer.chat(
            "Summarize the earlier part of this agent run for the same assistant. "
            "Keep facts, decisions, tool results that still matter, and open tasks. "
            "Be compact and do not invent.\n\n" + _render(older),
            max_iterations=1,
        )
        if response.usage_total is not None:
            context.record_usage(response.usage_total)
        summary = CLIverMessage(
            role="user",
            content="[Compacted earlier context]\n" + (response.message.content or ""),
        )
        return [summary, *recent]


def _split_at_turns(messages, keep_turns: int):
    user_positions = [i for i, message in enumerate(messages) if message.role == "user"]
    if len(user_positions) <= keep_turns:
        return [], list(messages)
    cut = user_positions[-keep_turns]
    return list(messages[:cut]), list(messages[cut:])


def _render(messages) -> str:
    lines = []
    for message in messages:
        if message.role == "assistant" and message.tool_calls:
            calls = ", ".join(
                f"{call.name}({json.dumps(call.args, ensure_ascii=False)})" for call in message.tool_calls
            )
            lines.append(f"assistant -> tools: {calls}")
        elif message.role == "tool":
            lines.append(f"tool result: {message.content}")
        else:
            lines.append(f"{message.role}: {message.content}")
    return "\n".join(lines)


def _conversation() -> list[CLIverMessage]:
    turns = []
    for index in range(6):
        turns.append(CLIverMessage(role="user", content=f"LEGACY-{index}: status of order A-{index}?"))
        turns.append(CLIverMessage(role="assistant", content=f"Order A-{index} was delivered."))
    return turns


async def test_compactor_shrinks_history_and_counts_summary_usage() -> None:
    skip_unless_keys("DEEPSEEK_API_KEY")
    calls = 0

    async def counting(context: CompactionContext):
        nonlocal calls
        calls += 1
        return await compactor(context)

    compactor = LlmCompactor(AgentCore(ModelBindings(llm=presets.deepseek(temperature=0))), threshold_tokens=1)
    agent = AgentCore(
        ModelBindings(llm=presets.deepseek(temperature=0)),
        builtin_tools=[lookup_order],
        context_compactor=counting,
        run_timeout=180,
    )

    try:
        response = await agent.chat(
            "LEGACY-CURRENT: check order C-300 and tell me its status.",
            system_prompt=SYSTEM,
            conversation=_conversation(),
        )
        trace = response.message.trace or []
        assert response.message.content
        assert calls == len(trace) >= 2
        assert "lookup_order" in tool_names(trace[0])

        first_request = trace[0]["request"]["messages"]
        second_text = json.dumps(trace[1]["request"]["messages"], ensure_ascii=False)
        assert "[Compacted earlier context]" in second_text
        assert "LEGACY-0:" not in second_text
        assert len(trace[1]["request"]["messages"]) < len(first_request)

        assert response.usage is not None and response.usage_total is not None
        assert response.usage_total.total_tokens > response.usage.total_tokens

        before = calls
        await agent.chat(
            "LEGACY-OFF: just say done.",
            system_prompt="Reply with one word.",
            context_compactor=None,
        )
        assert calls == before
    finally:
        await agent.aclose()
        await compactor.summarizer.aclose()


async def test_streaming_uses_the_same_seam() -> None:
    skip_unless_keys("DEEPSEEK_API_KEY")
    stream_calls = 0
    compactor = LlmCompactor(AgentCore(ModelBindings(llm=presets.deepseek(temperature=0))), threshold_tokens=1)

    async def counting(context: CompactionContext):
        nonlocal stream_calls
        stream_calls += 1
        return await compactor(context)

    agent = AgentCore(
        ModelBindings(llm=presets.deepseek(temperature=0)),
        builtin_tools=[lookup_order],
        context_compactor=counting,
        run_timeout=180,
    )

    try:
        chunks = await drain(
            agent.stream(
                "LEGACY-STREAM: check order D-400.",
                system_prompt=SYSTEM,
                conversation=_conversation(),
            )
        )
        text = "".join(chunk.content or "" for chunk in chunks)
        assert text.strip()
        trace = stream_trace(chunks)
        assert stream_calls == len(trace) >= 2
        second_text = json.dumps(trace[1]["request"]["messages"], ensure_ascii=False)
        assert "[Compacted earlier context]" in second_text
        assert trace[-1]["usage_total"] is not None
    finally:
        await agent.aclose()
        await compactor.summarizer.aclose()
