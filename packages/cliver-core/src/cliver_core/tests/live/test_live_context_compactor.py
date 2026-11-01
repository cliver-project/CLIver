"""Live: the context-window seam with the built-in plan compactor.

A real DeepSeek summarizer compacts a long legacy conversation between Re-Act
iterations.  The tests prove the materialized replacement history reaches the
wire, the summary model's usage is accounted for, full-history returns still
work, and a per-call ``None`` disables the constructor compactor.
"""

from __future__ import annotations

import json

import pytest

from cliver_core import (
    AgentCore,
    CLIverMessage,
    CompactionContext,
    LlmCompactor,
    ModelBindings,
    UsageInfo,
    presets,
)
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


def _conversation() -> list[CLIverMessage]:
    turns = []
    for index in range(6):
        turns.append(CLIverMessage(role="user", content=f"LEGACY-{index}: status of order A-{index}?"))
        turns.append(CLIverMessage(role="assistant", content=f"Order A-{index} was delivered."))
    return turns


def _second_request_text(trace: list[dict]) -> str:
    return json.dumps(trace[1]["request"]["messages"], ensure_ascii=False)


def _assert_plan_replaced_the_prefix(trace: list[dict], first_request: list[dict]) -> None:
    """The second request carries a summary instead of the exact legacy messages."""
    second = trace[1]["request"]["messages"]
    non_system = [message for message in second if message["role"] != "system"]
    assert non_system[0]["content"].startswith("[Compacted earlier context]")
    legacy_user = next(message for message in first_request if message["content"].startswith("LEGACY-0:"))
    assert all(message.get("content") != legacy_user["content"] for message in second)
    assert len(second) < len(first_request)
    # The retained suffix still ends with the tool result of the current turn.
    assert second[-1]["role"] == "tool"


async def test_plan_path_shrinks_history_and_counts_summary_usage() -> None:
    skip_unless_keys("DEEPSEEK_API_KEY")
    calls = 0
    summarizer = AgentCore(ModelBindings(llm=presets.deepseek(temperature=0)))
    compactor = LlmCompactor(summarizer, threshold_tokens=1)

    async def counting(context: CompactionContext):
        nonlocal calls
        calls += 1
        return await compactor(context)

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
        _assert_plan_replaced_the_prefix(trace, first_request)
        assert "[Compacted earlier context]" in _second_request_text(trace)

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
        await summarizer.aclose()


async def test_escape_hatch_full_history_is_still_supported() -> None:
    skip_unless_keys("DEEPSEEK_API_KEY")
    calls = 0

    async def trimmer(context: CompactionContext):
        nonlocal calls
        calls += 1
        if context.iteration == 0:
            return None
        split = context.split_at_user_turn(1)
        context.record_usage(UsageInfo(input_tokens=7, output_tokens=3))
        return [
            CLIverMessage(
                role="user",
                content="[Compacted earlier context]\nEarlier turns were trimmed; order A-5 was delivered.",
            ),
            *split.recent,
        ]

    agent = AgentCore(
        ModelBindings(llm=presets.deepseek(temperature=0)),
        builtin_tools=[lookup_order],
        context_compactor=trimmer,
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

        first_request = trace[0]["request"]["messages"]
        _assert_plan_replaced_the_prefix(trace, first_request)

        assert response.usage is not None and response.usage_total is not None
        # The escape hatch records its own usage; 7 + 3 must be in the total.
        assert response.usage_total.total_tokens - response.usage.total_tokens >= 10
    finally:
        await agent.aclose()


async def test_streaming_uses_the_same_seam() -> None:
    skip_unless_keys("DEEPSEEK_API_KEY")
    stream_calls = 0

    summarizer = AgentCore(ModelBindings(llm=presets.deepseek(temperature=0)))
    compactor = LlmCompactor(summarizer, threshold_tokens=1)

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
        _assert_plan_replaced_the_prefix(trace, trace[0]["request"]["messages"])
        assert trace[-1]["usage_total"] is not None
        assert trace[-1]["usage_total"]["input_tokens"] > trace[-1]["usage"]["input_tokens"]
    finally:
        await agent.aclose()
        await summarizer.aclose()
