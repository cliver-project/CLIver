"""Live: one AgentCore serving chat, stream, run_skill, and concurrent runs.

Proves the Re-Act loop, events, usage accounting, run ids, per-call options,
and skill tool narrowing against a real provider (DeepSeek).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from cliver_core import AgentCore, ModelBindings, SkillManager, presets
from cliver_core.events import InferenceEventType, ToolEventType
from cliver_core.tests.live.support import (
    LIVE_OFF_REASON,
    EventCollector,
    drain,
    live_enabled,
    skip_unless_keys,
    stream_text,
    stream_trace,
    tool_names,
)
from cliver_core.tool import tool

pytestmark = pytest.mark.skipif(not live_enabled(), reason=LIVE_OFF_REASON)

ORDER_SYSTEM = (
    "You are an order assistant. For any order question you must call lookup_order "
    "exactly once with the order id, then answer in one short sentence."
)


@tool(name="lookup_order", description="Return the packed status, ETA, and carrier for one order id.")
def lookup_order(order_id: str) -> str:
    return f"Order {order_id}: packed, ETA 2 days, carrier Foo Logistics."


@tool(name="convert_currency", description="Convert an amount from one currency to another.")
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    return f"{amount} {from_currency} = {amount} {to_currency} (live test rate)"


def _write_skills(tmp_path: Path) -> Path:
    skill_dir = tmp_path / "skills" / "order-brief"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: order-brief\n"
        "description: Use when the user asks for a short order status report.\n"
        "allowed-tools: lookup_order\n"
        "---\n\n"
        "# Order Brief\n\n"
        "Call `lookup_order` for the requested order id, then answer with one short sentence.\n",
        encoding="utf-8",
    )
    return tmp_path / "skills"


def _build_agent(tmp_path: Path, tool_events: EventCollector, inference_events: EventCollector) -> AgentCore:
    binding = presets.deepseek(temperature=0, on_event=inference_events)
    return AgentCore(
        ModelBindings(llm=binding),
        builtin_tools=[lookup_order, convert_currency],
        skill_manager=SkillManager([_write_skills(tmp_path)]),
        on_event=tool_events,
        run_timeout=180,
    )


def _assert_usage(response) -> None:
    assert response.usage is not None
    assert response.usage_total is not None
    assert response.usage_total.total_tokens >= response.usage.total_tokens


async def test_one_agent_serves_chat_stream_and_skill(tmp_path: Path) -> None:
    skip_unless_keys("DEEPSEEK_API_KEY")
    tool_events = EventCollector()
    inference_events = EventCollector()
    agent = _build_agent(tmp_path, tool_events, inference_events)

    try:
        chat = await agent.chat(
            "Where is order A-100? Call lookup_order exactly once with A-100.",
            system_prompt=ORDER_SYSTEM,
            run_id="live-chat",
        )
        assert chat.message.content
        assert chat.run_id == "live-chat"
        _assert_usage(chat)
        assert chat.usage_total.total_tokens > chat.usage.total_tokens
        chat_trace = chat.message.trace or []
        assert len(chat_trace) >= 2
        assert "lookup_order" in tool_names(chat_trace[0])
        # Binding params (temperature=0) reach the provider request.
        assert chat_trace[0]["request"]["options"].get("temperature") == 0

        chat_tool_events = [e for e in tool_events.for_run("live-chat") if e.tool_name == "lookup_order"]
        assert {e.event for e in chat_tool_events} == {ToolEventType.START, ToolEventType.END}
        chat_inference = inference_events.for_run("live-chat")
        completed = [e for e in chat_inference if e.event == InferenceEventType.COMPLETED]
        assert len(completed) >= 2
        assert all(e.data and e.data.get("usage") for e in completed)

        chunks = await drain(
            agent.stream(
                "Where is order B-200? Call lookup_order exactly once with B-200.",
                system_prompt=ORDER_SYSTEM,
                run_id="live-stream",
            )
        )
        assert stream_text(chunks).strip()
        stream_trace_entries = stream_trace(chunks)
        assert len(stream_trace_entries) >= 2
        assert stream_trace_entries[-1]["usage_total"] is not None
        trace_chunk = next(chunk for chunk in chunks if "__llm_trace__" in chunk.vendor_ext)
        assert trace_chunk.run_id == "live-stream"
        stream_tool_events = [e for e in tool_events.for_run("live-stream") if e.tool_name == "lookup_order"]
        assert {e.event for e in stream_tool_events} == {ToolEventType.START, ToolEventType.END}
        stream_inference = inference_events.for_run("live-stream")
        assert any(e.event == InferenceEventType.TOOL_CALL for e in stream_inference)

        skill = await agent.run_skill(
            "order-brief",
            "Brief status for order C-300.",
            run_id="live-skill",
        )
        assert skill.message.content
        assert skill.run_id == "live-skill"
        skill_trace = skill.message.trace or []
        advertised = tool_names(skill_trace[0])
        assert "lookup_order" in advertised
        assert "convert_currency" not in advertised
        skill_tool_names = {e.tool_name for e in tool_events.for_run("live-skill")}
        assert "lookup_order" in skill_tool_names
        assert "convert_currency" not in skill_tool_names
        assert inference_events.for_run("live-skill")

        assert len({chat.run_id, skill.run_id}) == 2
    finally:
        await agent.aclose()


async def test_concurrent_runs_on_one_agent_keep_events_isolated(tmp_path: Path) -> None:
    skip_unless_keys("DEEPSEEK_API_KEY")
    tool_events = EventCollector()
    inference_events = EventCollector()
    agent = _build_agent(tmp_path, tool_events, inference_events)
    run_ids = {"live-concurrent-1", "live-concurrent-2", "live-concurrent-3"}

    try:
        first, second, stream_chunks = await asyncio.gather(
            agent.chat(
                "Where is order D-400? Call lookup_order exactly once with D-400.",
                system_prompt=ORDER_SYSTEM,
                run_id="live-concurrent-1",
                options={"temperature": 0.2},
            ),
            agent.chat(
                "Where is order E-500? Call lookup_order exactly once with E-500.",
                system_prompt=ORDER_SYSTEM,
                run_id="live-concurrent-2",
            ),
            drain(
                agent.stream(
                    "Where is order F-600? Call lookup_order exactly once with F-600.",
                    system_prompt=ORDER_SYSTEM,
                    run_id="live-concurrent-3",
                )
            ),
        )
        assert first.message.content and second.message.content
        assert stream_text(stream_chunks).strip()
        _assert_usage(first)
        _assert_usage(second)
        # The call-site option wins over the binding default (temperature=0).
        assert first.message.trace[0]["request"]["options"].get("temperature") == 0.2

        for run_id in run_ids:
            assert inference_events.for_run(run_id), f"no inference events for {run_id}"
            per_run_tools = tool_events.for_run(run_id)
            assert per_run_tools, f"no tool events for {run_id}"
            assert {event.tool_name for event in per_run_tools} == {"lookup_order"}
        assert inference_events.run_ids() == run_ids
        assert tool_events.run_ids() == run_ids
    finally:
        await agent.aclose()
