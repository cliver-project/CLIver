"""Per-call event handlers: a call-site handler observes the same tool events.

The tool surface accepts sync and async handlers with the same semantics as
the inference surface: a sync handler runs inline, an async one is awaited,
and a handler exception propagates unchanged — it is never converted into a
tool result.
"""

from __future__ import annotations

import pytest

from cliver_core.agent import AgentCore
from cliver_core.bindings import ModelBinding, ModelBindings
from cliver_core.events import ToolEventType
from cliver_core.messages import ToolCall
from cliver_core.tool import CLIverTool


class _HandlerBoom(RuntimeError):
    pass


def _agent(on_event=None) -> AgentCore:
    return AgentCore(ModelBindings(llm=ModelBinding.llm(model="mock-model")), on_event=on_event)


def _sum_tool() -> CLIverTool:
    return CLIverTool(
        name="sum",
        description="add two numbers",
        parameters={
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
        execute=lambda a, b: [{"value": a + b}],
        safety="safe",
    )


async def test_call_site_event_handler_receives_tool_events_alongside_the_instance_handler():
    instance_events: list = []
    call_events: list = []

    async def instance_handler(event) -> None:
        instance_events.append(event)

    async def call_handler(event) -> None:
        call_events.append(event)

    agent = _agent(on_event=instance_handler)

    await agent._execute_tool(
        ToolCall(id="call-1", name="sum", args={"a": 1, "b": 2}),
        _sum_tool(),
        on_event=call_handler,
    )

    assert [event.event for event in instance_events] == [ToolEventType.START, ToolEventType.END]
    assert [event.event for event in call_events] == [ToolEventType.START, ToolEventType.END]
    assert call_events[0].tool_name == "sum"


async def test_chat_threads_the_call_site_handler_into_the_loop(monkeypatch):
    captured: dict = {}

    async def fake_run_loop(messages, tools, options, max_iterations, **kwargs):
        captured["on_event"] = kwargs.get("on_event")

    async def handler(event) -> None:
        pass

    agent = _agent()
    monkeypatch.setattr(agent, "_run_loop", fake_run_loop)

    await agent.chat("hello", on_event=handler)

    assert captured["on_event"] is handler


async def test_sync_tool_event_handler_is_supported():
    events: list = []

    def handler(event) -> None:
        events.append(event)

    agent = _agent()

    await agent._execute_tool(
        ToolCall(id="call-1", name="sum", args={"a": 1, "b": 2}),
        _sum_tool(),
        on_event=handler,
    )

    assert [event.event for event in events] == [ToolEventType.START, ToolEventType.END]


async def test_tool_handler_exception_on_start_propagates():
    async def handler(event) -> None:
        raise _HandlerBoom("start boom")

    agent = _agent()

    with pytest.raises(_HandlerBoom, match="start boom"):
        await agent._execute_tool(
            ToolCall(id="call-1", name="sum", args={"a": 1, "b": 2}),
            _sum_tool(),
            on_event=handler,
        )


async def test_tool_handler_exception_on_end_propagates():
    # Only a raise from the tool body becomes a tool-error result; a handler
    # raise is a caller bug and surfaces unchanged.
    def handler(event) -> None:
        if event.event is ToolEventType.END:
            raise _HandlerBoom("end boom")

    agent = _agent()

    with pytest.raises(_HandlerBoom, match="end boom"):
        await agent._execute_tool(
            ToolCall(id="call-1", name="sum", args={"a": 1, "b": 2}),
            _sum_tool(),
            on_event=handler,
        )
