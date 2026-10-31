"""Parallel tool execution and per-tool timeout/cancellation.

Parallel mode is opt-in: permission checks stay sequential in tool-call
order, executions run concurrently, and results are appended in the original
order.  A timeout is a normal tool-error result plus a ``tool.error`` event;
it wraps execution only (never the permission prompt), applies to MCP calls,
and caller cancellation propagates unchanged.
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from cliver_core.agent import AgentCore
from cliver_core.bindings import ModelBinding, ModelBindings
from cliver_core.events import ToolEvent, ToolEventType
from cliver_core.messages import CLIverMessage, ToolCall
from cliver_core.policy import Decision
from cliver_core.provider import CLIverRequest, CLIverResponse, Provider
from cliver_core.tool import CLIverTool


class _ScriptedProvider(Provider):
    protocol = "openai"
    brand = "mock-tool-exec"

    def __init__(self, responses: list[CLIverResponse]) -> None:
        super().__init__(api_key="", base_url="")
        self.responses = list(responses)
        self.requests: list[CLIverRequest] = []

    async def chat(self, request: CLIverRequest) -> CLIverResponse:
        self.requests.append(request)
        return self.responses.pop(0)

    async def stream(self, request):  # pragma: no cover
        raise AssertionError("stream() is not used here")


def _final(text: str = "done") -> CLIverResponse:
    return CLIverResponse(message=CLIverMessage(role="assistant", content=text))


def _tool_turn(*calls: tuple[str, str]) -> CLIverResponse:
    return CLIverResponse(
        message=CLIverMessage(
            role="assistant",
            content=None,
            tool_calls=[ToolCall(id=call_id, name=name, args={}) for call_id, name in calls],
        )
    )


def _agent(
    responses: list[CLIverResponse],
    tools: list[CLIverTool] | None = None,
    **options,
) -> tuple[AgentCore, _ScriptedProvider]:
    provider = _ScriptedProvider(responses)
    bindings = ModelBindings(llm=ModelBinding.llm(model="mock-model", provider=provider))
    return AgentCore(bindings, builtin_tools=list(tools or []), **options), provider


def _tool(name: str, execute, *, safety: str = "safe") -> CLIverTool:
    return CLIverTool(
        name=name,
        description=name,
        parameters={"type": "object", "properties": {}},
        execute=execute,
        safety=safety,
    )


def _timed_tool(name: str, log: list, lock: threading.Lock, delay: float) -> CLIverTool:
    def execute() -> list[dict]:
        with lock:
            log.append(("start", name, time.monotonic()))
        time.sleep(delay)
        with lock:
            log.append(("end", name, time.monotonic()))
        return [{"name": name}]

    return _tool(name, execute)


def _sleep_tool(name: str, seconds: float) -> CLIverTool:
    def execute() -> list[dict]:
        time.sleep(seconds)
        return [{"slept": seconds}]

    return _tool(name, execute)


def _tool_messages(request: CLIverRequest) -> list[CLIverMessage]:
    return [message for message in request.messages if message.role == "tool"]


# ── Parallel execution ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_parallel_tool_calls_overlap_and_keep_the_call_order() -> None:
    log: list = []
    lock = threading.Lock()
    agent, provider = _agent(
        [_tool_turn(("c1", "alpha"), ("c2", "beta")), _final()],
        [_timed_tool("alpha", log, lock, 0.2), _timed_tool("beta", log, lock, 0.2)],
    )

    response = await agent.chat("go", parallel_tool_calls=True)

    assert response.message.content == "done"
    starts = [entry for entry in log if entry[0] == "start"]
    ends = [entry for entry in log if entry[0] == "end"]
    assert len(starts) == len(ends) == 2
    # Both tools started before either one finished.
    assert max(start[2] for start in starts) < min(end[2] for end in ends)
    # Results reach the model in the original tool-call order.
    assert [message.tool_call_id for message in _tool_messages(provider.requests[1])] == ["c1", "c2"]


@pytest.mark.asyncio
async def test_tool_calls_stay_sequential_by_default() -> None:
    log: list = []
    lock = threading.Lock()
    agent, _ = _agent(
        [_tool_turn(("c1", "alpha"), ("c2", "beta")), _final()],
        [_timed_tool("alpha", log, lock, 0.05), _timed_tool("beta", log, lock, 0.05)],
    )

    await agent.chat("go")

    assert [(entry[0], entry[1]) for entry in log] == [
        ("start", "alpha"),
        ("end", "alpha"),
        ("start", "beta"),
        ("end", "beta"),
    ]


@pytest.mark.asyncio
async def test_parallel_mode_checks_permissions_one_at_a_time_in_call_order() -> None:
    order: list[str] = []
    violations: list[str] = []
    in_flight = 0
    executed: list[str] = []

    async def gate(context) -> Decision:
        nonlocal in_flight
        in_flight += 1
        if in_flight > 1:
            violations.append(context.tool_name)
        await asyncio.sleep(0.02)
        order.append(context.tool_name)
        in_flight -= 1
        return Decision.allow()

    def make(name: str) -> CLIverTool:
        def execute() -> list[dict]:
            executed.append(name)
            return [{"name": name}]

        return _tool(name, execute, safety="ask")

    agent, _ = _agent(
        [_tool_turn(("c1", "alpha"), ("c2", "beta")), _final()],
        [make("alpha"), make("beta")],
        permission_callback=gate,
    )

    await agent.chat("go", parallel_tool_calls=True)

    assert order == ["alpha", "beta"]
    assert violations == []
    assert sorted(executed) == ["alpha", "beta"]


@pytest.mark.asyncio
async def test_parallel_errors_stay_in_their_own_result_slot() -> None:
    def boom() -> list[dict]:
        raise RuntimeError("kaboom")

    agent, provider = _agent(
        [_tool_turn(("c1", "boom"), ("c2", "fine")), _final()],
        [_tool("boom", boom), _tool("fine", lambda: [{"ok": True}])],
    )

    await agent.chat("go", parallel_tool_calls=True)

    messages = _tool_messages(provider.requests[1])
    assert [message.tool_call_id for message in messages] == ["c1", "c2"]
    assert "kaboom" in messages[0].content
    assert "ok" in messages[1].content


# ── Timeout ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_tool_timeout_returns_an_error_result_and_emits_tool_error() -> None:
    events: list[ToolEvent] = []

    async def handler(event) -> None:
        events.append(event)

    agent, provider = _agent(
        [_tool_turn(("c1", "slow")), _final()],
        [_sleep_tool("slow", 0.5)],
        tool_timeout=0.05,
    )

    response = await agent.chat("go", on_event=handler)

    assert response.message.content == "done"
    errors = [event for event in events if event.event is ToolEventType.ERROR]
    assert len(errors) == 1
    assert "timed out after 0.05s" in errors[0].error
    assert errors[0].duration_ms < 400
    assert [event.event for event in events] == [ToolEventType.START, ToolEventType.ERROR]
    tool_message = _tool_messages(provider.requests[1])[0]
    assert "timed out after 0.05s" in tool_message.content


@pytest.mark.asyncio
async def test_per_call_timeout_overrides_the_constructor_default() -> None:
    events: list[ToolEvent] = []

    async def handler(event) -> None:
        events.append(event)

    agent, _ = _agent(
        [_tool_turn(("c1", "slow")), _final()],
        [_sleep_tool("slow", 0.5)],
        tool_timeout=10.0,
    )

    await agent.chat("go", tool_timeout=0.05, on_event=handler)

    assert [event.event for event in events] == [ToolEventType.START, ToolEventType.ERROR]


@pytest.mark.asyncio
async def test_per_call_none_disables_the_constructor_timeout() -> None:
    events: list[ToolEvent] = []

    async def handler(event) -> None:
        events.append(event)

    agent, provider = _agent(
        [_tool_turn(("c1", "slow")), _final()],
        [_sleep_tool("slow", 0.1)],
        tool_timeout=0.01,
    )

    await agent.chat("go", tool_timeout=None, on_event=handler)

    assert [event.event for event in events] == [ToolEventType.START, ToolEventType.END]
    assert "slept" in _tool_messages(provider.requests[1])[0].content


@pytest.mark.asyncio
async def test_parallel_mode_applies_the_timeout_per_tool() -> None:
    agent, provider = _agent(
        [_tool_turn(("c1", "slow"), ("c2", "fast")), _final()],
        [_sleep_tool("slow", 0.5), _tool("fast", lambda: [{"ok": True}])],
        tool_timeout=0.05,
    )

    await agent.chat("go", parallel_tool_calls=True)

    messages = _tool_messages(provider.requests[1])
    assert "timed out after 0.05s" in messages[0].content
    assert "ok" in messages[1].content


@pytest.mark.asyncio
async def test_timeout_does_not_time_out_the_permission_prompt() -> None:
    events: list[ToolEvent] = []

    async def handler(event) -> None:
        events.append(event)

    async def gate(context) -> Decision:
        await asyncio.sleep(0.1)  # longer than the tool timeout
        return Decision.allow()

    agent, _ = _agent(
        [_tool_turn(("c1", "quick")), _final()],
        [_tool("quick", lambda: [{"ok": True}], safety="ask")],
        tool_timeout=0.05,
        permission_callback=gate,
    )

    await agent.chat("go", on_event=handler)

    assert ToolEventType.ERROR not in [event.event for event in events]
    assert ToolEventType.END in [event.event for event in events]


def test_tool_timeout_must_be_positive() -> None:
    with pytest.raises(ValueError, match="tool_timeout"):
        AgentCore(
            ModelBindings(llm=ModelBinding.llm(model="mock-model")),
            tool_timeout=0,
        )


@pytest.mark.asyncio
async def test_call_site_tool_timeout_is_validated() -> None:
    agent, _ = _agent([_final()], [])

    with pytest.raises(ValueError, match="tool_timeout"):
        await agent.chat("go", tool_timeout=-1)


# ── MCP + cancellation ─────────────────────────────────────────


class _SlowMCPClient:
    def __init__(self) -> None:
        self.cancelled = False

    async def call_tool(self, name: str, args: dict) -> list[dict]:
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        return [{"ok": True}]  # pragma: no cover


@pytest.mark.asyncio
async def test_timeout_cancels_an_mcp_call_and_returns_an_error_result() -> None:
    events: list[ToolEvent] = []

    async def handler(event) -> None:
        events.append(event)

    client = _SlowMCPClient()
    agent = AgentCore(
        ModelBindings(llm=ModelBinding.llm(model="mock-model", provider=_ScriptedProvider([_final()]))),
        mcp_client=client,
    )

    slow_tool = CLIverTool(
        name="mcp__srv__slow",
        description="slow MCP tool",
        parameters={"type": "object", "properties": {}},
        execute=lambda **_kwargs: [{"ok": True}],
        origin="mcp",
    )
    result = await agent._execute_tool(
        ToolCall(id="c1", name="mcp__srv__slow", args={}),
        slow_tool,
        tool_timeout=0.05,
        on_event=handler,
    )

    assert "timed out after 0.05s" in result[0]["error"]
    assert client.cancelled is True
    assert [event.event for event in events] == [ToolEventType.START, ToolEventType.ERROR]


class _BlockingMCPClient:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.cancelled = False

    async def start(self) -> None:
        pass

    async def get_tools(self, servers=None) -> list:
        return [
            CLIverTool(
                name="mcp__srv__slow",
                description="blocking MCP tool",
                parameters={"type": "object", "properties": {}},
                execute=lambda **_kwargs: [{"ok": True}],
                origin="mcp",
            )
        ]

    async def call_tool(self, name: str, args: dict) -> list[dict]:
        self.started.set()
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        return [{"ok": True}]  # pragma: no cover


@pytest.mark.asyncio
async def test_cancelling_the_chat_cancels_the_running_tool() -> None:
    client = _BlockingMCPClient()
    agent = AgentCore(
        ModelBindings(
            llm=ModelBinding.llm(
                model="mock-model",
                provider=_ScriptedProvider([_tool_turn(("c1", "mcp__srv__slow")), _final("never")]),
            )
        ),
        mcp_client=client,
    )

    task = asyncio.create_task(agent.chat("go"))
    await asyncio.wait_for(client.started.wait(), timeout=5)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert client.cancelled is True
