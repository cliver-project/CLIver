"""Async ``@tool`` functions execute on the agent's event loop.

They used to be converted to a sync wrapper that called ``asyncio.run()``
inside a worker thread — a fresh, invisible loop per call, not cancellable,
and unable to use any loop-bound resource the agent owns.  They now run on
the agent's loop: ``tool_timeout`` and caller cancellation reach them.
Synchronous tools keep running in the executor, off the loop.
"""

from __future__ import annotations

import asyncio

import pytest

from cliver_core.agent import AgentCore
from cliver_core.bindings import ModelBinding, ModelBindings
from cliver_core.messages import CLIverMessage, ToolCall
from cliver_core.provider import CLIverRequest, CLIverResponse, Provider
from cliver_core.tool import tool


class _ScriptedProvider(Provider):
    protocol = "openai"
    brand = "mock-async-tools"

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


def _tool_turn(call_id: str, name: str) -> CLIverResponse:
    return CLIverResponse(
        message=CLIverMessage(
            role="assistant",
            content=None,
            tool_calls=[ToolCall(id=call_id, name=name, args={})],
        )
    )


def _agent(
    responses: list[CLIverResponse],
    tools: list,
) -> tuple[AgentCore, _ScriptedProvider]:
    provider = _ScriptedProvider(responses)
    bindings = ModelBindings(llm=ModelBinding.llm(model="mock-model", provider=provider))
    return AgentCore(bindings, builtin_tools=tools), provider


def _last_tool_text(request: CLIverRequest) -> str:
    messages = [message for message in request.messages if message.role == "tool"]
    assert messages
    return messages[-1].content or ""


@pytest.mark.asyncio
async def test_async_tool_runs_on_the_agents_loop() -> None:
    seen: dict[str, object] = {}
    agent_loop = asyncio.get_running_loop()

    @tool(name="async_probe", description="probe")
    async def async_probe() -> list[dict]:
        seen["loop"] = asyncio.get_running_loop()
        await asyncio.sleep(0)
        return [{"text": "from async tool"}]

    agent, provider = _agent([_tool_turn("c1", "async_probe"), _final()], [async_probe])

    await agent.chat("go")

    assert seen["loop"] is agent_loop
    assert "from async tool" in _last_tool_text(provider.requests[-1])


@pytest.mark.asyncio
async def test_timeout_cancels_an_async_tool() -> None:
    cancelled = asyncio.Event()

    @tool(name="slow_async", description="slow")
    async def slow_async() -> list[dict]:
        try:
            await asyncio.sleep(0.3)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return [{"text": "too late"}]

    agent, provider = _agent([_tool_turn("c1", "slow_async"), _final()], [slow_async])

    await agent.chat("go", tool_timeout=0.05)

    await asyncio.wait_for(cancelled.wait(), timeout=1)
    assert "timed out" in _last_tool_text(provider.requests[-1])


@pytest.mark.asyncio
async def test_caller_cancellation_reaches_an_async_tool() -> None:
    started = asyncio.Event()
    cancelled = asyncio.Event()

    @tool(name="slow_async", description="slow")
    async def slow_async() -> list[dict]:
        started.set()
        try:
            await asyncio.sleep(1)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return [{"text": "too late"}]

    agent, _provider = _agent([_tool_turn("c1", "slow_async"), _final()], [slow_async])
    task = asyncio.ensure_future(agent.chat("go"))
    await asyncio.wait_for(started.wait(), timeout=1)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    await asyncio.wait_for(cancelled.wait(), timeout=1)


@pytest.mark.asyncio
async def test_sync_tool_still_runs_in_a_worker_thread() -> None:
    seen: dict[str, bool] = {}

    def sync_probe() -> list[dict]:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            seen["on_loop"] = False
        else:
            seen["on_loop"] = True
        return [{"text": "sync ok"}]

    from cliver_core.tool import CLIverTool

    probe = CLIverTool(
        name="sync_probe",
        description="sync probe",
        parameters={"type": "object", "properties": {}},
        execute=sync_probe,
    )
    agent, provider = _agent([_tool_turn("c1", "sync_probe"), _final()], [probe])

    await agent.chat("go")

    assert seen["on_loop"] is False
    assert "sync ok" in _last_tool_text(provider.requests[-1])


def test_async_tool_execute_remains_callable_directly() -> None:
    @tool(name="async_probe", description="probe")
    async def async_probe() -> list[dict]:
        return [{"text": "direct"}]

    assert async_probe.execute() == [{"text": "direct"}]
