"""Run-level deadline: ``run_timeout`` bounds the whole run.

The clock covers the preamble (skill preload, MCP start, selection, message
building), every model call, every tool batch, failover, and event handler
awaits.  Expiry raises :class:`RunTimeoutError` — a ``TimeoutError`` that is
never converted into a provider error or a response.  Caller cancellation
still propagates as ``asyncio.CancelledError``.
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from cliver_core.agent import AgentCore
from cliver_core.bindings import ModelBinding, ModelBindings
from cliver_core.errors import RunTimeoutError
from cliver_core.events import ToolEvent, ToolEventType
from cliver_core.messages import CLIverMessage, CLIverMessageChunk, ToolCall
from cliver_core.provider import CLIverRequest, CLIverResponse, Provider
from cliver_core.skill import Skill
from cliver_core.tool import CLIverTool


class _SlowProvider(Provider):
    protocol = "openai"
    brand = "mock-slow"

    def __init__(self, delay: float = 10.0, text: str = "late") -> None:
        super().__init__(api_key="", base_url="")
        self.delay = delay
        self.text = text
        self.requests: list[CLIverRequest] = []
        self.cancelled = False

    async def chat(self, request: CLIverRequest) -> CLIverResponse:
        self.requests.append(request)
        try:
            await asyncio.sleep(self.delay)
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        return CLIverResponse(message=CLIverMessage(role="assistant", content=self.text))

    async def stream(self, request):  # pragma: no cover
        raise AssertionError("stream() is not used here")


class _ScriptedProvider(Provider):
    protocol = "openai"
    brand = "mock-scripted"

    def __init__(self, responses: list[CLIverResponse]) -> None:
        super().__init__(api_key="", base_url="")
        self.responses = list(responses)
        self.requests: list[CLIverRequest] = []

    async def chat(self, request: CLIverRequest) -> CLIverResponse:
        self.requests.append(request)
        return self.responses.pop(0)

    async def stream(self, request):  # pragma: no cover
        raise AssertionError("stream() is not used here")


class _SlowStreamProvider(Provider):
    """Yields its chunks, then stalls before the terminal native chunk."""

    protocol = "openai"
    brand = "mock-slow-stream"

    def __init__(self, *, delay: float = 10.0, chunks: tuple[str, ...] = ("one", "two")) -> None:
        super().__init__(api_key="", base_url="")
        self.delay = delay
        self.chunks = chunks
        self.requests: list[CLIverRequest] = []
        self.closed = False

    async def chat(self, request: CLIverRequest) -> CLIverResponse:  # pragma: no cover
        raise AssertionError("chat() is not used here")

    async def stream(self, request: CLIverRequest):
        self.requests.append(request)
        try:
            for chunk in self.chunks:
                yield CLIverMessageChunk(content=chunk)
            await asyncio.sleep(self.delay)
            yield CLIverMessageChunk(native={"role": "assistant", "content": "".join(self.chunks)})
        finally:
            self.closed = True


class _StreamingToolProvider(Provider):
    """Turn 1 asks for ``slow``; turn 2 answers with text."""

    protocol = "openai"
    brand = "mock-stream-tool"

    def __init__(self) -> None:
        super().__init__(api_key="", base_url="")
        self.requests: list[CLIverRequest] = []

    async def chat(self, request: CLIverRequest) -> CLIverResponse:  # pragma: no cover
        raise AssertionError("chat() is not used here")

    async def stream(self, request: CLIverRequest):
        self.requests.append(request)
        if len(self.requests) == 1:
            yield CLIverMessageChunk(
                native={
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "slow", "arguments": "{}"}}],
                }
            )
        else:
            yield CLIverMessageChunk(content="done")
            yield CLIverMessageChunk(native={"role": "assistant", "content": "done"})


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


def _agent(provider: Provider, tools: list[CLIverTool] | None = None, **options) -> AgentCore:
    bindings = ModelBindings(llm=ModelBinding.llm(model="mock-model", provider=provider))
    return AgentCore(bindings, builtin_tools=list(tools or []), **options)


def _tool(name: str, execute) -> CLIverTool:
    return CLIverTool(
        name=name,
        description=name,
        parameters={"type": "object", "properties": {}},
        execute=execute,
    )


def _async_sleep_tool(state: dict, seconds: float = 10.0) -> CLIverTool:
    async def execute():
        try:
            await asyncio.sleep(seconds)
        except asyncio.CancelledError:
            state["cancelled"] = True
            raise
        return [{"ok": True}]

    return _tool("slow", execute)


# ── chat ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_chat_raises_run_timeout_when_the_model_stalls():
    provider = _SlowProvider()

    with pytest.raises(RunTimeoutError) as info:
        await _agent(provider).chat("go", run_timeout=0.05, run_id="run-timeout-1")

    assert info.value.timeout == 0.05
    assert info.value.run_id == "run-timeout-1"
    assert "0.05s" in str(info.value)
    assert provider.cancelled is True


@pytest.mark.asyncio
async def test_constructor_run_timeout_is_the_default():
    with pytest.raises(RunTimeoutError):
        await _agent(_SlowProvider(), run_timeout=0.05).chat("go")


@pytest.mark.asyncio
async def test_per_call_run_timeout_overrides_the_constructor_default():
    with pytest.raises(RunTimeoutError):
        await _agent(_SlowProvider(), run_timeout=10.0).chat("go", run_timeout=0.05)


@pytest.mark.asyncio
async def test_per_call_none_disables_the_constructor_timeout():
    response = await _agent(_SlowProvider(delay=0.05), run_timeout=0.01).chat("go", run_timeout=None)

    assert response.message.content == "late"


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [0, -1, True, float("nan"), float("inf")])
async def test_call_site_run_timeout_is_validated(value):
    with pytest.raises(ValueError, match="run_timeout"):
        await _agent(_SlowProvider()).chat("go", run_timeout=value)


def test_constructor_run_timeout_is_validated():
    with pytest.raises(ValueError, match="run_timeout"):
        AgentCore(ModelBindings(llm=ModelBinding.llm(model="mock-model")), run_timeout=0)


@pytest.mark.asyncio
async def test_the_deadline_covers_the_preamble():
    provider = _SlowProvider()
    agent = _agent(provider)

    async def slow_preload() -> None:
        await asyncio.sleep(10)

    agent._preload_skills = slow_preload  # type: ignore[method-assign]

    with pytest.raises(RunTimeoutError):
        await agent.chat("go", run_timeout=0.05)

    assert provider.requests == []


@pytest.mark.asyncio
async def test_external_cancellation_is_not_a_run_timeout():
    task = asyncio.create_task(_agent(_SlowProvider()).chat("go", run_timeout=10))
    await asyncio.sleep(0.01)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_sync_tool_thread_finishes_after_the_run_times_out():
    finished = threading.Event()

    def slow_tool() -> list[dict]:
        time.sleep(0.2)
        finished.set()
        return [{"ok": True}]

    provider = _ScriptedProvider([_tool_turn(("c1", "slow")), _final("never")])
    agent = _agent(provider, [_tool("slow", slow_tool)])

    with pytest.raises(RunTimeoutError):
        await agent.chat("go", run_timeout=0.05)

    assert finished.wait(2)


# ── run_skill ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_skill_is_bounded_and_uses_one_clock(tmp_path):
    agent = _agent(_SlowProvider())
    captured: dict = {}
    original_chat = agent.chat

    async def capturing_chat(*args, **kwargs):
        captured.update(kwargs)
        return await original_chat(*args, **kwargs)

    agent.chat = capturing_chat  # type: ignore[method-assign]
    skill = Skill(name="greet", description="greet", body="Say hello.", base_dir=tmp_path)

    with pytest.raises(RunTimeoutError) as info:
        await agent.run_skill(skill, "go", run_timeout=0.05, run_id="skill-deadline")

    assert info.value.run_id == "skill-deadline"
    # The inner chat must not restart the clock.
    assert captured["run_timeout"] is None
    assert captured["run_id"] == "skill-deadline"


# ── stream ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_stream_yields_earlier_chunks_then_times_out():
    provider = _SlowStreamProvider(delay=10)
    chunks: list[CLIverMessageChunk] = []

    with pytest.raises(RunTimeoutError) as info:
        async for chunk in _agent(provider).stream("go", run_timeout=0.05, run_id="stream-deadline"):
            chunks.append(chunk)

    assert [chunk.content for chunk in chunks] == ["one", "two"]
    assert info.value.run_id == "stream-deadline"
    assert provider.closed is True


@pytest.mark.asyncio
async def test_stream_deadline_starts_at_the_first_pull():
    agent = _agent(_SlowStreamProvider())

    async def slow_preload() -> None:
        await asyncio.sleep(10)

    agent._preload_skills = slow_preload  # type: ignore[method-assign]
    stream = agent.stream("go", run_timeout=0.05)

    with pytest.raises(RunTimeoutError):
        await anext(stream)


@pytest.mark.asyncio
async def test_consumer_parking_counts_toward_the_deadline():
    provider = _SlowStreamProvider(delay=0)
    stream = _agent(provider).stream("go", run_timeout=0.05)

    first = await anext(stream)
    assert first.content == "one"
    await asyncio.sleep(0.1)  # parked longer than the budget

    with pytest.raises(RunTimeoutError):
        await anext(stream)
    assert provider.closed is True


@pytest.mark.asyncio
async def test_abandoning_the_stream_closes_the_provider_iterator():
    provider = _SlowStreamProvider(delay=0)
    stream = _agent(provider).stream("go")

    first = await anext(stream)
    assert first.content == "one"
    await stream.aclose()

    assert provider.closed is True


@pytest.mark.asyncio
async def test_stream_tool_batch_is_bounded_by_the_run_deadline():
    state: dict = {}
    agent = _agent(_StreamingToolProvider(), [_async_sleep_tool(state)])

    with pytest.raises(RunTimeoutError):
        async for _chunk in agent.stream("go", run_timeout=0.05):
            pass

    assert state.get("cancelled") is True


@pytest.mark.asyncio
async def test_stream_forwards_parallel_and_tool_timeout_to_the_batch():
    captured: dict = {}

    async def fake_execute(
        messages,
        tool_calls,
        consecutive_errors,
        consecutive_denials,
        tools,
        *,
        parallel=False,
        tool_timeout=None,
        on_event=None,
        run_id=None,
    ):
        captured.update(parallel=parallel, tool_timeout=tool_timeout, run_id=run_id)
        return 0, 0, False

    agent = _agent(_StreamingToolProvider(), [])
    agent._execute_tool_calls = fake_execute  # type: ignore[method-assign]

    chunks = [
        chunk
        async for chunk in agent.stream("go", parallel_tool_calls=True, tool_timeout=0.05, run_id="stream-forward")
    ]

    assert captured == {"parallel": True, "tool_timeout": 0.05, "run_id": "stream-forward"}
    assert any(chunk.content == "done" for chunk in chunks)


@pytest.mark.asyncio
async def test_stream_tool_timeout_still_applies_per_tool():
    events: list[ToolEvent] = []

    async def handler(event) -> None:
        events.append(event)

    def slow_tool() -> list[dict]:
        time.sleep(0.2)
        return [{"ok": True}]

    agent = _agent(_StreamingToolProvider(), [_tool("slow", slow_tool)])

    chunks = [chunk async for chunk in agent.stream("go", tool_timeout=0.05, on_event=handler)]

    errors = [event for event in events if event.event is ToolEventType.ERROR]
    assert len(errors) == 1
    assert "timed out after 0.05s" in errors[0].error
    assert "".join(chunk.content or "" for chunk in chunks) == "done"
