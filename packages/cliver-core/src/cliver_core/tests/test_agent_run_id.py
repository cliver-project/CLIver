"""Run correlation (``run_id``): one id per logical run.

``run_id`` covers a whole run — every model call, every tool batch (serial or
parallel), and every event.  ``request_id`` keeps its meaning: one id per
model call, emitted by the protocol providers.  A caller may supply the id;
otherwise the public entry point generates it once.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cliver_core.agent import AgentCore
from cliver_core.bindings import ModelBinding, ModelBindings
from cliver_core.events import InferenceEvent, PipelineEventType, ToolEvent, ToolEventType
from cliver_core.messages import CLIverMessage, CLIverMessageChunk, ToolCall
from cliver_core.pipeline import PipelineStep, SkillPipeline
from cliver_core.provider import CLIverRequest, CLIverResponse, Provider
from cliver_core.provider.providers.anthropic import AnthropicProvider
from cliver_core.provider.providers.openai import OpenAIProvider
from cliver_core.skill import Skill
from cliver_core.tool import CLIverTool


class _ScriptedProvider(Provider):
    protocol = "openai"
    brand = "mock-run-id"

    def __init__(self, responses: list[CLIverResponse]) -> None:
        super().__init__(api_key="", base_url="")
        self.responses = list(responses)
        self.requests: list[CLIverRequest] = []

    async def chat(self, request: CLIverRequest) -> CLIverResponse:
        self.requests.append(request)
        return self.responses.pop(0)

    async def stream(self, request):  # pragma: no cover
        raise AssertionError("stream() is not used here")


class _StreamingProvider(Provider):
    """One turn: a text answer.  Reused by the stream correlation tests."""

    protocol = "openai"
    brand = "mock-run-stream"

    def __init__(self) -> None:
        super().__init__(api_key="", base_url="")
        self.requests: list[CLIverRequest] = []

    async def chat(self, request: CLIverRequest) -> CLIverResponse:  # pragma: no cover
        raise AssertionError("chat() is not used here")

    async def stream(self, request: CLIverRequest):
        self.requests.append(request)
        yield CLIverMessageChunk(content="Hel")
        yield CLIverMessageChunk(content="lo")
        yield CLIverMessageChunk(native={"role": "assistant", "content": "Hello"})


class _StreamingToolProvider(Provider):
    """One streamed turn that asks for a tool; the next turn never runs."""

    protocol = "openai"
    brand = "mock-run-stream-tool"

    def __init__(self) -> None:
        super().__init__(api_key="", base_url="")
        self.requests: list[CLIverRequest] = []

    async def chat(self, request: CLIverRequest) -> CLIverResponse:  # pragma: no cover
        raise AssertionError("chat() is not used here")

    async def stream(self, request: CLIverRequest):
        self.requests.append(request)
        yield CLIverMessageChunk(
            native={
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "echo", "arguments": "{}"}}],
            }
        )


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


def _tool(name: str, execute=None) -> CLIverTool:
    return CLIverTool(
        name=name,
        description=name,
        parameters={"type": "object", "properties": {}},
        execute=execute or (lambda: [{"ok": True}]),
    )


def _agent(
    responses: list[CLIverResponse] | None = None,
    tools: list[CLIverTool] | None = None,
    provider: Provider | None = None,
    **options,
) -> tuple[AgentCore, Provider]:
    provider = provider or _ScriptedProvider(responses or [_final()])
    bindings = ModelBindings(llm=ModelBinding.llm(model="mock-model", provider=provider))
    return AgentCore(bindings, builtin_tools=list(tools or []), **options), provider


class _FakeAgent:
    """Records the chat/run_skill kwargs the pipeline's default runner passes."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def chat(self, user_input: str, **kwargs) -> CLIverResponse:
        self.calls.append({"input": user_input, **kwargs})
        return CLIverResponse(message=CLIverMessage(role="assistant", content="ok"))


# ── chat ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_chat_generates_one_run_id_for_the_response_and_its_events():
    events: list[ToolEvent] = []

    async def handler(event) -> None:
        events.append(event)

    agent, provider = _agent([_tool_turn(("c1", "echo")), _final()], [_tool("echo")])

    response = await agent.chat("go", on_event=handler)

    assert isinstance(response.run_id, str)
    assert len(response.run_id) == 32
    int(response.run_id, 16)  # hex
    assert [request.run_id for request in provider.requests] == [response.run_id, response.run_id]
    assert events
    assert all(event.run_id == response.run_id for event in events)


@pytest.mark.asyncio
async def test_chat_echoes_a_caller_supplied_run_id_everywhere():
    events: list[ToolEvent] = []

    async def handler(event) -> None:
        events.append(event)

    agent, provider = _agent([_tool_turn(("c1", "echo")), _final()], [_tool("echo")])

    response = await agent.chat("go", run_id="run-abc", on_event=handler)

    assert response.run_id == "run-abc"
    assert provider.requests[0].run_id == "run-abc"
    assert all(event.run_id == "run-abc" for event in events)


@pytest.mark.asyncio
@pytest.mark.parametrize("run_id", ["", "   ", 123])
async def test_chat_rejects_an_invalid_run_id(run_id) -> None:
    agent, _ = _agent()

    with pytest.raises(ValueError, match="run_id"):
        await agent.chat("go", run_id=run_id)


@pytest.mark.asyncio
async def test_parallel_tool_events_all_carry_the_run_id():
    events: list[ToolEvent] = []

    async def handler(event) -> None:
        events.append(event)

    agent, _ = _agent(
        [_tool_turn(("c1", "alpha"), ("c2", "beta")), _final()],
        [_tool("alpha"), _tool("beta")],
    )

    response = await agent.chat("go", parallel_tool_calls=True, on_event=handler)

    assert {event.run_id for event in events} == {response.run_id}
    assert len([event for event in events if event.event is ToolEventType.START]) == 2


# ── run_skill ──────────────────────────────────────────────────


def _skill(tmp_path: Path) -> Skill:
    return Skill(name="greet", description="greet", body="Say hello.", base_dir=tmp_path)


@pytest.mark.asyncio
async def test_run_skill_generates_one_run_id_and_pins_it_on_the_inner_chat(tmp_path):
    agent, provider = _agent([_final()])

    response = await agent.run_skill(_skill(tmp_path), "go")

    assert len(response.run_id) == 32
    assert provider.requests[0].run_id == response.run_id


@pytest.mark.asyncio
async def test_run_skill_forwards_a_caller_supplied_run_id_to_the_inner_chat(tmp_path):
    agent, provider = _agent([_final()])

    response = await agent.run_skill(_skill(tmp_path), "go", run_id="skill-run")

    assert response.run_id == "skill-run"
    assert provider.requests[0].run_id == "skill-run"


# ── stream ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_stream_marks_terminal_chunks_and_keeps_deltas_unmarked():
    provider = _StreamingProvider()
    agent, _ = _agent(provider=provider)

    chunks = [chunk async for chunk in agent.stream("go", run_id="stream-1")]

    assert provider.requests[0].run_id == "stream-1"
    trace_chunks = [chunk for chunk in chunks if "__llm_trace__" in chunk.vendor_ext]
    assert len(trace_chunks) == 1
    assert trace_chunks[0].run_id == "stream-1"
    deltas = [chunk for chunk in chunks if "__llm_trace__" not in chunk.vendor_ext]
    assert deltas
    assert all(chunk.run_id is None for chunk in deltas)


@pytest.mark.asyncio
async def test_stream_iteration_limit_chunks_carry_the_run_id():
    provider = _StreamingToolProvider()
    agent, _ = _agent([_tool("echo")], provider=provider)

    chunks = [chunk async for chunk in agent.stream("go", max_iterations=1, run_id="limit-1")]

    terminal = [chunk for chunk in chunks if chunk.content or "__llm_trace__" in chunk.vendor_ext][-2:]
    assert [chunk.content for chunk in terminal] == [
        "Reached maximum iterations without a final answer.",
        None,
    ]
    assert all(chunk.run_id == "limit-1" for chunk in terminal)


@pytest.mark.asyncio
async def test_stream_validates_a_supplied_run_id_on_first_pull():
    agent, _ = _agent()

    with pytest.raises(ValueError, match="run_id"):
        async for _chunk in agent.stream("go", run_id="  "):
            pass


# ── pipeline ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_pipeline_run_id_flows_to_events_results_and_the_default_runner():
    events: list = []

    async def handler(event) -> None:
        events.append(event)

    agent = _FakeAgent()
    pipeline = SkillPipeline([PipelineStep(name="one", prompt="do it")], agent=agent, on_event=handler)

    result = await pipeline.run("hi", run_id="pipe-1")

    assert result.run_id == "pipe-1"
    assert events and all(event.run_id == "pipe-1" for event in events)
    assert {event.event for event in events} == {
        PipelineEventType.START,
        PipelineEventType.STEP_START,
        PipelineEventType.STEP_END,
        PipelineEventType.END,
    }
    assert agent.calls[0]["run_id"] == "pipe-1"


@pytest.mark.asyncio
async def test_pipeline_generates_one_run_id_when_none_is_supplied():
    events: list = []

    async def handler(event) -> None:
        events.append(event)

    pipeline = SkillPipeline([PipelineStep(name="one", prompt="do it")], agent=_FakeAgent(), on_event=handler)

    result = await pipeline.run("hi")

    assert len(result.run_id) == 32
    assert all(event.run_id == result.run_id for event in events)


@pytest.mark.asyncio
async def test_pipeline_custom_runner_receives_the_run_id_in_its_context():
    seen: list[str] = []

    async def runner(step, step_input, context, emit):
        seen.append(context.run_id)
        return "done"

    pipeline = SkillPipeline([PipelineStep(name="one", prompt="do it", runner=runner)])

    result = await pipeline.run("hi", run_id="pipe-2")

    assert seen == ["pipe-2"]
    assert result.run_id == "pipe-2"


@pytest.mark.asyncio
async def test_pipeline_rejects_an_invalid_run_id():
    pipeline = SkillPipeline([PipelineStep(name="one", prompt="do it")], agent=_FakeAgent())

    with pytest.raises(ValueError, match="run_id"):
        await pipeline.run("hi", run_id="")


# ── provider inference events ──────────────────────────────────


class _FakeTransport:
    """Pure pipe: returns the canned wire payload, records the body."""

    label = "fake"

    def __init__(self, chat_payload: dict) -> None:
        self.chat_payload = chat_payload
        self.chat_calls: list[dict] = []

    async def chat(self, body: dict) -> dict:
        self.chat_calls.append(body)
        return self.chat_payload

    async def close(self) -> None:
        pass


_OPENAI_CHAT_PAYLOAD = {
    "id": "c1",
    "object": "chat.completion",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
}

_ANTHROPIC_CHAT_PAYLOAD = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "content": [{"type": "text", "text": "ok"}],
    "usage": {"input_tokens": 1, "output_tokens": 1},
}


@pytest.mark.asyncio
async def test_openai_provider_copies_the_request_run_id_to_inference_events():
    events: list[InferenceEvent] = []

    async def on_event(event) -> None:
        events.append(event)

    transport = _FakeTransport(_OPENAI_CHAT_PAYLOAD)
    provider = OpenAIProvider("k", "https://example.test/v1", on_event=on_event, transport=transport)
    messages = [CLIverMessage(role="user", content="hi")]

    await provider.chat(CLIverRequest(messages=messages, model="m", run_id="r1"))
    await provider.chat(CLIverRequest(messages=messages, model="m", run_id=None))

    assert [event.run_id for event in events] == ["r1", "r1", None, None]
    # request_id keeps its per-model-call meaning
    assert events[0].request_id == events[1].request_id != ""
    assert events[2].request_id == events[3].request_id != events[0].request_id


@pytest.mark.asyncio
async def test_anthropic_provider_copies_the_request_run_id_to_inference_events():
    events: list[InferenceEvent] = []

    async def on_event(event) -> None:
        events.append(event)

    transport = _FakeTransport(_ANTHROPIC_CHAT_PAYLOAD)
    provider = AnthropicProvider("k", "https://example.test", on_event=on_event, transport=transport)

    await provider.chat(CLIverRequest(messages=[CLIverMessage(role="user", content="hi")], model="m", run_id="r2"))

    assert [event.run_id for event in events] == ["r2", "r2"]
