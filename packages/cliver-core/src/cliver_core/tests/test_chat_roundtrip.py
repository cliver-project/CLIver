"""
Smoke test: AgentCore's Re-Act loops (chat + stream) run end-to-end against
a mocked Provider. Proves the loops, message construction, and the streaming
terminal-chunk contract all wire together — independent of any real provider
SDK.
"""

from __future__ import annotations

import json

import pytest

from cliver_core.agent import AgentCore
from cliver_core.bindings import ModelBinding, ModelBindings
from cliver_core.messages import CLIverMessage, CLIverMessageChunk, ToolCall, UsageInfo
from cliver_core.provider import (
    CLIverRequest,
    CLIverResponse,
    Provider,
)
from cliver_core.tool import tool


class _MockProvider(Provider):
    """Provider that returns a single canned response — no SDK import."""

    protocol = "mock"
    brand = "mock"

    def __init__(self) -> None:
        super().__init__(api_key="", base_url="")
        self.calls: list[CLIverRequest] = []

    async def chat(self, request: CLIverRequest) -> CLIverResponse:
        self.calls.append(request)
        return CLIverResponse(
            message=CLIverMessage(role="assistant", content="Hello from mock."),
            usage=None,
            media=None,
        )

    async def stream(self, request):
        # Not exercised in the chat tests
        if False:
            yield None


class _ToolCallingChatProvider(Provider):
    """Turn 1 asks for ``echo`` (held only in call-time ``extra_tools``);
    turn 2 answers with text."""

    protocol = "openai"
    brand = "mock-tool-call"

    def __init__(self) -> None:
        super().__init__(api_key="", base_url="")
        self.calls: list[CLIverRequest] = []

    async def chat(self, request: CLIverRequest) -> CLIverResponse:
        self.calls.append(request)
        if len(self.calls) == 1:
            return CLIverResponse(
                message=CLIverMessage(
                    role="assistant",
                    content=None,
                    tool_calls=[ToolCall(id="call_1", name="echo", args={"value": "hi"})],
                ),
                usage=None,
                media=None,
            )
        return CLIverResponse(message=CLIverMessage(role="assistant", content="done"), usage=None, media=None)

    async def stream(self, request):
        # Not exercised in the chat tests
        if False:
            yield None


class _StreamingProvider(Provider):
    """Streams deltas, then the terminal chunk carrying the aggregated native.

    Turn 1 asks for a tool; turn 2 answers with text — the two shapes the
    streaming loop must handle.
    """

    protocol = "openai"
    brand = "mock"

    def __init__(self) -> None:
        super().__init__(api_key="", base_url="")
        self.calls: list[CLIverRequest] = []

    async def chat(self, request: CLIverRequest) -> CLIverResponse:  # pragma: no cover
        raise AssertionError("chat() is not used by the streaming loop")

    async def stream(self, request):
        self.calls.append(request)
        if len(self.calls) == 1:
            yield CLIverMessageChunk(content="Let me ")
            yield CLIverMessageChunk(content="check.", vendor_ext={"reasoning_content": "hmm"})
            yield CLIverMessageChunk(
                native={
                    "role": "assistant",
                    "content": "Let me check.",
                    # vendor extras ride the terminal native into the message
                    "reasoning_content": "hmm",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "echo", "arguments": '{"value": "hi"}'},
                        }
                    ],
                },
                usage=UsageInfo(input_tokens=3, output_tokens=4),
            )
        else:
            yield CLIverMessageChunk(content="done")
            yield CLIverMessageChunk(native={"role": "assistant", "content": "done"})


class _AnthropicStreamingProvider(Provider):
    """Streams one anthropic-shaped turn — pins ``from_native``'s protocol source.

    ``protocol = "anthropic"`` here while the llm binding below keeps the
    factory default (``protocol="openai"``): a loop that read the binding
    instead of the provider would decode the block-form native as openai and
    leave the raw block list in ``content``.
    """

    protocol = "anthropic"
    brand = "mock-anthropic"

    def __init__(self) -> None:
        super().__init__(api_key="", base_url="")
        self.calls: list[CLIverRequest] = []

    async def chat(self, request: CLIverRequest) -> CLIverResponse:  # pragma: no cover
        raise AssertionError("chat() is not used by the streaming loop")

    async def stream(self, request):
        self.calls.append(request)
        yield CLIverMessageChunk(content="ok")
        yield CLIverMessageChunk(
            native={"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
        )


@tool(name="echo", description="Echoes the input back.")
def echo(value: str) -> str:
    return f"echo: {value}"


@pytest.mark.asyncio
async def test_chat_roundtrip_with_mocked_provider() -> None:
    provider = _MockProvider()
    agent = AgentCore(
        ModelBindings(llm=ModelBinding.llm(model="mock-model", provider=provider)),
        builtin_tools=[],
    )
    response = await agent.chat("hi")
    assert response.message.role == "assistant"
    assert "Hello from mock" in response.message.content
    assert len(provider.calls) == 1


@pytest.mark.asyncio
async def test_agent_core_constructs_without_tools() -> None:
    """Smoke check: AgentCore can be constructed with no tools and still chat."""
    provider = _MockProvider()
    agent = AgentCore(
        ModelBindings(llm=ModelBinding.llm(model="mock-model", provider=provider)),
        builtin_tools=[],
    )
    assert agent.tool_registry is not None
    response = await agent.chat("hi")
    assert response.message.role == "assistant"


@pytest.mark.asyncio
async def test_stream_loop_executes_tools_from_terminal_native() -> None:
    provider = _StreamingProvider()
    agent = AgentCore(
        ModelBindings(llm=ModelBinding.llm(model="mock-model", provider=provider)),
        builtin_tools=[echo],
    )

    chunks = [c async for c in agent.stream("hi")]

    # live deltas flow outward; the terminal chunk is consumed, not forwarded
    assert "".join(c.content or "" for c in chunks) == "Let me check.done"
    assert all(c.native is None for c in chunks)

    # the directly-registered tool was advertised to the model — execution
    # and advertising resolve the same set
    assert [t.name for t in provider.calls[0].tools] == ["echo"]

    # the streamed assistant came from the terminal native: its tool_calls
    # executed, and the message kept its native for the next request
    assert len(provider.calls) == 2
    second = provider.calls[1]
    assistant = next(m for m in second.messages if m.role == "assistant")
    assert assistant.native_protocol == "openai"
    assert assistant.native["reasoning_content"] == "hmm"
    assert assistant.tool_calls[0].name == "echo"
    tool_message = next(m for m in second.messages if m.role == "tool")
    assert tool_message.tool_call_id == "call_1"
    assert "echo: hi" in tool_message.content

    # the streamed iteration's diagnostic trace carries usage — cost-tracking
    # parity with the non-streaming loop
    trace_chunk = next(c for c in chunks if "__llm_trace__" in c.vendor_ext)
    trace = json.loads(trace_chunk.vendor_ext["__llm_trace__"])
    assert trace[0]["usage"] == {"input_tokens": 3, "output_tokens": 4, "cache_read_input_tokens": None}
    assert trace[0]["message"]["content"] == "Let me check."
    assert trace[1]["usage"] is None  # turn 2's terminal chunk carried none


# ── call-time extra_tools: advertised AND executable ───────────


@pytest.mark.asyncio
async def test_extra_tools_execute_in_the_chat_loop() -> None:
    provider = _ToolCallingChatProvider()
    agent = AgentCore(ModelBindings(llm=ModelBinding.llm(model="mock-model", provider=provider)))
    assert agent.tool_registry.get("echo") is None  # held only in extra_tools

    response = await agent.chat("hi", extra_tools=[echo])

    assert response.message.content == "done"
    assert [t.name for t in provider.calls[0].tools] == ["echo"]
    tool_message = next(m for m in provider.calls[1].messages if m.role == "tool")
    assert tool_message.tool_call_id == "call_1"
    assert "echo: hi" in tool_message.content


@pytest.mark.asyncio
async def test_extra_tools_execute_in_the_stream_loop() -> None:
    provider = _StreamingProvider()
    agent = AgentCore(ModelBindings(llm=ModelBinding.llm(model="mock-model", provider=provider)))
    assert agent.tool_registry.get("echo") is None

    chunks = [c async for c in agent.stream("hi", extra_tools=[echo])]

    assert "".join(c.content or "" for c in chunks) == "Let me check.done"
    assert [t.name for t in provider.calls[0].tools] == ["echo"]
    tool_message = next(m for m in provider.calls[1].messages if m.role == "tool")
    assert tool_message.tool_call_id == "call_1"
    assert "echo: hi" in tool_message.content


# ── bindings.llm: the chat model handle ────────────────────────


def test_legacy_provider_and_model_kwargs_are_gone() -> None:
    """The loose provider/model pair was replaced by ``bindings.llm`` (no aliases)."""
    with pytest.raises(TypeError):
        AgentCore(provider=_MockProvider(), model="mock-model")  # type: ignore[call-arg]


@pytest.mark.asyncio
async def test_llm_binding_params_are_default_request_options() -> None:
    provider = _MockProvider()
    agent = AgentCore(
        ModelBindings(
            llm=ModelBinding.llm(model="mock-model", provider=provider, temperature=0.3, max_tokens=99),
        )
    )

    await agent.chat("hi")
    assert provider.calls[0].options == {"temperature": 0.3, "max_tokens": 99}

    # call-site options win on conflict; untouched defaults stay in place
    await agent.chat("hi", options={"temperature": 0.9, "top_p": 0.5})
    assert provider.calls[1].options == {"temperature": 0.9, "max_tokens": 99, "top_p": 0.5}


@pytest.mark.asyncio
async def test_stream_merges_llm_binding_params_into_every_iteration() -> None:
    provider = _StreamingProvider()
    agent = AgentCore(
        ModelBindings(llm=ModelBinding.llm(model="mock-model", provider=provider, temperature=0.3)),
        builtin_tools=[echo],
    )

    _ = [c async for c in agent.stream("hi")]

    assert len(provider.calls) == 2  # tool turn + answer turn
    assert all(call.options == {"temperature": 0.3} for call in provider.calls)


@pytest.mark.asyncio
async def test_streamed_native_uses_the_provider_protocol_not_the_binding() -> None:
    provider = _AnthropicStreamingProvider()
    bindings = ModelBindings(llm=ModelBinding.llm(model="mock-model", provider=provider))  # protocol stays "openai"
    agent = AgentCore(bindings)

    chunks = [c async for c in agent.stream("hi")]

    assert "".join(c.content or "" for c in chunks) == "ok"
    trace_chunk = next(c for c in chunks if "__llm_trace__" in c.vendor_ext)
    trace = json.loads(trace_chunk.vendor_ext["__llm_trace__"])
    # block-form native decoded as anthropic — reading the binding's protocol
    # would leave the raw block list in content
    assert trace[0]["message"]["content"] == "ok"
