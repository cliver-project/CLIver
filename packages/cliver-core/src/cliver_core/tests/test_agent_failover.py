"""Same-purpose failover — a retryable failure moves to the next binding.

The bindings API is unchanged: one binding mounts directly and a list mounts
its first element as the primary.  Failover never crosses purposes and only
triggers on retryable provider failures (rate limits, timeouts, 5xx).  Once a
run switches, it stays on the working binding for its remaining iterations;
a stream may switch only before it has yielded anything.
"""

from __future__ import annotations

import pytest

from cliver_core.agent import AgentCore
from cliver_core.bindings import ModelBinding, ModelBindings
from cliver_core.errors import ProviderError
from cliver_core.media import MediaContent, MediaType
from cliver_core.messages import CLIverMessage, CLIverMessageChunk, ToolCall
from cliver_core.provider import (
    CLIverRequest,
    CLIverResponse,
    EmbeddingsResult,
    Provider,
)
from cliver_core.tool import tool


@tool(name="echo", description="Echoes the input back.")
def echo(value: str) -> str:
    return f"echo: {value}"


def _final(text: str) -> CLIverResponse:
    return CLIverResponse(message=CLIverMessage(role="assistant", content=text))


def _tool_turn() -> CLIverResponse:
    return CLIverResponse(
        message=CLIverMessage(
            role="assistant",
            content=None,
            tool_calls=[ToolCall(id="call_1", name="echo", args={"value": "hi"})],
        )
    )


# ── Stub providers ─────────────────────────────────────────────


class _ChatProvider(Provider):
    protocol = "openai"
    brand = "mock-failover-chat"

    def __init__(
        self,
        *,
        error: ProviderError | None = None,
        responses: list[CLIverResponse] | None = None,
    ) -> None:
        super().__init__(api_key="", base_url="")
        self.error = error
        self.responses = list(responses or [])
        self.requests: list[CLIverRequest] = []

    async def chat(self, request: CLIverRequest) -> CLIverResponse:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return self.responses.pop(0)

    async def stream(self, request):  # pragma: no cover
        raise AssertionError("stream() is not used here")


class _StreamProvider(Provider):
    protocol = "openai"
    brand = "mock-failover-stream"

    def __init__(
        self,
        *,
        chunks: list[CLIverMessageChunk] | None = None,
        error: ProviderError | None = None,
        error_after: ProviderError | None = None,
    ) -> None:
        super().__init__(api_key="", base_url="")
        self.chunks = list(chunks or [])
        self.error = error
        self.error_after = error_after
        self.requests: list[CLIverRequest] = []

    async def chat(self, request: CLIverRequest) -> CLIverResponse:  # pragma: no cover
        raise AssertionError("chat() is not used here")

    async def stream(self, request):
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        for chunk in self.chunks:
            yield chunk
        if self.error_after is not None:
            raise self.error_after


def _image_success(text: str = "backup image") -> CLIverResponse:
    return CLIverResponse(
        message=CLIverMessage(role="assistant", content=text),
        media=[MediaContent(type=MediaType.IMAGE, data="https://cdn.example/out.png", mime_type="image/png")],
    )


def _retryable_failure(text: str = "rate limited") -> CLIverResponse:
    return CLIverResponse(message=CLIverMessage(role="assistant", content=text), retryable=True)


def _final_failure(text: str = "bad prompt") -> CLIverResponse:
    return CLIverResponse(message=CLIverMessage(role="assistant", content=text), retryable=False)


class _ImageProvider(Provider):
    protocol = "openai"
    brand = "mock-failover-image"

    def __init__(self, response: CLIverResponse) -> None:
        super().__init__(api_key="", base_url="")
        self.response = response
        self.calls: list[dict] = []

    async def image(self, prompt, model, *, media=None, output_dir=None, **options):
        self.calls.append({"prompt": prompt, "model": model, "options": options})
        return self.response

    async def chat(self, request: CLIverRequest) -> CLIverResponse:  # pragma: no cover
        raise NotImplementedError

    async def stream(self, request):  # pragma: no cover
        if False:
            yield None


class _AsrProvider(Provider):
    protocol = "openai"
    brand = "mock-failover-asr"

    def __init__(self, response: CLIverResponse) -> None:
        super().__init__(api_key="", base_url="")
        self.response = response
        self.calls: list[dict] = []

    async def asr(self, media, model, *, language=None, **options):
        self.calls.append({"media": media, "model": model, "language": language, "options": options})
        return self.response

    async def chat(self, request: CLIverRequest) -> CLIverResponse:  # pragma: no cover
        raise NotImplementedError

    async def stream(self, request):  # pragma: no cover
        if False:
            yield None


class _EmbedProvider(Provider):
    protocol = "openai"
    brand = "mock-failover-embed"

    def __init__(
        self,
        *,
        result: EmbeddingsResult | None = None,
        error: ProviderError | None = None,
    ) -> None:
        super().__init__(api_key="", base_url="")
        self.result = result
        self.error = error
        self.calls: list[dict] = []

    async def embed(self, texts, *, model, **options):
        self.calls.append({"texts": texts, "model": model, "options": options})
        if self.error is not None:
            raise self.error
        return self.result

    async def chat(self, request: CLIverRequest) -> CLIverResponse:  # pragma: no cover
        raise NotImplementedError

    async def stream(self, request):  # pragma: no cover
        if False:
            yield None


def _agent(llm_providers: list[Provider] | None = None, **bindings_lists) -> AgentCore:
    """AgentCore with a minimal chat binding unless the test supplies LLMs."""
    if llm_providers is None:
        llm_providers = [_ChatProvider(responses=[_final("ok")])]
    llm = [ModelBinding.llm(model=f"llm-{index}", provider=provider) for index, provider in enumerate(llm_providers)]
    return AgentCore(ModelBindings(llm=llm, **bindings_lists), builtin_tools=[echo])


# ── Chat: retryable → next LLM ─────────────────────────────────


@pytest.mark.asyncio
async def test_chat_fails_over_to_the_next_llm_on_a_retryable_error() -> None:
    primary = _ChatProvider(error=ProviderError("429 rate limited", retryable=True))
    backup = _ChatProvider(responses=[_final("backup answer")])
    agent = AgentCore(
        ModelBindings(
            llm=[
                ModelBinding.llm(model="primary-model", provider=primary, temperature=0.1),
                ModelBinding.llm(model="backup-model", provider=backup, max_tokens=64),
            ]
        )
    )

    response = await agent.chat("hi", options={"temperature": 0.9})

    assert response.message.content == "backup answer"
    assert len(primary.requests) == 1
    assert len(backup.requests) == 1
    # Each binding's defaults apply while it runs; the call site wins.
    assert backup.requests[0].model == "backup-model"
    assert backup.requests[0].options == {"max_tokens": 64, "temperature": 0.9}


@pytest.mark.asyncio
async def test_chat_does_not_fail_over_on_a_non_retryable_error() -> None:
    primary = _ChatProvider(error=ProviderError("400 bad request"))
    backup = _ChatProvider(responses=[_final("backup answer")])
    agent = _agent([primary, backup])

    with pytest.raises(ProviderError, match="400 bad request"):
        await agent.chat("hi")

    assert backup.requests == []


@pytest.mark.asyncio
async def test_chat_raises_the_last_error_when_every_llm_fails() -> None:
    first = _ChatProvider(error=ProviderError("429 first", retryable=True))
    second = _ChatProvider(error=ProviderError("429 second", retryable=True))
    agent = _agent([first, second])

    with pytest.raises(ProviderError, match="429 second"):
        await agent.chat("hi")

    assert len(first.requests) == 1
    assert len(second.requests) == 1


@pytest.mark.asyncio
async def test_chat_stays_on_the_working_binding_for_the_rest_of_the_run() -> None:
    primary = _ChatProvider(error=ProviderError("429", retryable=True))
    backup = _ChatProvider(responses=[_tool_turn(), _final("final")])
    agent = AgentCore(
        ModelBindings(
            llm=[
                ModelBinding.llm(model="primary-model", provider=primary),
                ModelBinding.llm(model="backup-model", provider=backup),
            ]
        ),
        builtin_tools=[echo],
    )

    response = await agent.chat("hi")

    assert response.message.content == "final"
    assert len(primary.requests) == 1  # not retried once the backup worked
    assert len(backup.requests) == 2  # the tool turn and the final turn


# ── Streaming: fail over only before the first chunk ───────────


@pytest.mark.asyncio
async def test_stream_fails_over_before_the_first_chunk() -> None:
    primary = _StreamProvider(error=ProviderError("429", retryable=True))
    backup = _StreamProvider(
        chunks=[
            CLIverMessageChunk(content="backup "),
            CLIverMessageChunk(content="answer"),
            CLIverMessageChunk(native={"role": "assistant", "content": "backup answer"}),
        ]
    )
    agent = _agent([primary, backup])

    chunks = [chunk async for chunk in agent.stream("hi")]

    assert "".join(chunk.content or "" for chunk in chunks) == "backup answer"
    assert len(primary.requests) == 1
    assert len(backup.requests) == 1
    assert backup.requests[0].model == "llm-1"


@pytest.mark.asyncio
async def test_stream_does_not_fail_over_after_a_chunk_was_emitted() -> None:
    primary = _StreamProvider(
        chunks=[CLIverMessageChunk(content="partial")],
        error_after=ProviderError("429", retryable=True),
    )
    backup = _StreamProvider(chunks=[CLIverMessageChunk(native={"role": "assistant", "content": "never"})])
    agent = _agent([primary, backup])

    seen: list[CLIverMessageChunk] = []
    with pytest.raises(ProviderError, match="429"):
        async for chunk in agent.stream("hi"):
            seen.append(chunk)

    assert [chunk.content for chunk in seen] == ["partial"]
    assert backup.requests == []


# ── Media capabilities: retryable failure response → next ─────


def _image_agent(primary: _ImageProvider, backup: _ImageProvider) -> AgentCore:
    return _agent(
        image=[
            ModelBinding.image(model="img-a", provider=primary, size="1024x1024"),
            ModelBinding.image(model="img-b", provider=backup),
        ]
    )


@pytest.mark.asyncio
async def test_image_fails_over_on_a_retryable_media_failure() -> None:
    primary = _ImageProvider(_retryable_failure())
    backup = _ImageProvider(_image_success())
    agent = _image_agent(primary, backup)

    response = await agent.image("a cat")

    assert response is backup.response
    assert response.media
    assert len(primary.calls) == 1
    assert len(backup.calls) == 1
    assert primary.calls[0]["options"] == {"size": "1024x1024"}  # binding defaults per candidate
    assert backup.calls[0]["model"] == "img-b"
    assert backup.calls[0]["options"] == {}


@pytest.mark.asyncio
async def test_image_does_not_fail_over_on_a_non_retryable_failure() -> None:
    primary = _ImageProvider(_final_failure())
    backup = _ImageProvider(_image_success())
    agent = _image_agent(primary, backup)

    response = await agent.image("a cat")

    assert response.message.content == "bad prompt"
    assert backup.calls == []


@pytest.mark.asyncio
async def test_image_returns_the_last_failure_when_every_binding_fails() -> None:
    primary = _ImageProvider(_retryable_failure("first failure"))
    backup = _ImageProvider(_retryable_failure("second failure"))
    agent = _image_agent(primary, backup)

    response = await agent.image("a cat")

    assert response.message.content == "second failure"
    assert len(primary.calls) == 1
    assert len(backup.calls) == 1


@pytest.mark.asyncio
async def test_asr_fails_over_on_a_retryable_media_failure() -> None:
    primary = _AsrProvider(_retryable_failure())
    backup = _AsrProvider(_image_success("transcript"))
    agent = _agent(
        asr=[
            ModelBinding.asr(model="asr-a", provider=primary),
            ModelBinding.asr(model="asr-b", provider=backup),
        ]
    )

    response = await agent.asr(b"RIFF")

    assert response.message.content == "transcript"
    assert backup.calls[0]["model"] == "asr-b"


# ── Typed capabilities: retryable ProviderError → next ────────


@pytest.mark.asyncio
async def test_embed_fails_over_on_a_retryable_provider_error() -> None:
    primary = _EmbedProvider(error=ProviderError("429", retryable=True))
    backup = _EmbedProvider(result=EmbeddingsResult(vectors=[[1.0, 2.0]], model="embed-b"))
    agent = _agent(
        embed=[
            ModelBinding.embed(model="embed-a", provider=primary),
            ModelBinding.embed(model="embed-b", provider=backup),
        ]
    )

    result = await agent.embed("hello")

    assert result.vectors == [[1.0, 2.0]]
    assert len(primary.calls) == 1
    assert len(backup.calls) == 1
    assert backup.calls[0]["model"] == "embed-b"


@pytest.mark.asyncio
async def test_embed_does_not_fail_over_on_a_non_retryable_error() -> None:
    primary = _EmbedProvider(error=ProviderError("400"))
    backup = _EmbedProvider(result=EmbeddingsResult(vectors=[[1.0]], model="embed-b"))
    agent = _agent(
        embed=[
            ModelBinding.embed(model="embed-a", provider=primary),
            ModelBinding.embed(model="embed-b", provider=backup),
        ]
    )

    with pytest.raises(ProviderError, match="400"):
        await agent.embed("hello")

    assert backup.calls == []
