"""AgentCore close contract.

``aclose()`` closes only the providers the bindings built; a caller-supplied
provider is never closed.  Once closed, the agent rejects every public entry
point instead of silently rebuilding providers on a closed loop.  Close is
idempotent and concurrent callers await the same shutdown pass.
"""

from __future__ import annotations

import asyncio

import pytest

from cliver_core.agent import AgentCore
from cliver_core.bindings import ModelBinding, ModelBindings
from cliver_core.messages import CLIverMessage, CLIverMessageChunk
from cliver_core.provider import CLIverRequest, CLIverResponse, Provider


class _FakeProvider(Provider):
    protocol = "openai"
    brand = "fake-lifecycle"

    def __init__(self) -> None:
        super().__init__(api_key="", base_url="")
        self.closed = 0
        self.calls = 0

    async def chat(self, request: CLIverRequest) -> CLIverResponse:
        self.calls += 1
        return CLIverResponse(message=CLIverMessage(role="assistant", content="ok"))

    async def stream(self, request):
        self.calls += 1
        yield CLIverMessageChunk(content="ok")
        yield CLIverMessageChunk(native={"role": "assistant", "content": "ok"})

    async def close(self) -> None:
        self.closed += 1


def _supplied_agent() -> tuple[AgentCore, _FakeProvider]:
    provider = _FakeProvider()
    bindings = ModelBindings(llm=ModelBinding.llm(model="mock-model", provider=provider))
    return AgentCore(bindings), provider


def _built_agent(monkeypatch) -> tuple[AgentCore, _FakeProvider]:
    provider = _FakeProvider()
    monkeypatch.setattr("cliver_core.bindings.create_provider", lambda **kwargs: provider)
    bindings = ModelBindings(llm=ModelBinding.llm(model="mock-model", base_url="https://example.test"))
    return AgentCore(bindings), provider


@pytest.mark.asyncio
async def test_chat_after_close_raises() -> None:
    agent, provider = _supplied_agent()
    await agent.aclose()

    with pytest.raises(RuntimeError, match="closed"):
        await agent.chat("hi")
    assert provider.closed == 0


@pytest.mark.asyncio
async def test_stream_after_close_raises() -> None:
    agent, _provider = _supplied_agent()
    await agent.aclose()

    with pytest.raises(RuntimeError, match="closed"):
        async for _chunk in agent.stream("hi"):
            pass


@pytest.mark.asyncio
async def test_run_skill_after_close_raises() -> None:
    agent, _provider = _supplied_agent()
    await agent.aclose()

    with pytest.raises(RuntimeError, match="closed"):
        await agent.run_skill("anything", "hi")


@pytest.mark.asyncio
async def test_capability_after_close_raises() -> None:
    agent, _provider = _supplied_agent()
    await agent.aclose()

    with pytest.raises(RuntimeError, match="closed"):
        await agent.embed("hi")


@pytest.mark.asyncio
async def test_submit_after_close_raises() -> None:
    agent, _provider = _supplied_agent()
    await agent.aclose()

    with pytest.raises(RuntimeError, match="closed"):
        agent.submit_image("hi")


@pytest.mark.asyncio
async def test_aclose_closes_a_built_provider_exactly_once(monkeypatch) -> None:
    agent, provider = _built_agent(monkeypatch)
    await agent.chat("hi")  # build the provider

    await agent.aclose()
    await agent.aclose()

    assert provider.closed == 1


@pytest.mark.asyncio
async def test_aclose_never_closes_a_supplied_provider(monkeypatch) -> None:
    agent, provider = _supplied_agent()
    await agent.chat("hi")

    await agent.aclose()

    assert provider.closed == 0
    assert provider.calls == 1


@pytest.mark.asyncio
async def test_concurrent_aclose_calls_share_one_shutdown(monkeypatch) -> None:
    agent, provider = _built_agent(monkeypatch)
    await agent.chat("hi")

    await asyncio.gather(agent.aclose(), agent.aclose())

    assert provider.closed == 1


class _SlowImageProvider(_FakeProvider):
    """An image provider whose generation blocks until cancelled."""

    def __init__(self) -> None:
        super().__init__()
        self.generation_started = asyncio.Event()

    async def image(self, prompt: str, model: str, **kwargs) -> CLIverResponse:
        self.generation_started.set()
        await asyncio.sleep(3600)
        return CLIverResponse(message=CLIverMessage(role="assistant", content="too late"))


@pytest.mark.asyncio
async def test_aclose_cancels_media_jobs_the_agent_started() -> None:
    image_provider = _SlowImageProvider()
    bindings = ModelBindings(
        llm=ModelBinding.llm(model="mock-model", provider=_FakeProvider()),
        image=ModelBinding.image(model="mock-image", provider=image_provider),
    )
    agent = AgentCore(bindings)
    job = agent.submit_image("hi")
    await asyncio.wait_for(image_provider.generation_started.wait(), timeout=1)

    await agent.aclose()

    assert job.status == "cancelled"
    assert image_provider.closed == 0  # caller-supplied provider stays open


class _FakeMCPClient:
    """Stub MCP client — records start/close; serves no tools."""

    def __init__(self) -> None:
        self.started = 0
        self.closed = 0

    async def start(self) -> None:
        self.started += 1

    async def get_tools(self, servers=None) -> list:
        return []

    async def close(self) -> None:
        self.closed += 1


@pytest.mark.asyncio
async def test_aclose_never_closes_a_caller_supplied_mcp_client() -> None:
    client = _FakeMCPClient()
    bindings = ModelBindings(llm=ModelBinding.llm(model="mock-model", provider=_FakeProvider()))
    agent = AgentCore(bindings, mcp_client=client)

    await agent.chat("hi")
    await agent.aclose()

    assert client.started == 1
    assert client.closed == 0  # whoever built the client owns its lifecycle
