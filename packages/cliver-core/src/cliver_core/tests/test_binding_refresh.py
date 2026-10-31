"""Explicit credential rotation — ``arefresh()`` on bindings and ``AgentCore``.

Rotation is application-driven: nothing refreshes automatically on a 401.
``arefresh()`` resolves the credential, builds the replacement provider
*before* swapping, and retires the old provider.  A retired provider with no
in-flight call is closed immediately; one held by an in-flight call (a
``lease_provider()`` lease) stays open until its last lease is released and
the next refresh prunes it.  ``aclose()`` closes whatever remains.
"""

from __future__ import annotations

import asyncio

import pytest

from cliver_core.agent import AgentCore
from cliver_core.bindings import ModelBinding, ModelBindings
from cliver_core.capabilities import embed
from cliver_core.messages import CLIverMessage, CLIverMessageChunk
from cliver_core.provider import CLIverRequest, CLIverResponse, EmbeddingsResult, Provider


class _KeyProvider(Provider):
    protocol = "openai"
    brand = "mock-key"
    instances: list["_KeyProvider"] = []

    def __init__(self, api_key: str, base_url: str, **kwargs) -> None:
        super().__init__(api_key=api_key, base_url=base_url)
        self.closed = False
        self.release: asyncio.Event | None = None
        self.started: asyncio.Event | None = None
        _KeyProvider.instances.append(self)

    async def chat(self, request: CLIverRequest) -> CLIverResponse:
        if self.started is not None:
            self.started.set()
        if self.release is not None:
            await self.release.wait()
        return CLIverResponse(message=CLIverMessage(role="assistant", content=f"key={self.api_key}"))

    async def stream(self, request):
        if self.started is not None:
            self.started.set()
        yield CLIverMessageChunk(content="streamed")
        if self.release is not None:
            await self.release.wait()
        yield CLIverMessageChunk(native={"role": "assistant", "content": "streamed"})

    async def embed(self, texts, *, model, **options):
        if self.started is not None:
            self.started.set()
        if self.release is not None:
            await self.release.wait()
        return EmbeddingsResult(vectors=[[0.1] for _ in texts], model=model)

    async def close(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def _clear_instances():
    _KeyProvider.instances.clear()
    yield
    _KeyProvider.instances.clear()


def _binding(credential_provider=None, *, purpose: str = "llm", **kwargs) -> ModelBinding:
    return ModelBinding(
        model="mock-model",
        api_key="old-key",
        base_url="https://example.test/v1",
        provider_class=_KeyProvider,
        purpose=purpose,
        credential_provider=credential_provider,
        **kwargs,
    )


@pytest.mark.asyncio
async def test_sync_credential_provider_rebuilds_the_provider():
    binding = _binding(lambda: "new-key")
    old = binding.get_provider()
    assert old.api_key == "old-key"

    await binding.arefresh()

    new = binding.get_provider()
    assert new is not old
    assert new.api_key == "new-key"
    assert binding.api_key == "new-key"
    # No call held a lease, so the retired provider is closed immediately.
    assert old.closed is True

    await binding.aclose()
    assert new.closed is True
    assert old.closed is True


@pytest.mark.asyncio
async def test_async_credential_provider_is_awaited():
    async def fetch() -> str:
        await asyncio.sleep(0)
        return "async-key"

    binding = _binding(fetch)
    binding.get_provider()

    await binding.arefresh()

    assert binding.get_provider().api_key == "async-key"


@pytest.mark.asyncio
async def test_unbuilt_binding_updates_the_key_for_the_first_build():
    binding = _binding(lambda: "fresh-key")

    await binding.arefresh()

    assert binding.api_key == "fresh-key"
    assert _KeyProvider.instances == []
    assert binding.get_provider().api_key == "fresh-key"


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", ["", "   ", 123, None])
async def test_invalid_credentials_leave_the_binding_untouched(bad):
    binding = _binding(lambda: bad)
    old = binding.get_provider()

    with pytest.raises(ValueError, match="non-empty string"):
        await binding.arefresh()

    assert binding.api_key == "old-key"
    assert binding.get_provider() is old


@pytest.mark.asyncio
async def test_a_raising_credential_provider_leaves_the_binding_untouched():
    async def fetch() -> str:
        raise RuntimeError("vault unreachable")

    binding = _binding(fetch)
    old = binding.get_provider()

    with pytest.raises(RuntimeError, match="vault unreachable"):
        await binding.arefresh()

    assert binding.api_key == "old-key"
    assert binding.get_provider() is old


@pytest.mark.asyncio
async def test_without_a_credential_provider_the_current_key_is_reused():
    binding = _binding()
    old = binding.get_provider()

    await binding.arefresh()

    assert binding.api_key == "old-key"
    new = binding.get_provider()
    assert new is not old
    assert new.api_key == "old-key"


@pytest.mark.asyncio
async def test_supplied_provider_is_never_rebuilt_or_closed():
    supplied = _KeyProvider("supplied-key", "https://example.test/v1")
    binding = ModelBinding(
        model="mock-model",
        provider=supplied,
        base_url="https://example.test/v1",
        purpose="llm",
        credential_provider=lambda: "rotated-key",
    )

    assert binding.get_provider() is supplied
    await binding.arefresh()

    assert binding.api_key == "rotated-key"
    assert binding.get_provider() is supplied
    await binding.aclose()
    assert supplied.closed is False


def test_arefresh_requires_the_owning_loop():
    binding = _binding(lambda: "new-key")

    async def claim() -> None:
        binding.get_provider()

    asyncio.run(claim())
    with pytest.raises(RuntimeError, match="another event loop"):
        asyncio.run(binding.arefresh())


@pytest.mark.asyncio
async def test_a_lease_keeps_a_retired_provider_open_across_a_refresh():
    binding = _binding(lambda: "new-key")
    old = binding.get_provider()

    async with binding.lease_provider() as leased:
        assert leased is old
        await binding.arefresh()
        # The call is still in flight on ``old`` — retiring it must not close it.
        assert old.closed is False
        assert binding.get_provider() is not old

    # Released: the provider is now idle and the next refresh prunes it.
    assert old.closed is False
    await binding.arefresh()
    assert old.closed is True


@pytest.mark.asyncio
async def test_in_flight_leased_call_finishes_on_the_retired_provider():
    binding = _binding(lambda: "new-key")
    old = binding.get_provider()
    old.release = asyncio.Event()
    old.started = asyncio.Event()

    async def call() -> CLIverResponse:
        async with binding.lease_provider() as provider:
            return await provider.chat(CLIverRequest(messages=[], model="mock-model"))

    task = asyncio.create_task(call())
    await old.started.wait()
    await binding.arefresh()
    old.release.set()

    response = await task
    assert response.message.content == "key=old-key"
    assert old.closed is False


@pytest.mark.asyncio
async def test_agent_chat_holds_a_lease_across_a_refresh():
    binding = _binding(lambda: "new-key")
    agent = AgentCore(ModelBindings(llm=binding))
    old = binding.get_provider()
    old.release = asyncio.Event()
    old.started = asyncio.Event()

    task = asyncio.create_task(agent.chat("hi", max_iterations=1, skill_dispatch=False))
    await old.started.wait()
    await agent.arefresh_credentials()
    assert old.closed is False
    await binding.arefresh()
    assert old.closed is False  # still in flight; not pruned

    old.release.set()
    response = await task
    assert response.message.content == "key=old-key"

    await agent.arefresh_credentials()
    assert old.closed is True
    await agent.aclose()


@pytest.mark.asyncio
async def test_agent_stream_holds_a_lease_until_the_stream_closes():
    binding = _binding(lambda: "new-key")
    agent = AgentCore(ModelBindings(llm=binding))
    old = binding.get_provider()
    old.release = asyncio.Event()
    old.started = asyncio.Event()

    async def consume() -> list[CLIverMessageChunk]:
        return [chunk async for chunk in agent.stream("hi", max_iterations=1, skill_dispatch=False)]

    task = asyncio.create_task(consume())
    await old.started.wait()
    await agent.arefresh_credentials()
    assert old.closed is False

    old.release.set()
    chunks = await task
    assert any(chunk.content == "streamed" for chunk in chunks)
    assert old.closed is False

    await agent.arefresh_credentials()
    assert old.closed is True
    await agent.aclose()


@pytest.mark.asyncio
async def test_capability_call_holds_a_lease_across_a_refresh():
    binding = _binding(lambda: "new-key", purpose="embed")
    old = binding.get_provider()
    old.release = asyncio.Event()
    old.started = asyncio.Event()

    task = asyncio.create_task(embed(texts=["a", "b"], binding=binding))
    await old.started.wait()
    await binding.arefresh()
    assert old.closed is False

    old.release.set()
    result = await task
    assert result.vectors == [[0.1], [0.1]]

    await binding.arefresh()
    assert old.closed is True


@pytest.mark.asyncio
async def test_agent_refreshes_every_binding_across_purposes():
    calls: list[str] = []

    def credential(name: str):
        def fetch() -> str:
            calls.append(name)
            return f"{name}-key"

        return fetch

    llm_primary = _binding(credential("llm-primary"))
    llm_backup = _binding(credential("llm-backup"))
    embed = _binding(credential("embed"), purpose="embed")
    agent = AgentCore(ModelBindings(llm=[llm_primary, llm_backup], embed=embed))

    await agent.arefresh_credentials()

    assert calls == ["llm-primary", "llm-backup", "embed"]
    assert llm_primary.api_key == "llm-primary-key"
    assert llm_backup.api_key == "llm-backup-key"
    assert embed.api_key == "embed-key"


@pytest.mark.asyncio
async def test_agent_refresh_rejects_a_closed_agent():
    agent = AgentCore(ModelBindings(llm=_binding(lambda: "new-key")))
    await agent.aclose()

    with pytest.raises(RuntimeError, match="closed"):
        await agent.arefresh_credentials()


@pytest.mark.asyncio
async def test_aclose_is_idempotent_with_retired_providers():
    binding = _binding(lambda: "new-key")
    old = binding.get_provider()
    await binding.arefresh()
    new = binding.get_provider()

    await binding.aclose()
    await binding.aclose()

    assert old.closed is True
    assert new.closed is True


def test_factories_accept_a_credential_provider():
    def fetch() -> str:
        return "k"

    assert ModelBinding.llm(model="m", base_url="https://x/v1", credential_provider=fetch).credential_provider is fetch
    image = ModelBinding.image(model="m", base_url="https://x/v1", credential_provider=fetch)
    assert image.credential_provider is fetch
