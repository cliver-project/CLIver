"""A ModelBinding belongs to one event loop — unless its provider is loop-agnostic.

A provider the binding builds holds loop-bound clients (the SDK and its
``httpx`` pool, stream tasks).  Using the same binding from a second loop — in
practice, a second thread — fails deep inside the client or hangs.  The binding
records the loop on first use and rejects a different one with
:class:`LoopAffinityError` and a message that says what to do: one binding (and
agent) per event loop.  ``aclose`` releases the claim, so a binding can be
reused after its agent shuts down.

A caller-supplied provider marked ``loop_agnostic`` (the in-process local
embed/rerank providers) owns no loop-bound resources, so the binding does not
claim a loop for it and it is safe to share across ``asyncio.run`` calls.
:meth:`ModelBinding.for_current_loop` rebinds a binding spec to a new loop.
"""

from __future__ import annotations

import asyncio

import pytest

from cliver_core.agent import AgentCore
from cliver_core.bindings import ModelBinding, ModelBindings
from cliver_core.errors import LoopAffinityError
from cliver_core.local.backends import LoadedEmbedding
from cliver_core.messages import CLIverMessage
from cliver_core.provider import CLIverRequest, CLIverResponse, Provider


class _FakeProvider(Provider):
    protocol = "openai"
    brand = "fake-loop-affinity"

    def __init__(self) -> None:
        super().__init__(api_key="", base_url="")
        self.calls = 0

    async def chat(self, request: CLIverRequest) -> CLIverResponse:
        self.calls += 1
        return CLIverResponse(message=CLIverMessage(role="assistant", content="ok"))

    async def stream(self, request):  # pragma: no cover
        raise AssertionError("stream() is not used here")


def _agent(binding: ModelBinding) -> AgentCore:
    return AgentCore(ModelBindings(llm=binding))


def test_binding_rejects_a_second_event_loop() -> None:
    binding = ModelBinding.llm(model="mock-model", provider=_FakeProvider())

    async def run() -> str:
        return (await _agent(binding).chat("hi")).message.content or ""

    assert asyncio.run(run()) == "ok"
    with pytest.raises(LoopAffinityError, match="another event loop"):
        asyncio.run(run())


def test_aclose_releases_loop_affinity_for_a_built_provider(monkeypatch) -> None:
    """A binding that builds its provider can be closed and reused: aclose
    closes the client, and the next loop builds a fresh one."""
    built: list[_FakeProvider] = []

    def fake_create(**kwargs):
        provider = _FakeProvider()
        built.append(provider)
        return provider

    monkeypatch.setattr("cliver_core.bindings.create_provider", fake_create)
    binding = ModelBinding.llm(model="mock-model", api_key="k", base_url="https://example.test/v1")

    async def run() -> None:
        agent = _agent(binding)
        await agent.chat("hi")
        await agent.aclose()

    asyncio.run(run())
    asyncio.run(run())  # the binding rebuilds on the new loop
    assert len(built) == 2
    assert built[0] is not built[1]


def test_aclose_keeps_the_claim_of_a_loop_bound_supplied_provider() -> None:
    """A caller-supplied loop-bound provider is never closed, so closing the
    binding must not release its loop claim — reusing it on another loop would
    hand a still-open, loop-bound client to a different event loop."""
    binding = ModelBinding.llm(model="mock-model", provider=_FakeProvider())

    async def run() -> None:
        agent = _agent(binding)
        try:
            await agent.chat("hi")
        finally:
            await agent.aclose()

    asyncio.run(run())
    with pytest.raises(LoopAffinityError, match="another event loop"):
        asyncio.run(run())


def test_aclose_from_another_loop_raises() -> None:
    binding = ModelBinding.llm(model="mock-model", provider=_FakeProvider())

    async def use() -> None:
        await _agent(binding).chat("hi")

    asyncio.run(use())
    with pytest.raises(LoopAffinityError, match="another event loop"):
        asyncio.run(binding.aclose())


@pytest.mark.asyncio
async def test_concurrent_calls_in_one_loop_share_the_binding() -> None:
    provider = _FakeProvider()
    binding = ModelBinding.llm(model="mock-model", provider=provider)
    agent = _agent(binding)

    await asyncio.gather(agent.chat("hi"), agent.chat("hi"))

    assert binding.get_provider() is provider
    assert provider.calls == 2


# ── Loop-agnostic supplied providers ──────────────────────────


class _LoopAgnosticProvider(_FakeProvider):
    """Stands in for LocalEmbeddingProvider: owns no loop-bound resources."""

    loop_agnostic = True


def test_supplied_loop_agnostic_provider_works_across_asyncio_run() -> None:
    """One process-wide local provider behind one cached binding, each call
    its own ``asyncio.run`` — the cliver-playbook pattern."""
    provider = _LoopAgnosticProvider()
    binding = ModelBinding.llm(model="mock-model", provider=provider)

    async def run() -> None:
        agent = _agent(binding)
        assert (await agent.chat("hi")).message.content == "ok"

    asyncio.run(run())
    asyncio.run(run())  # the cached binding is reused without an aclose in between
    assert provider.calls == 2

    async def run_and_close() -> None:
        agent = _agent(binding)
        await agent.chat("hi")
        await agent.aclose()  # closing from a later loop is safe for this provider

    asyncio.run(run_and_close())
    assert provider.calls == 3


def test_binding_built_provider_still_rejects_a_second_event_loop(monkeypatch) -> None:
    built = _FakeProvider()
    monkeypatch.setattr("cliver_core.bindings.create_provider", lambda **kwargs: built)
    binding = ModelBinding.llm(model="mock-model", api_key="k", base_url="https://example.test/v1")

    async def run() -> None:
        await _agent(binding).chat("hi")

    asyncio.run(run())
    with pytest.raises(LoopAffinityError, match="another event loop"):
        asyncio.run(run())


def test_loop_affinity_error_is_a_public_runtime_error() -> None:
    import cliver_core

    assert issubclass(LoopAffinityError, RuntimeError)
    assert cliver_core.LoopAffinityError is LoopAffinityError


def test_local_providers_are_declared_loop_agnostic() -> None:
    from cliver_core.local.providers import LocalEmbeddingProvider, LocalRerankProvider
    from cliver_core.provider import Provider

    assert Provider.loop_agnostic is False  # the default stays loop-bound
    assert LocalEmbeddingProvider.loop_agnostic is True
    assert LocalRerankProvider.loop_agnostic is True


class _StubEmbeddingBackend:
    """Minimal in-process backend: no model download, deterministic vectors."""

    name = "stub"

    def load_embedding(self, spec):
        return LoadedEmbedding(
            handle=object(),
            resolved_model="stub",
            revision=None,
            device="cpu",
            dtype="float32",
            dim=2,
            max_length=spec.max_length,
        )

    def embed(self, loaded, texts, *, batch_size):
        return [[1.0, 0.0] for _ in texts]

    def unload_embedding(self, loaded) -> None:
        loaded.handle = None


def test_real_local_embedding_binding_runs_across_asyncio_run() -> None:
    """The acceptance case with the real LocalEmbeddingProvider: one binding,
    two consecutive ``asyncio.run`` calls, then a close on a third loop."""
    from cliver_core.capabilities import embed
    from cliver_core.local import LocalEmbeddingProvider, local_embed

    provider = LocalEmbeddingProvider(model="stub/model", backend=_StubEmbeddingBackend())
    binding = local_embed(provider=provider)

    async def run() -> list[list[float]]:
        return (await embed(texts=["hello"], binding=binding)).vectors

    assert asyncio.run(run()) == [[1.0, 0.0]]
    assert asyncio.run(run()) == [[1.0, 0.0]]
    asyncio.run(provider.close())


# ── Rebinding with for_current_loop() ─────────────────────────


def test_for_current_loop_shares_a_supplied_loop_agnostic_provider() -> None:
    provider = _LoopAgnosticProvider()
    binding = ModelBinding.llm(model="mock-model", provider=provider, temperature=0.2)

    clone = binding.for_current_loop()

    assert clone is not binding
    assert clone.provider is provider
    assert clone.model == "mock-model"
    assert clone.params == {"temperature": 0.2}


def test_for_current_loop_on_a_built_binding_builds_a_fresh_provider(monkeypatch) -> None:
    built: list[_FakeProvider] = []

    def fake_create(**kwargs):
        provider = _FakeProvider()
        built.append(provider)
        return provider

    monkeypatch.setattr("cliver_core.bindings.create_provider", fake_create)
    binding = ModelBinding.llm(model="mock-model", api_key="k", base_url="https://example.test/v1")

    async def run(b: ModelBinding) -> None:
        await _agent(b).chat("hi")

    asyncio.run(run(binding))
    clone = binding.for_current_loop()
    asyncio.run(run(clone))

    assert len(built) == 2
    assert built[0] is not built[1]


def test_for_current_loop_refuses_a_loop_bound_supplied_provider() -> None:
    binding = ModelBinding.llm(model="mock-model", provider=_FakeProvider())

    with pytest.raises(ValueError, match="loop-bound"):
        binding.for_current_loop()
