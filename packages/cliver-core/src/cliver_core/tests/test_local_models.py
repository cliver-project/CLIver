"""In-process local providers — fake-backend unit tests.

No torch, sentence-transformers, or FlagEmbedding is imported here: the
providers are exercised through backend objects, which is also the documented
extension point for applications and one-off integrations.  The real backends
are covered by the gated live tests in ``test_local_live.py``.
"""

from __future__ import annotations

import asyncio
import math
import os
import subprocess
import sys
import textwrap
import threading
import time

import pytest

from cliver_core.agent import AgentCore
from cliver_core.bindings import ModelBinding, ModelBindings
from cliver_core.capabilities import embed, rerank
from cliver_core.errors import CapabilityNotSupported
from cliver_core.local import (
    LoadedEmbedding,
    LoadedRerank,
    LocalBackendNotSupportedError,
    LocalConfigurationError,
    LocalDimensionError,
    LocalEmbeddingProvider,
    LocalInferenceError,
    LocalProvider,
    LocalProviderClosedError,
    LocalRerankProvider,
    local_embed,
    local_rerank,
    register_backend,
    resolve_backend,
)
from cliver_core.provider import CLIverRequest, CLIverResponse, Provider

# ── Fake backends ─────────────────────────────────────────────


class FakeEmbeddingBackend:
    name = "fake-embedding"

    def __init__(
        self,
        *,
        dim: int = 2,
        vectors: list[list[float]] | None = None,
        delay: float = 0.0,
    ) -> None:
        self.dim = dim
        self.vectors = vectors
        self.delay = delay
        self.load_calls = 0
        self.unload_calls = 0
        self.embed_calls: list[dict] = []
        self._guard = threading.Lock()
        self._active = 0
        self.max_active = 0
        self.embed_started = threading.Event()

    def _row(self) -> list[float]:
        row = [3.0, 4.0]
        while len(row) < self.dim:
            row.append(1.0)
        return row[: self.dim]

    def load_embedding(self, spec) -> LoadedEmbedding:
        time.sleep(self.delay)
        self.load_calls += 1
        return LoadedEmbedding(
            handle=object(),
            resolved_model="/fake/embedder",
            revision="sha-embed",
            device="cpu",
            dtype="float32",
            dim=self.dim,
            max_length=spec.max_length,
        )

    def embed(self, loaded: LoadedEmbedding, texts: list[str], *, batch_size: int) -> list[list[float]]:
        self.embed_started.set()
        with self._guard:
            self._active += 1
            self.max_active = max(self.max_active, self._active)
        try:
            time.sleep(self.delay)
            self.embed_calls.append({"texts": list(texts), "batch_size": batch_size})
            if self.vectors is not None:
                return [list(row) for row in self.vectors[: len(texts)]]
            return [self._row() for _ in texts]
        finally:
            with self._guard:
                self._active -= 1

    def unload_embedding(self, loaded: LoadedEmbedding) -> None:
        loaded.handle = None
        self.unload_calls += 1


class FakeRerankBackend:
    name = "fake-rerank"

    def __init__(self, *, scores: list[float] | None = None, delay: float = 0.0) -> None:
        self.scores = [0.1, 0.9, 0.5] if scores is None else list(scores)
        self.delay = delay
        self.load_calls = 0
        self.unload_calls = 0
        self.score_calls: list[dict] = []

    def load_rerank(self, spec) -> LoadedRerank:
        time.sleep(self.delay)
        self.load_calls += 1
        return LoadedRerank(
            handle=object(),
            resolved_model="/fake/reranker",
            revision="sha-rerank",
            device="cpu",
            dtype="float32",
            max_length=spec.max_length,
        )

    def score(self, loaded: LoadedRerank, pairs: list[tuple[str, str]], *, batch_size: int) -> list[float]:
        time.sleep(self.delay)
        self.score_calls.append({"pairs": list(pairs), "batch_size": batch_size})
        return list(self.scores[: len(pairs)])

    def unload_rerank(self, loaded: LoadedRerank) -> None:
        loaded.handle = None
        self.unload_calls += 1


class _SlowEmbeddingBackend(FakeEmbeddingBackend):
    def __init__(self, delay: float = 0.05) -> None:
        super().__init__(delay=delay)


class _FailingOnceBackend(FakeEmbeddingBackend):
    def __init__(self) -> None:
        super().__init__()
        self.failures = 1

    def load_embedding(self, spec) -> LoadedEmbedding:
        if self.failures:
            self.failures -= 1
            self.load_calls += 1
            raise RuntimeError("boom")
        return super().load_embedding(spec)


class _UnknownDimBackend(FakeEmbeddingBackend):
    def __init__(self) -> None:
        super().__init__(dim=3)

    def load_embedding(self, spec) -> LoadedEmbedding:
        loaded = super().load_embedding(spec)
        loaded.dim = None
        return loaded


class _StubChatProvider(Provider):
    protocol = "mock"
    brand = "mock"

    def __init__(self) -> None:
        super().__init__(api_key="", base_url="")

    async def chat(self, request: CLIverRequest) -> CLIverResponse:  # pragma: no cover
        raise NotImplementedError

    async def stream(self, request: CLIverRequest):  # pragma: no cover
        if False:
            yield None


def _embedding_provider(backend: FakeEmbeddingBackend, **kwargs) -> LocalEmbeddingProvider:
    return LocalEmbeddingProvider(model="fake/model", backend=backend, **kwargs)


def _rerank_provider(backend: FakeRerankBackend, **kwargs) -> LocalRerankProvider:
    return LocalRerankProvider(model="fake/reranker", backend=backend, **kwargs)


# ── Embedding: results ────────────────────────────────────────


@pytest.mark.asyncio
async def test_embed_returns_vectors_in_input_order_and_loads_once() -> None:
    backend = FakeEmbeddingBackend()
    provider = _embedding_provider(backend)
    binding = local_embed(provider=provider)

    first = await embed(texts=["a", "bb"], binding=binding)
    second = await embed(texts=["ccc"], binding=binding)

    assert first.model == "fake/model"
    assert first.vectors == [[3.0, 4.0], [3.0, 4.0]]
    assert second.vectors == [[3.0, 4.0]]
    assert backend.load_calls == 1
    assert backend.embed_calls[0] == {"texts": ["a", "bb"], "batch_size": 32}
    assert binding.provider is provider
    await provider.close()


@pytest.mark.asyncio
async def test_normalize_l2_normalizes_and_leaves_zero_vectors() -> None:
    backend = FakeEmbeddingBackend(vectors=[[3.0, 4.0], [0.0, 0.0]])
    provider = _embedding_provider(backend)
    binding = local_embed(provider=provider, normalize=True)

    result = await embed(texts=["a", "b"], binding=binding)

    assert result.vectors[0] == pytest.approx([0.6, 0.8])
    assert result.vectors[1] == [0.0, 0.0]
    await provider.close()


@pytest.mark.asyncio
async def test_empty_texts_return_empty_without_loading() -> None:
    backend = FakeEmbeddingBackend()
    provider = _embedding_provider(backend)

    result = await embed(texts=[], binding=local_embed(provider=provider))

    assert result.vectors == []
    assert result.model == "fake/model"
    assert backend.load_calls == 0


@pytest.mark.asyncio
async def test_dim_validates_the_reported_dimension() -> None:
    backend = FakeEmbeddingBackend(dim=2)
    provider = _embedding_provider(backend)

    result = await embed(texts=["x"], binding=local_embed(provider=provider, dim=2))
    assert len(result.vectors[0]) == 2

    with pytest.raises(LocalDimensionError, match="2-d vectors"):
        await embed(texts=["x"], binding=local_embed(provider=provider, dim=3))
    await provider.close()


@pytest.mark.asyncio
async def test_dim_is_checked_against_actual_rows_and_learned_when_unknown() -> None:
    backend = _UnknownDimBackend()
    provider = _embedding_provider(backend)

    with pytest.raises(LocalDimensionError, match="3-d vector"):
        await embed(texts=["x"], binding=local_embed(provider=provider, dim=2))

    result = await embed(texts=["x"], binding=local_embed(provider=provider, dim=3))
    assert len(result.vectors[0]) == 3
    assert provider.info().dim == 3
    await provider.close()


@pytest.mark.asyncio
async def test_binding_defaults_apply_and_call_site_options_win() -> None:
    backend = FakeEmbeddingBackend()
    provider = _embedding_provider(backend)
    binding = local_embed(provider=provider, batch_size=8, normalize=True)

    await embed(texts=["x"], binding=binding)
    assert backend.embed_calls[-1]["batch_size"] == 8

    await embed(texts=["x"], binding=binding, batch_size=2)
    assert backend.embed_calls[-1]["batch_size"] == 2
    await provider.close()


# ── Embedding: validation and errors ─────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("options", "match"),
    [
        ({"batch_size": 0}, "batch_size"),
        ({"normalize": "yes"}, "normalize"),
        ({"path": "/v1/embeddings"}, "no endpoint path"),
        ({"dimensions": 3}, "HTTP-only"),
        ({"device": "cpu"}, "load-time"),
        ({"unknown": 1}, "unsupported embedding option"),
    ],
)
async def test_embed_rejects_invalid_call_options(options: dict, match: str) -> None:
    provider = _embedding_provider(FakeEmbeddingBackend())
    with pytest.raises(LocalConfigurationError, match=match):
        await provider.embed(["x"], model="fake/model", **options)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("vectors", "match"),
    [
        ([[3.0, 4.0]], "1 vectors for 2 texts"),
        ([["nope", 4.0], [3.0, 4.0]], "non-numeric vector"),
        ([[], [3.0, 4.0]], "empty vector"),
        ([[float("inf"), 0.0], [3.0, 4.0]], "non-finite value"),
    ],
)
async def test_embed_rejects_bad_backend_output(vectors: list[list[float]], match: str) -> None:
    backend = FakeEmbeddingBackend(vectors=vectors)
    provider = _embedding_provider(backend)
    with pytest.raises(LocalInferenceError, match=match):
        await provider.embed(["a", "b"], model="fake/model")


@pytest.mark.asyncio
async def test_binding_model_must_match_the_provider_model() -> None:
    provider = _embedding_provider(FakeEmbeddingBackend())
    binding = ModelBinding.embed(model="other/model", provider=provider, protocol="local")
    with pytest.raises(LocalConfigurationError, match="does not match"):
        await embed(texts=["x"], binding=binding)


def test_local_embed_factory_validates_and_forwards_defaults() -> None:
    provider = _embedding_provider(FakeEmbeddingBackend())
    binding = local_embed(provider=provider, dim=2, normalize=True, batch_size=8)

    assert binding.model == "fake/model"
    assert binding.protocol == "local"
    assert binding.purpose == "embed"
    assert binding.provider is provider
    assert binding.call_defaults() == {"dim": 2, "normalize": True, "batch_size": 8}

    assert local_embed(provider=provider).call_defaults() == {"normalize": False, "batch_size": 32}
    with pytest.raises(TypeError, match="LocalEmbeddingProvider"):
        local_embed(provider=object())
    with pytest.raises(LocalConfigurationError, match="dim"):
        local_embed(provider=provider, dim=0)
    with pytest.raises(LocalConfigurationError, match="batch_size"):
        local_embed(provider=provider, batch_size=0)
    with pytest.raises(LocalConfigurationError, match="normalize"):
        local_embed(provider=provider, normalize="yes")


def test_provider_constructor_validation() -> None:
    backend = FakeEmbeddingBackend()

    with pytest.raises(LocalConfigurationError, match="model is required"):
        LocalEmbeddingProvider(model="", backend=backend)
    with pytest.raises(LocalConfigurationError, match="backend is required"):
        LocalEmbeddingProvider(model="m", backend=None)
    with pytest.raises(LocalConfigurationError, match="device"):
        LocalEmbeddingProvider(model="m", backend=backend, device="")
    with pytest.raises(LocalConfigurationError, match="max_length"):
        LocalEmbeddingProvider(model="m", backend=backend, max_length=0)
    with pytest.raises(LocalConfigurationError, match="trust_remote_code"):
        LocalEmbeddingProvider(model="m", backend=backend, trust_remote_code="yes")
    with pytest.raises(LocalConfigurationError, match="name"):
        LocalEmbeddingProvider(model="m", backend=object())
    with pytest.raises(TypeError, match="base class"):
        LocalProvider(model="m", backend=backend)

    with pytest.raises(LocalBackendNotSupportedError, match="unknown local backend"):
        LocalEmbeddingProvider(model="m", backend="no-such-backend")

    with pytest.raises(LocalBackendNotSupportedError, match="does not implement embedding"):
        LocalEmbeddingProvider(model="m", backend=FakeRerankBackend())

    cache_dir = LocalEmbeddingProvider(model="m", backend=backend, cache_dir="~/cliver-models").cache_dir
    assert cache_dir == os.path.abspath(os.path.expanduser("~/cliver-models"))


def test_builtin_backends_resolve_lazily() -> None:
    assert resolve_backend("sentence-transformers").name == "sentence-transformers"
    assert resolve_backend("flag-embedding").name == "flag-embedding"


def test_custom_backend_registration() -> None:
    backend = FakeEmbeddingBackend()
    register_backend("unit-test-backend", backend)
    assert resolve_backend("unit-test-backend") is backend
    with pytest.raises(LocalConfigurationError, match="already registered"):
        register_backend("unit-test-backend", backend)


# ── Lifecycle ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_preload_info_and_close_semantics() -> None:
    backend = FakeEmbeddingBackend()
    provider = _embedding_provider(backend, max_length=256)

    before = provider.info()
    assert before.loaded is False
    assert before.closed is False
    assert before.dim is None
    assert before.max_length == 256

    await provider.preload(warmup=True)
    after = provider.info()
    assert after.loaded is True
    assert after.closed is False
    assert after.resolved_model == "/fake/embedder"
    assert after.revision == "sha-embed"
    assert after.device == "cpu"
    assert after.dtype == "float32"
    assert after.dim == 2
    assert after.max_length == 256
    assert after.load_seconds is not None
    assert backend.embed_calls == [{"texts": ["warmup"], "batch_size": 1}]

    await provider.close()
    assert provider.info().closed is True
    assert provider.info().loaded is False
    assert backend.unload_calls == 1

    await provider.close()  # idempotent
    assert backend.unload_calls == 1

    with pytest.raises(LocalProviderClosedError, match="closed"):
        await provider.preload()
    with pytest.raises(LocalProviderClosedError, match="closed"):
        await provider.embed(["x"], model="fake/model")


@pytest.mark.asyncio
async def test_binding_never_closes_a_supplied_provider() -> None:
    provider = _embedding_provider(FakeEmbeddingBackend())
    bindings = ModelBindings(embed=local_embed(provider=provider))

    await bindings.aclose()

    assert provider.info().closed is False
    await provider.close()


@pytest.mark.asyncio
async def test_concurrent_calls_load_once_and_serialize_inference() -> None:
    backend = _SlowEmbeddingBackend(delay=0.05)
    provider = _embedding_provider(backend)

    results = await asyncio.gather(*(provider.embed([f"t{index}"], model="fake/model") for index in range(4)))

    assert backend.load_calls == 1
    assert backend.max_active == 1
    assert all(len(result.vectors) == 1 for result in results)
    await provider.close()


@pytest.mark.asyncio
async def test_close_waits_for_in_flight_inference() -> None:
    backend = _SlowEmbeddingBackend(delay=0.2)
    provider = _embedding_provider(backend)
    await provider.preload()

    task = asyncio.create_task(provider.embed(["x"], model="fake/model"))
    assert await asyncio.to_thread(backend.embed_started.wait, 2.0)

    await provider.close()
    result = await task

    assert result.vectors == [[3.0, 4.0]]
    assert backend.unload_calls == 1


@pytest.mark.asyncio
async def test_failed_load_is_retried_on_the_next_call() -> None:
    backend = _FailingOnceBackend()
    provider = _embedding_provider(backend)

    with pytest.raises(RuntimeError, match="boom"):
        await provider.embed(["x"], model="fake/model")

    result = await provider.embed(["x"], model="fake/model")
    assert result.vectors == [[3.0, 4.0]]
    assert backend.load_calls == 2
    assert provider.info().loaded is True
    await provider.close()


@pytest.mark.asyncio
async def test_chat_and_stream_are_not_supported() -> None:
    provider = _embedding_provider(FakeEmbeddingBackend())

    with pytest.raises(CapabilityNotSupported, match="chat"):
        await provider.chat(None)
    with pytest.raises(CapabilityNotSupported, match="streaming"):
        await provider.stream(None).__anext__()


# ── Rerank ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_rerank_sorts_descending_and_applies_top_k() -> None:
    backend = FakeRerankBackend(scores=[0.1, 0.9, 0.5])
    provider = _rerank_provider(backend)
    binding = local_rerank(provider=provider, top_k=2)

    items = await rerank(query="q", documents=["a", "b", "c"], binding=binding)

    assert [item.index for item in items] == [1, 2]
    assert [item.relevance_score for item in items] == [0.9, 0.5]
    assert [item.document for item in items] == ["b", "c"]
    assert backend.score_calls[0]["pairs"] == [("q", "a"), ("q", "b"), ("q", "c")]
    assert backend.score_calls[0]["batch_size"] == 32
    await provider.close()


@pytest.mark.asyncio
async def test_rerank_ties_keep_input_order() -> None:
    provider = _rerank_provider(FakeRerankBackend(scores=[0.5, 0.5, 0.5]))
    items = await rerank(query="q", documents=["a", "b", "c"], binding=local_rerank(provider=provider))
    assert [item.index for item in items] == [0, 1, 2]
    await provider.close()


@pytest.mark.asyncio
async def test_rerank_normalize_is_a_stable_sigmoid() -> None:
    backend = FakeRerankBackend(scores=[0.0, math.log(3.0), -1000.0])
    provider = _rerank_provider(backend)
    binding = local_rerank(provider=provider, normalize=True)

    items = await rerank(query="q", documents=["a", "b", "c"], binding=binding)

    by_index = {item.index: item.relevance_score for item in items}
    assert by_index[0] == pytest.approx(0.5)
    assert by_index[1] == pytest.approx(0.75)
    assert by_index[2] == pytest.approx(0.0)
    await provider.close()


@pytest.mark.asyncio
async def test_rerank_empty_documents_and_top_k_zero_skip_loading() -> None:
    backend = FakeRerankBackend()
    provider = _rerank_provider(backend)
    binding = local_rerank(provider=provider, top_k=0)

    assert await rerank(query="q", documents=[], binding=binding) == []
    assert await rerank(query="q", documents=["a"], binding=binding) == []
    assert backend.load_calls == 0


@pytest.mark.asyncio
async def test_rerank_call_site_top_k_overrides_the_binding_default() -> None:
    provider = _rerank_provider(FakeRerankBackend())
    binding = local_rerank(provider=provider, top_k=3)
    items = await rerank(query="q", documents=["a", "b", "c"], binding=binding, top_k=1)
    assert len(items) == 1
    assert items[0].index == 1
    await provider.close()


@pytest.mark.asyncio
async def test_rerank_rejects_invalid_options() -> None:
    provider = _rerank_provider(FakeRerankBackend())

    with pytest.raises(LocalConfigurationError, match="top_k"):
        await rerank(query="q", documents=["a"], binding=local_rerank(provider=provider), top_k=-1)
    with pytest.raises(LocalConfigurationError, match="unsupported rerank option 'dim'"):
        await provider.rerank("q", ["a"], model="fake/reranker", dim=2)
    with pytest.raises(LocalConfigurationError, match="no endpoint path"):
        await provider.rerank("q", ["a"], model="fake/reranker", path="/rerank")
    with pytest.raises(LocalConfigurationError, match="load-time"):
        await provider.rerank("q", ["a"], model="fake/reranker", device="cpu")
    with pytest.raises(LocalConfigurationError, match="does not match"):
        await provider.rerank("q", ["a"], model="other/model")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("scores", "match"),
    [
        ([0.1], "1 scores for 2 documents"),
        (["nope", 0.2], "non-numeric score"),
        ([float("nan"), 0.2], "non-finite score"),
    ],
)
async def test_rerank_rejects_bad_backend_output(scores: list[float], match: str) -> None:
    provider = _rerank_provider(FakeRerankBackend(scores=scores))
    with pytest.raises(LocalInferenceError, match=match):
        await provider.rerank("q", ["a", "b"], model="fake/reranker")


def test_local_rerank_factory_validates_and_forwards_defaults() -> None:
    provider = _rerank_provider(FakeRerankBackend())
    binding = local_rerank(provider=provider, top_k=2, normalize=True, batch_size=4)

    assert binding.model == "fake/reranker"
    assert binding.protocol == "local"
    assert binding.purpose == "rerank"
    assert binding.call_defaults() == {"top_k": 2, "normalize": True, "batch_size": 4}

    with pytest.raises(TypeError, match="LocalRerankProvider"):
        local_rerank(provider=object())
    with pytest.raises(LocalConfigurationError, match="top_k"):
        local_rerank(provider=provider, top_k=-1)
    with pytest.raises(LocalBackendNotSupportedError, match="does not implement rerank"):
        LocalRerankProvider(model="m", backend=FakeEmbeddingBackend())


@pytest.mark.asyncio
async def test_rerank_close_releases_the_model() -> None:
    backend = FakeRerankBackend()
    provider = _rerank_provider(backend)
    await provider.preload(warmup=True)
    assert backend.score_calls == [{"pairs": [("warmup", "warmup")], "batch_size": 1}]

    await provider.close()
    assert backend.unload_calls == 1
    with pytest.raises(LocalProviderClosedError):
        await provider.rerank("q", ["a"], model="fake/reranker")


# ── AgentCore integration ───────────────────────────────────


@pytest.mark.asyncio
async def test_agent_core_routes_embed_to_the_local_provider_without_closing_it() -> None:
    backend = FakeEmbeddingBackend()
    provider = _embedding_provider(backend)
    agent = AgentCore(
        ModelBindings(
            llm=ModelBinding.llm(model="main-model", provider=_StubChatProvider()),
            embed=local_embed(provider=provider, normalize=True),
        )
    )

    result = await agent.embed(texts=["a", "b"])
    assert result.vectors[0] == pytest.approx([0.6, 0.8])

    await agent.aclose()
    assert provider.info().closed is False
    await provider.close()


# ── Import guard ────────────────────────────────────────────


BLOCKED = frozenset({"torch", "transformers", "sentence_transformers", "FlagEmbedding", "huggingface_hub", "laya"})

_IMPORT_GUARD_SCRIPT = f"""
import asyncio
import importlib.abc
import sys

BLOCKED = {set(BLOCKED)!r}

class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        top = fullname.split(".", 1)[0]
        if top in BLOCKED:
            raise ModuleNotFoundError(f"No module named '{{fullname}}'", name=fullname)
        return None

sys.meta_path.insert(0, Blocker())

import cliver_core.local as local
from cliver_core.local import LoadedEmbedding

class FakeBackend:
    name = "fake"

    def load_embedding(self, spec):
        return LoadedEmbedding(
            handle=object(),
            resolved_model="/fake",
            revision=None,
            device="cpu",
            dtype="float32",
            dim=2,
            max_length=None,
        )

    def embed(self, loaded, texts, *, batch_size):
        return [[3.0, 4.0] for _ in texts]

    def unload_embedding(self, loaded):
        loaded.handle = None

provider = local.LocalEmbeddingProvider(model="fake/model", backend=FakeBackend())

# Each call gets its own event loop: the provider must not be loop-bound.
asyncio.run(provider.preload())
result = asyncio.run(provider.embed(["x"], model="fake/model"))
assert result.vectors == [[3.0, 4.0]], result.vectors
asyncio.run(provider.close())

try:
    asyncio.run(provider.preload())
except local.LocalProviderClosedError:
    pass
else:
    raise AssertionError("closed provider accepted a call")

leaked = [name for name in sys.modules if name.split(".", 1)[0] in BLOCKED]
assert not leaked, f"heavy modules were imported: {{leaked}}"
print("OK")
"""


def test_local_package_is_usable_without_heavy_dependencies() -> None:
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(_IMPORT_GUARD_SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
    assert "OK" in result.stdout
