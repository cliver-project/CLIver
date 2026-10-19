"""Gated live tests for the local backends.

Nothing here runs by default.  Enable with ``CLIVER_LOCAL_MODEL_TESTS=1`` and
install the matching extra; the first run downloads the model into the
Hugging Face cache::

    CLIVER_LOCAL_MODEL_TESTS=1 uv run pytest \
        packages/cliver-core/src/cliver_core/tests/test_local_live.py -q

Models are plain parameters and are **not** bound to library code.  Override
them for a different checkpoint::

    CLIVER_LOCAL_EMBED_MODEL   (default BAAI/bge-base-zh-v1.5)
    CLIVER_LOCAL_RERANK_MODEL  (default BAAI/bge-reranker-v2-m3)
    CLIVER_LOCAL_CACHE_DIR     (optional Hugging Face cache override)
    CLIVER_LOCAL_DEVICE        (default cpu; set cuda to exercise a GPU)

A Hugging Face token, when needed, comes from ``HF_TOKEN`` in the environment.
"""

from __future__ import annotations

import math
import os

import pytest

from cliver_core.capabilities import embed, rerank
from cliver_core.local import (
    LocalDimensionError,
    LocalEmbeddingProvider,
    LocalRerankProvider,
    local_embed,
    local_rerank,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("CLIVER_LOCAL_MODEL_TESTS") != "1",
    reason="set CLIVER_LOCAL_MODEL_TESTS=1 to run live local-model tests",
)

_DEVICE = os.environ.get("CLIVER_LOCAL_DEVICE", "cpu")
_CACHE_DIR = os.environ.get("CLIVER_LOCAL_CACHE_DIR") or None
_EMBED_MODEL = os.environ.get("CLIVER_LOCAL_EMBED_MODEL", "BAAI/bge-base-zh-v1.5")
_RERANK_MODEL = os.environ.get("CLIVER_LOCAL_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")


def _cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


@pytest.mark.asyncio
async def test_live_sentence_transformers_embedding() -> None:
    pytest.importorskip("sentence_transformers")
    provider = LocalEmbeddingProvider(
        model=_EMBED_MODEL,
        backend="sentence-transformers",
        device=_DEVICE,
        cache_dir=_CACHE_DIR,
    )
    try:
        binding = local_embed(provider=provider, normalize=True)
        result = await embed(
            texts=[
                "今天天气真好。",
                "阳光明媚，适合出门散步。",
                "量子力学是物理学的一个分支。",
            ],
            binding=binding,
        )

        assert len(result.vectors) == 3
        dimensions = {len(vector) for vector in result.vectors}
        assert len(dimensions) == 1
        dim = dimensions.pop()
        assert dim > 0
        for vector in result.vectors:
            assert math.isclose(math.hypot(*vector), 1.0, rel_tol=1e-4, abs_tol=1e-4)

        # Normalized vectors: the dot product is the cosine similarity.  The
        # two weather sentences must be closer than the physics sentence.
        assert _cosine(result.vectors[0], result.vectors[1]) > _cosine(result.vectors[0], result.vectors[2])

        info = provider.info()
        assert info.loaded is True
        assert info.closed is False
        assert info.resolved_model
        assert info.revision
        assert info.device
        assert info.dtype
        assert info.dim == dim

        # dim is a caller parameter, never inferred from the model name.
        checked = await embed(texts=["x"], binding=local_embed(provider=provider, dim=dim))
        assert len(checked.vectors[0]) == dim
        with pytest.raises(LocalDimensionError):
            await embed(texts=["x"], binding=local_embed(provider=provider, dim=dim + 1))
    finally:
        await provider.close()

    assert provider.info().closed is True


@pytest.mark.asyncio
async def test_live_flag_embedding_rerank() -> None:
    pytest.importorskip("FlagEmbedding")
    provider = LocalRerankProvider(
        model=_RERANK_MODEL,
        backend="flag-embedding",
        device=_DEVICE,
        cache_dir=_CACHE_DIR,
    )
    try:
        documents = [
            "量子力学是物理学的一个分支。",
            "今天天气真好，阳光明媚。",
            "苹果是一种常见的水果。",
        ]
        items = await rerank(
            query="今天天气怎么样？",
            documents=documents,
            binding=local_rerank(provider=provider),
        )

        assert len(items) == 3
        assert items[0].index == 1
        assert items[0].document == documents[1]
        scores = [item.relevance_score for item in items]
        assert scores == sorted(scores, reverse=True)
        assert all(math.isfinite(score) for score in scores)

        normalized = await rerank(
            query="今天天气怎么样？",
            documents=documents,
            binding=local_rerank(provider=provider, normalize=True, top_k=2),
        )
        assert len(normalized) == 2
        assert all(0.0 <= item.relevance_score <= 1.0 for item in normalized)

        info = provider.info()
        assert info.loaded is True
        assert info.closed is False
        assert info.resolved_model
        assert info.revision
        assert info.device
        assert info.dtype
    finally:
        await provider.close()

    assert provider.info().closed is True
