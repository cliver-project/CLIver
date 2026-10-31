"""Binding factories for in-process local models.

The factories here mirror the remote per-purpose factories on
:class:`~cliver_core.bindings.ModelBinding`, but take a **provider object**
instead of endpoint settings.  A local provider is long-lived and owns its
loaded model; the binding only carries it, the model id, and generic call
defaults (``dim`` / ``normalize`` / ``batch_size`` / ``top_k`` / ``max_len``).

Because the provider is caller-supplied, ``ModelBinding.aclose()`` never closes
it — the application owns the provider lifecycle::

    provider = LocalEmbeddingProvider(model="BAAI/bge-base-zh-v1.5", backend="sentence-transformers")
    bindings = ModelBindings(
        llm=...,
        embed=local_embed(provider=provider, normalize=True),
    )
    ...
    await bindings.aclose()   # leaves ``provider`` open
    await provider.close()    # the application closes the model

Future local purposes should add ``local_<purpose>`` factories with the same
shape: a validated provider object plus only generic, explicitly-passed
parameters.
"""

from __future__ import annotations

from cliver_core.bindings import ModelBinding
from cliver_core.local.providers import (
    LocalEmbeddingProvider,
    LocalRerankProvider,
    LocalSystemOneProvider,
    _as_bool,
    _as_optional_non_negative_int,
    _as_optional_positive_int,
    _as_positive_int,
)
from cliver_core.util import drop_none

__all__ = ["local_embed", "local_rerank", "local_system_one"]


def _require_embedding_provider(provider: object) -> LocalEmbeddingProvider:
    if not isinstance(provider, LocalEmbeddingProvider):
        raise TypeError(
            "local_embed() requires a LocalEmbeddingProvider; "
            "construct one with LocalEmbeddingProvider(model=..., backend=...)"
        )
    return provider


def _require_rerank_provider(provider: object) -> LocalRerankProvider:
    if not isinstance(provider, LocalRerankProvider):
        raise TypeError(
            "local_rerank() requires a LocalRerankProvider; "
            "construct one with LocalRerankProvider(model=..., backend=...)"
        )
    return provider


def _require_system_one_provider(provider: object) -> LocalSystemOneProvider:
    if not isinstance(provider, LocalSystemOneProvider):
        raise TypeError(
            "local_system_one() requires a LocalSystemOneProvider; "
            "construct one with LocalSystemOneProvider(model=..., backend=...)"
        )
    return provider


def local_embed(
    *,
    provider: LocalEmbeddingProvider,
    dim: int | None = None,
    normalize: bool = False,
    batch_size: int = 32,
) -> ModelBinding:
    """Build an embedding binding around a local provider.

    Args:
        provider: The long-lived :class:`LocalEmbeddingProvider` that owns the
            loaded model.  The binding never closes it.
        dim: Optional output-dimension check.  When set, every call validates
            the model's output size against it; the library never infers a
            model's dimension from its name.
        normalize: L2-normalize returned vectors.  Off by default — vectors
            are raw model output.
        batch_size: Inference batch size (positive integer).
    """
    provider = _require_embedding_provider(provider)
    return ModelBinding.embed(
        model=provider.model,
        provider=provider,
        protocol="local",
        **drop_none(dim=_as_optional_positive_int(dim, "dim")),
        normalize=_as_bool(normalize, "normalize"),
        batch_size=_as_positive_int(batch_size, "batch_size"),
    )


def local_rerank(
    *,
    provider: LocalRerankProvider,
    top_k: int | None = None,
    normalize: bool = False,
    batch_size: int = 32,
) -> ModelBinding:
    """Build a rerank binding around a local provider.

    Args:
        provider: The long-lived :class:`LocalRerankProvider` that owns the
            loaded model.  The binding never closes it.
        top_k: Default number of items to return.  Applied after sorting; the
            call site may override it.
        normalize: Apply a stable Sigmoid to scores, yielding ``[0, 1]``
            relevance scores.  Off by default — scores are raw model output.
        batch_size: Inference batch size (positive integer).
    """
    provider = _require_rerank_provider(provider)
    return ModelBinding.rerank(
        model=provider.model,
        provider=provider,
        protocol="local",
        top_k=_as_optional_non_negative_int(top_k, "top_k"),
        normalize=_as_bool(normalize, "normalize"),
        batch_size=_as_positive_int(batch_size, "batch_size"),
    )


def local_system_one(
    *,
    provider: LocalSystemOneProvider,
    max_len: int | None = None,
) -> ModelBinding:
    """Build a System One decision binding around a local provider.

    Args:
        provider: The long-lived :class:`LocalSystemOneProvider` that owns the
            loaded model.  The binding never closes it.
        max_len: Default token budget for the state (positive integer).  The
            call site may override it; when unset the provider's
            ``max_length`` applies.
    """
    provider = _require_system_one_provider(provider)
    return ModelBinding.system_one(
        model=provider.model,
        provider=provider,
        protocol="local",
        **drop_none(max_len=_as_optional_positive_int(max_len, "max_len")),
    )
