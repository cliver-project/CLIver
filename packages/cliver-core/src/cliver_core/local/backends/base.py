"""Backend contract for in-process local models.

A backend is the adapter between a model-loading library (sentence-
transformers, FlagEmbedding, ...) and the local providers.  Backends are
stateless: every ``load_*`` call returns an opaque ``Loaded*`` handle that the
provider owns and later passes back.

Heavy imports belong **inside** ``load_*``; importing a backend module must not
import torch, sentence-transformers, or FlagEmbedding.  This is what keeps the
base install light and lets unit tests run with a fake backend.

The method names are purpose-specific (``load_embedding`` / ``score``) so one
class can implement several local capabilities — the same shape future local
TTS / ASR / image / video backends should follow.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "EmbeddingBackend",
    "LoadSpec",
    "LoadedEmbedding",
    "LoadedRerank",
    "RerankBackend",
]


@dataclass(frozen=True)
class LoadSpec:
    """Everything a backend needs to resolve and load one model.

    ``device`` and ``dtype`` are the requested values (possibly ``"auto"``);
    the backend resolves them against the runtime and reports the result in
    the ``Loaded*`` handle.
    """

    model: str
    device: str
    dtype: str
    revision: str | None
    trust_remote_code: bool
    cache_dir: str | None
    local_files_only: bool
    max_length: int | None
    logger: logging.Logger


@dataclass
class LoadedEmbedding:
    """Opaque embedding model handle plus resolved metadata."""

    handle: Any
    resolved_model: str
    revision: str | None
    device: str
    dtype: str
    dim: int | None
    max_length: int | None


@dataclass
class LoadedRerank:
    """Opaque rerank model handle plus resolved metadata."""

    handle: Any
    resolved_model: str
    revision: str | None
    device: str
    dtype: str
    max_length: int | None


@runtime_checkable
class EmbeddingBackend(Protocol):
    """Loads a model and turns texts into raw (unnormalized) vectors.

    ``embed`` must return exactly one row per input text, in input order.
    Normalization is the provider's job, not the backend's.
    """

    name: str

    def load_embedding(self, spec: LoadSpec) -> LoadedEmbedding: ...

    def embed(self, loaded: LoadedEmbedding, texts: list[str], *, batch_size: int) -> list[list[float]]: ...

    def unload_embedding(self, loaded: LoadedEmbedding) -> None: ...


@runtime_checkable
class RerankBackend(Protocol):
    """Loads a model and scores ``(query, document)`` pairs.

    ``score`` must return one raw score per pair, in input order.  Normalizing
    scores is the provider's job.
    """

    name: str

    def load_rerank(self, spec: LoadSpec) -> LoadedRerank: ...

    def score(self, loaded: LoadedRerank, pairs: list[tuple[str, str]], *, batch_size: int) -> list[float]: ...

    def unload_rerank(self, loaded: LoadedRerank) -> None: ...
