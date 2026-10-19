"""FlagEmbedding backend — dense embeddings and cross-encoder rerank.

One class implements both local capabilities because FlagEmbedding exposes
both behind the same package: ``BGEM3FlagModel`` for dense embeddings and
``FlagReranker`` for reranking.  The model id is still a plain parameter; the
backend only knows the library API.

The library import happens inside the ``load_*`` methods, so importing this
module is cheap and torch-free.
"""

from __future__ import annotations

from typing import Any

from cliver_core.local.backends.base import LoadedEmbedding, LoadedRerank, LoadSpec
from cliver_core.local.backends.common import (
    empty_device_cache,
    resolve_device,
    resolve_dtype,
    wrap_load_error,
)
from cliver_core.local.errors import (
    LocalDependencyError,
    LocalInferenceError,
    LocalModelError,
)
from cliver_core.local.resolution import resolve_local_model

__all__ = ["FlagEmbeddingBackend"]


def _hidden_size(model: Any) -> int | None:
    """Best-effort output dimension from the underlying transformers config."""
    for attribute in ("model", "embedder"):
        inner = getattr(model, attribute, None)
        config = getattr(inner, "config", None)
        dim = getattr(config, "hidden_size", None)
        if isinstance(dim, int) and dim > 0:
            return dim
    return None


class FlagEmbeddingBackend:
    """FlagEmbedding dense embedding and rerank models."""

    name = "flag-embedding"

    def load_embedding(self, spec: LoadSpec) -> LoadedEmbedding:
        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError as exc:
            raise LocalDependencyError(
                "local backend 'flag-embedding' requires the optional package "
                "'FlagEmbedding'. Install it with: pip install \"cliver-core[local-flag]\" "
                '(or "cliver-core[local]" for all local backends).'
            ) from exc

        resolved_path, revision = resolve_local_model(
            spec.model,
            revision=spec.revision,
            cache_dir=spec.cache_dir,
            local_files_only=spec.local_files_only,
            logger=spec.logger,
        )
        device = resolve_device(spec.device)
        dtype = resolve_dtype(spec.dtype, device)

        try:
            model = BGEM3FlagModel(
                resolved_path,
                normalize_embeddings=False,
                use_fp16=dtype == "float16",
                use_bf16=dtype == "bfloat16",
                devices=device,
                trust_remote_code=spec.trust_remote_code,
            )
        except Exception as exc:
            raise wrap_load_error(exc, model=spec.model, backend=self.name) from exc

        return LoadedEmbedding(
            handle=model,
            resolved_model=resolved_path,
            revision=revision,
            device=device,
            dtype=dtype,
            dim=_hidden_size(model),
            max_length=spec.max_length,
        )

    def embed(self, loaded: LoadedEmbedding, texts: list[str], *, batch_size: int) -> list[list[float]]:
        kwargs: dict[str, Any] = {
            "batch_size": batch_size,
            "return_dense": True,
            "return_sparse": False,
            "return_colbert_vecs": False,
        }
        if loaded.max_length is not None:
            kwargs["max_length"] = loaded.max_length
        try:
            result = loaded.handle.encode(texts, **kwargs)
            dense = result["dense_vecs"]
        except LocalModelError:
            raise
        except Exception as exc:
            raise LocalInferenceError(
                f"embedding with backend {self.name!r} failed for model {loaded.resolved_model!r}: {exc}"
            ) from exc
        return [[float(value) for value in row] for row in dense]

    def unload_embedding(self, loaded: LoadedEmbedding) -> None:
        loaded.handle = None
        empty_device_cache(loaded.device)

    def load_rerank(self, spec: LoadSpec) -> LoadedRerank:
        try:
            from FlagEmbedding import FlagReranker
        except ImportError as exc:
            raise LocalDependencyError(
                "local backend 'flag-embedding' requires the optional package "
                "'FlagEmbedding'. Install it with: pip install \"cliver-core[local-flag]\" "
                '(or "cliver-core[local]" for all local backends).'
            ) from exc

        resolved_path, revision = resolve_local_model(
            spec.model,
            revision=spec.revision,
            cache_dir=spec.cache_dir,
            local_files_only=spec.local_files_only,
            logger=spec.logger,
        )
        device = resolve_device(spec.device)
        dtype = resolve_dtype(spec.dtype, device, allow_bfloat16=False)

        kwargs: dict[str, Any] = {
            "use_fp16": dtype == "float16",
            "normalize": False,
            "devices": device,
            "trust_remote_code": spec.trust_remote_code,
        }
        if spec.max_length is not None:
            kwargs["max_length"] = spec.max_length
        try:
            model = FlagReranker(resolved_path, **kwargs)
        except Exception as exc:
            raise wrap_load_error(exc, model=spec.model, backend=self.name) from exc

        return LoadedRerank(
            handle=model,
            resolved_model=resolved_path,
            revision=revision,
            device=device,
            dtype=dtype,
            max_length=spec.max_length,
        )

    def score(self, loaded: LoadedRerank, pairs: list[tuple[str, str]], *, batch_size: int) -> list[float]:
        kwargs: dict[str, Any] = {"batch_size": batch_size, "normalize": False}
        if loaded.max_length is not None:
            kwargs["max_length"] = loaded.max_length
        try:
            raw = loaded.handle.compute_score(pairs, **kwargs)
        except LocalModelError:
            raise
        except Exception as exc:
            raise LocalInferenceError(
                f"rerank with backend {self.name!r} failed for model {loaded.resolved_model!r}: {exc}"
            ) from exc
        if isinstance(raw, (int, float)):
            raw = [raw]
        return [float(value) for value in raw]

    def unload_rerank(self, loaded: LoadedRerank) -> None:
        loaded.handle = None
        empty_device_cache(loaded.device)
