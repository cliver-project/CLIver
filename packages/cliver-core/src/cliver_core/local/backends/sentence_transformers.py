"""sentence-transformers embedding backend.

Loads any model ``SentenceTransformer`` understands from a local directory.
The model reference has already been resolved by :mod:`cliver_core.local.
resolution`; this module only resolves the runtime device/dtype and calls the
library.

The library import happens inside :meth:`load_embedding`, so importing this
module is cheap and torch-free.
"""

from __future__ import annotations

from cliver_core.local.backends.base import LoadedEmbedding, LoadSpec
from cliver_core.local.backends.common import (
    empty_device_cache,
    resolve_device,
    resolve_dtype,
    sentence_transformers_dtype_kwarg,
    torch_dtype,
    wrap_load_error,
)
from cliver_core.local.errors import LocalDependencyError, LocalInferenceError, LocalModelError
from cliver_core.local.resolution import resolve_local_model

__all__ = ["SentenceTransformerBackend"]


class SentenceTransformerBackend:
    """Generic Hugging Face embedding models via sentence-transformers."""

    name = "sentence-transformers"

    def load_embedding(self, spec: LoadSpec) -> LoadedEmbedding:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise LocalDependencyError(
                "local backend 'sentence-transformers' requires the optional package "
                "'sentence-transformers'. Install it with: pip install \"cliver-core[local-st]\" "
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
        model_kwargs = None
        if dtype != "float32":
            model_kwargs = {sentence_transformers_dtype_kwarg(): torch_dtype(dtype)}

        try:
            model = SentenceTransformer(
                resolved_path,
                device=device,
                cache_folder=spec.cache_dir,
                trust_remote_code=spec.trust_remote_code,
                local_files_only=True,
                model_kwargs=model_kwargs,
            )
        except Exception as exc:
            raise wrap_load_error(exc, model=spec.model, backend=self.name) from exc

        if spec.max_length is not None:
            model.max_seq_length = spec.max_length

        # sentence-transformers 6 renamed this method; prefer the current name
        # and fall back for older releases.
        get_dimension = getattr(model, "get_embedding_dimension", None)
        if get_dimension is None:
            get_dimension = model.get_sentence_embedding_dimension
        dim = get_dimension()
        return LoadedEmbedding(
            handle=model,
            resolved_model=resolved_path,
            revision=revision,
            device=device,
            dtype=dtype,
            dim=dim if isinstance(dim, int) else None,
            max_length=spec.max_length,
        )

    def embed(self, loaded: LoadedEmbedding, texts: list[str], *, batch_size: int) -> list[list[float]]:
        try:
            vectors = loaded.handle.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=False,
                convert_to_numpy=True,
                convert_to_tensor=False,
                show_progress_bar=False,
            )
        except LocalModelError:
            raise
        except Exception as exc:
            raise LocalInferenceError(
                f"embedding with backend {self.name!r} failed for model {loaded.resolved_model!r}: {exc}"
            ) from exc
        return [[float(value) for value in row] for row in vectors]

    def unload_embedding(self, loaded: LoadedEmbedding) -> None:
        loaded.handle = None
        empty_device_cache(loaded.device)
