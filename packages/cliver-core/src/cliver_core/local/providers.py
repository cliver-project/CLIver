"""In-process local model providers.

A local provider owns exactly one loaded model per provider instance.  It is
designed to be **long-lived**: create one provider per model configuration and
reuse it across bindings and calls.  There is deliberately no process-wide
model cache — sharing is expressed by sharing the provider object, which the
existing ``ModelBinding(provider=...)`` seam already supports.

Blocking work (model download/load and inference) runs in ``asyncio.to_thread``
and is synchronized with plain ``threading`` primitives, so a provider is safe
to use from the short-lived event loops an application may create per call.

The API surface is intentionally small and explicit:

* the model id and backend are parameters, never inferred from a model name;
* options are limited to generic inference knobs (``dim``, ``normalize``,
  ``batch_size``, ``top_k``); query/document semantics and instruction prefixes
  belong to the application;
* ``chat`` / ``stream`` are unsupported and raise
  :class:`~cliver_core.errors.CapabilityNotSupported`.

Future local purposes (TTS, ASR, image, video, ...) should subclass
:class:`LocalProvider`, implement the matching provider method, and add a
``local_<purpose>`` binding factory that only forwards generic parameters.
"""

from __future__ import annotations

import asyncio
import logging
import math
import numbers
import os
import threading
import time
from collections.abc import AsyncIterator
from typing import Any

from cliver_core.errors import CapabilityNotSupported
from cliver_core.local.backends import (
    EmbeddingBackend,
    LoadedEmbedding,
    LoadedRerank,
    LoadSpec,
    RerankBackend,
    resolve_backend,
)
from cliver_core.local.errors import (
    LocalBackendNotSupportedError,
    LocalConfigurationError,
    LocalDimensionError,
    LocalInferenceError,
    LocalProviderClosedError,
)
from cliver_core.local.info import LocalModelInfo
from cliver_core.messages import CLIverMessageChunk
from cliver_core.provider import CLIverRequest, CLIverResponse, EmbeddingsResult, Provider, RerankItem

__all__ = ["LocalEmbeddingProvider", "LocalProvider", "LocalRerankProvider"]

_LOAD_TIME_OPTIONS = frozenset(
    {
        "backend",
        "cache_dir",
        "device",
        "dtype",
        "local_files_only",
        "max_length",
        "revision",
        "trust_remote_code",
    }
)


# ── Option helpers ─────────────────────────────────────────────


def _require_model(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LocalConfigurationError("model is required and must be a non-empty string")
    return value.strip()


def _text_or_default(value: Any, name: str, default: str) -> str:
    if value is None:
        return default
    if not isinstance(value, str) or not value.strip():
        raise LocalConfigurationError(f"{name} must be a non-empty string")
    return value.strip()


def _optional_text(value: Any, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise LocalConfigurationError(f"{name} must be a string or None")
    return value.strip() or None


def _as_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise LocalConfigurationError(f"{name} must be a bool, got {type(value).__name__}")
    return value


def _as_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or int(value) <= 0:
        raise LocalConfigurationError(f"{name} must be a positive integer")
    return int(value)


def _as_optional_positive_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    return _as_positive_int(value, name)


def _as_optional_non_negative_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or int(value) < 0:
        raise LocalConfigurationError(f"{name} must be a non-negative integer or None")
    return int(value)


def _resolve_cache_dir(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, os.PathLike):
        value = os.fspath(value)
    if not isinstance(value, str):
        raise LocalConfigurationError("cache_dir must be a path-like string or None")
    value = value.strip()
    return os.path.abspath(os.path.expanduser(value)) if value else None


def _resolve_backend_value(value: Any) -> Any:
    if isinstance(value, str):
        return resolve_backend(value)
    if value is None:
        raise LocalConfigurationError("backend is required; pass a backend name or a backend object")
    name = getattr(value, "name", None)
    if not isinstance(name, str) or not name.strip():
        raise LocalConfigurationError("a backend object must expose a non-empty string 'name' attribute")
    return value


def _reject_option(key: str, *, allowed: frozenset[str], purpose: str) -> None:
    if key in allowed:
        return
    if key == "path":
        raise LocalConfigurationError("local providers have no endpoint path; remove path= from the binding or call")
    if key == "dimensions":
        raise LocalConfigurationError(
            "'dimensions' is an HTTP-only option; for local models use dim= to validate the output size"
        )
    if key in _LOAD_TIME_OPTIONS:
        raise LocalConfigurationError(
            f"{key!r} is a load-time option; pass it to the local provider constructor, not to {purpose}"
        )
    options = ", ".join(sorted(allowed))
    raise LocalConfigurationError(f"unsupported {purpose} option {key!r}; allowed options: {options}")


def _parse_embed_options(options: dict[str, Any]) -> dict[str, Any]:
    allowed = frozenset({"batch_size", "dim", "normalize"})
    for key in options:
        _reject_option(key, allowed=allowed, purpose="embedding")
    return {
        "normalize": _as_bool(options.get("normalize", False), "normalize"),
        "batch_size": _as_positive_int(options.get("batch_size", 32), "batch_size"),
        "dim": _as_optional_positive_int(options.get("dim"), "dim"),
    }


def _parse_rerank_options(options: dict[str, Any]) -> dict[str, Any]:
    allowed = frozenset({"batch_size", "normalize"})
    for key in options:
        _reject_option(key, allowed=allowed, purpose="rerank")
    return {
        "normalize": _as_bool(options.get("normalize", False), "normalize"),
        "batch_size": _as_positive_int(options.get("batch_size", 32), "batch_size"),
    }


# ── Result helpers ─────────────────────────────────────────────


def _validate_vectors(
    raw: list[list[float]],
    *,
    count: int,
    expected_dim: int | None,
    model: str,
) -> list[list[float]]:
    if len(raw) != count:
        raise LocalInferenceError(f"local model {model!r} returned {len(raw)} vectors for {count} texts")
    vectors: list[list[float]] = []
    dim = expected_dim
    for index, row in enumerate(raw):
        try:
            values = [float(value) for value in row]
        except (TypeError, ValueError) as exc:
            raise LocalInferenceError(f"local model {model!r} returned a non-numeric vector at index {index}") from exc
        if not values:
            raise LocalInferenceError(f"local model {model!r} returned an empty vector at index {index}")
        if dim is None:
            dim = len(values)
        elif len(values) != dim:
            raise LocalDimensionError(
                f"local model {model!r} returned a {len(values)}-d vector at index {index}, expected {dim}-d. "
                "Set dim to the model's real output size or remove dim=."
            )
        if not all(math.isfinite(value) for value in values):
            raise LocalInferenceError(f"local model {model!r} returned a non-finite value at index {index}")
        vectors.append(values)
    return vectors


def _normalize_vectors(vectors: list[list[float]]) -> list[list[float]]:
    normalized: list[list[float]] = []
    for values in vectors:
        norm = math.hypot(*values)
        if norm == 0.0:
            normalized.append(values)
        else:
            normalized.append([value / norm for value in values])
    return normalized


def _sigmoid(value: float) -> float:
    """Numerically stable logistic function."""
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


# ── Providers ──────────────────────────────────────────────────


class LocalProvider(Provider):
    """Base class for in-process local providers.

    Subclasses set :attr:`_backend_protocol` and :attr:`_capability`, then
    implement ``_load_backend`` / ``_unload_backend`` / ``_warmup`` and the
    capability method (``embed``, ``rerank``, ...).
    """

    protocol = "local"
    brand = "local"

    _backend_protocol: Any = None
    _capability: str = ""

    def __init__(
        self,
        *,
        model: str,
        backend: Any,
        device: str = "auto",
        dtype: str = "auto",
        revision: str | None = None,
        trust_remote_code: bool = False,
        cache_dir: str | os.PathLike[str] | None = None,
        local_files_only: bool = False,
        max_length: int | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__(api_key="", base_url="", logger=logger)
        if self._backend_protocol is None:
            raise TypeError(
                "LocalProvider is a base class; use LocalEmbeddingProvider or LocalRerankProvider "
                "(or subclass LocalProvider for a new local purpose)"
            )
        self.model = _require_model(model)
        self.backend = _resolve_backend_value(backend)
        if not isinstance(self.backend, self._backend_protocol):
            backend_name = getattr(self.backend, "name", type(self.backend).__name__)
            raise LocalBackendNotSupportedError(
                f"backend {backend_name!r} does not implement {self._capability}; "
                "choose a backend that supports it or pass a custom backend object"
            )
        self.device = _text_or_default(device, "device", "auto")
        self.dtype = _text_or_default(dtype, "dtype", "auto")
        self.revision = _optional_text(revision, "revision")
        self.trust_remote_code = _as_bool(trust_remote_code, "trust_remote_code")
        self.cache_dir = _resolve_cache_dir(cache_dir)
        self.local_files_only = _as_bool(local_files_only, "local_files_only")
        self.max_length = _as_optional_positive_int(max_length, "max_length")

        self._state_lock = threading.Lock()
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self._loaded: LoadedEmbedding | LoadedRerank | None = None
        self._closed = False
        self._load_seconds: float | None = None

    # ── Diagnostics / lifecycle ───────────────────────────────

    @property
    def backend_name(self) -> str:
        return getattr(self.backend, "name", type(self.backend).__name__)

    def _build_spec(self) -> LoadSpec:
        return LoadSpec(
            model=self.model,
            device=self.device,
            dtype=self.dtype,
            revision=self.revision,
            trust_remote_code=self.trust_remote_code,
            cache_dir=self.cache_dir,
            local_files_only=self.local_files_only,
            max_length=self.max_length,
            logger=self.logger,
        )

    def info(self) -> LocalModelInfo:
        """Return a snapshot of the provider's configuration and load state."""
        with self._state_lock:
            loaded = self._loaded
            closed = self._closed
            load_seconds = self._load_seconds
        return LocalModelInfo(
            model=self.model,
            backend=self.backend_name,
            loaded=loaded is not None,
            closed=closed,
            revision=loaded.revision if loaded is not None else None,
            resolved_model=loaded.resolved_model if loaded is not None else None,
            device=loaded.device if loaded is not None else None,
            dtype=loaded.dtype if loaded is not None else None,
            dim=getattr(loaded, "dim", None) if loaded is not None else None,
            max_length=loaded.max_length if loaded is not None else self.max_length,
            load_seconds=load_seconds,
        )

    async def preload(self, *, warmup: bool = False) -> None:
        """Load the model now (optionally run one small inference).

        Call this at application startup when the first request must not pay
        the download/load cost.  ``warmup`` runs one tiny inference after the
        load; it is off by default.
        """
        loaded = await asyncio.to_thread(self._load)
        if warmup:
            await asyncio.to_thread(self._warmup, loaded)

    async def close(self) -> None:
        """Unload the model.  Idempotent; later inference calls raise.

        Closing waits for an in-flight inference to finish.  A load that is
        already in progress finishes in its worker thread and releases the
        loaded model immediately, since the provider is already closed.
        """
        await asyncio.to_thread(self._close_sync)

    # ── Internal load/inference plumbing ──────────────────────

    def _load(self) -> LoadedEmbedding | LoadedRerank:
        with self._state_lock:
            if self._closed:
                raise LocalProviderClosedError(f"local provider for {self.model!r} is closed; create a new provider")
            if self._loaded is not None:
                return self._loaded

        with self._load_lock:
            with self._state_lock:
                if self._closed:
                    raise LocalProviderClosedError(
                        f"local provider for {self.model!r} is closed; create a new provider"
                    )
                if self._loaded is not None:
                    return self._loaded

            started = time.monotonic()
            self.logger.info(
                "Loading local model %r with backend %r (device=%s, dtype=%s, revision=%s)",
                self.model,
                self.backend_name,
                self.device,
                self.dtype,
                self.revision or "default",
            )
            try:
                loaded = self._load_backend(self._build_spec())
            except Exception:
                self.logger.warning(
                    "Failed to load local model %r with backend %r",
                    self.model,
                    self.backend_name,
                )
                raise
            elapsed = time.monotonic() - started

            with self._state_lock:
                closed = self._closed
                if not closed:
                    self._loaded = loaded
                    self._load_seconds = elapsed
            if closed:
                # close() raced with the load; release the model immediately.
                try:
                    self._unload_backend(loaded)
                except Exception:
                    self.logger.warning("Could not release local model %r after a close race", self.model)
                raise LocalProviderClosedError(f"local provider for {self.model!r} was closed while loading")

            self.logger.info(
                "Loaded local model %r in %.1fs (resolved=%s, revision=%s, device=%s, dtype=%s)",
                self.model,
                elapsed,
                loaded.resolved_model,
                loaded.revision or "none",
                loaded.device,
                loaded.dtype,
            )
            return loaded

    def _run_inference(self, fn: Any) -> Any:
        with self._inference_lock:
            with self._state_lock:
                if self._closed:
                    raise LocalProviderClosedError(
                        f"local provider for {self.model!r} is closed; create a new provider"
                    )
                loaded = self._loaded
            if loaded is None:
                raise LocalProviderClosedError(f"local provider for {self.model!r} was closed while a call was waiting")
            return fn(loaded)

    def _close_sync(self) -> None:
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
        with self._inference_lock:
            with self._state_lock:
                loaded, self._loaded = self._loaded, None
            if loaded is not None:
                try:
                    self._unload_backend(loaded)
                except Exception as exc:
                    self.logger.warning(
                        "Could not release local model %r cleanly: %s",
                        self.model,
                        exc,
                    )

    def _check_model(self, model: str) -> None:
        if model != self.model:
            raise LocalConfigurationError(
                f"binding model {model!r} does not match the local provider's model {self.model!r}; "
                "build the binding with the provider's model (local_embed/local_rerank do this automatically)"
            )

    # ── Unsupported provider surface ──────────────────────────

    async def chat(self, request: CLIverRequest) -> CLIverResponse:
        raise CapabilityNotSupported(f"local provider for {self.model!r} does not support chat")

    async def stream(self, request: CLIverRequest) -> AsyncIterator[CLIverMessageChunk]:
        raise CapabilityNotSupported(f"local provider for {self.model!r} does not support streaming")
        if False:  # pragma: no cover - makes this an async generator, matching the ABC
            yield None

    # ── Subclass hooks ────────────────────────────────────────

    def _load_backend(self, spec: LoadSpec) -> LoadedEmbedding | LoadedRerank:
        raise NotImplementedError

    def _unload_backend(self, loaded: LoadedEmbedding | LoadedRerank) -> None:
        raise NotImplementedError

    def _warmup(self, loaded: LoadedEmbedding | LoadedRerank) -> None:
        raise NotImplementedError


class LocalEmbeddingProvider(LocalProvider):
    """Embedding provider backed by one in-process model."""

    _backend_protocol = EmbeddingBackend
    _capability = "embedding"

    def _load_backend(self, spec: LoadSpec) -> LoadedEmbedding:
        return self.backend.load_embedding(spec)

    def _unload_backend(self, loaded: LoadedEmbedding) -> None:
        self.backend.unload_embedding(loaded)

    def _warmup(self, loaded: LoadedEmbedding) -> None:
        self._run_inference(lambda current: self.backend.embed(current, ["warmup"], batch_size=1))

    async def embed(self, texts: list[str], *, model: str, **options: Any) -> EmbeddingsResult:
        """Embed *texts* in input order.

        Vectors are raw by default; ``normalize=True`` L2-normalizes each
        vector.  ``dim`` optionally validates the model's output size.
        """
        self._check_model(model)
        parsed = _parse_embed_options(options)
        if not texts:
            return EmbeddingsResult(vectors=[], model=model)

        loaded = await asyncio.to_thread(self._load)
        expected_dim = parsed["dim"]
        reported_dim = loaded.dim
        if expected_dim is not None and reported_dim is not None and expected_dim != reported_dim:
            raise LocalDimensionError(
                f"model {self.model!r} produces {reported_dim}-d vectors, but dim={expected_dim} was requested. "
                "Remove dim or set it to the model's output size."
            )

        batch_size = parsed["batch_size"]
        raw = await asyncio.to_thread(
            self._run_inference,
            lambda current: self.backend.embed(current, list(texts), batch_size=batch_size),
        )
        vectors = _validate_vectors(
            raw,
            count=len(texts),
            expected_dim=expected_dim if expected_dim is not None else reported_dim,
            model=self.model,
        )
        if parsed["normalize"]:
            vectors = _normalize_vectors(vectors)
        if reported_dim is None and vectors:
            loaded.dim = len(vectors[0])
        return EmbeddingsResult(vectors=vectors, model=model)


class LocalRerankProvider(LocalProvider):
    """Rerank provider backed by one in-process cross-encoder model."""

    _backend_protocol = RerankBackend
    _capability = "rerank"

    def _load_backend(self, spec: LoadSpec) -> LoadedRerank:
        return self.backend.load_rerank(spec)

    def _unload_backend(self, loaded: LoadedRerank) -> None:
        self.backend.unload_rerank(loaded)

    def _warmup(self, loaded: LoadedRerank) -> None:
        self._run_inference(lambda current: self.backend.score(current, [("warmup", "warmup")], batch_size=1))

    async def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        model: str,
        top_k: int | None = None,
        **options: Any,
    ) -> list[RerankItem]:
        """Score *documents* against *query*, best first.

        Scores are raw model outputs by default; ``normalize=True`` applies a
        stable Sigmoid, yielding ``[0, 1]`` relevance scores.  ``top_k`` is
        applied after sorting; ``top_k=0`` returns an empty list without
        loading the model.
        """
        self._check_model(model)
        parsed = _parse_rerank_options(options)
        top_k = _as_optional_non_negative_int(top_k, "top_k")
        documents = list(documents)
        if not documents or top_k == 0:
            return []

        await asyncio.to_thread(self._load)
        pairs = [(query, document) for document in documents]
        batch_size = parsed["batch_size"]
        raw = await asyncio.to_thread(
            self._run_inference,
            lambda current: self.backend.score(current, pairs, batch_size=batch_size),
        )
        if len(raw) != len(documents):
            raise LocalInferenceError(
                f"local model {self.model!r} returned {len(raw)} scores for {len(documents)} documents"
            )

        scores: list[float] = []
        for index, value in enumerate(raw):
            try:
                score = float(value)
            except (TypeError, ValueError) as exc:
                raise LocalInferenceError(
                    f"local model {self.model!r} returned a non-numeric score at index {index}"
                ) from exc
            if not math.isfinite(score):
                raise LocalInferenceError(f"local model {self.model!r} returned a non-finite score at index {index}")
            scores.append(_sigmoid(score) if parsed["normalize"] else score)

        items = [
            RerankItem(index=index, relevance_score=score, document=documents[index])
            for index, score in enumerate(scores)
        ]
        # Stable sort: equal scores keep input order, so the result is
        # deterministic regardless of the backend's ordering.
        items.sort(key=lambda item: item.relevance_score, reverse=True)
        return items[:top_k] if top_k is not None else items
