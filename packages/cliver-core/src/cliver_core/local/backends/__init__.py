"""Backend registry.

Three built-in backend names are resolved lazily:

* ``sentence-transformers`` — generic Hugging Face embedding models via
  ``sentence_transformers.SentenceTransformer`` (embedding only);
* ``flag-embedding`` — FlagEmbedding's ``BGEM3FlagModel`` (embedding) and
  ``FlagReranker`` (rerank);
* ``laya`` — System One decision models (``english`` / ``multilingual`` /
  ``typed-decisions`` checkpoints, or any local fine-tune).

Nothing here imports the backing libraries.  ``resolve_backend`` imports the
module named in :data:`_BUILTIN` only when the name is requested, and those
modules import their heavy dependencies inside ``load_*``.

Applications can register additional backends by name.  A backend object can
also be passed directly to a provider, which is the simplest route for tests
and one-off integrations.
"""

from __future__ import annotations

import importlib
from typing import Any

from cliver_core.local.backends.base import (
    DecisionBackend,
    EmbeddingBackend,
    LoadedDecision,
    LoadedEmbedding,
    LoadedRerank,
    LoadSpec,
    RerankBackend,
)
from cliver_core.local.errors import LocalBackendNotSupportedError, LocalConfigurationError

__all__ = [
    "DecisionBackend",
    "EmbeddingBackend",
    "LoadSpec",
    "LoadedDecision",
    "LoadedEmbedding",
    "LoadedRerank",
    "RerankBackend",
    "register_backend",
    "resolve_backend",
]

_BUILTIN: dict[str, tuple[str, str]] = {
    "sentence-transformers": ("cliver_core.local.backends.sentence_transformers", "SentenceTransformerBackend"),
    "flag-embedding": ("cliver_core.local.backends.flag_embedding", "FlagEmbeddingBackend"),
    "laya": ("cliver_core.local.backends.laya", "LayaBackend"),
}

_REGISTERED: dict[str, Any] = {}


def _normalize_name(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise LocalConfigurationError("backend must be a non-empty string")
    return name.strip().lower()


def register_backend(name: str, backend: Any) -> None:
    """Register a backend instance under *name*.

    The backend is returned by :func:`resolve_backend` for that exact name.
    Re-registering a built-in or already registered name raises, so a typo
    cannot silently shadow the built-in behavior.
    """
    key = _normalize_name(name)
    if key in _BUILTIN or key in _REGISTERED:
        raise LocalConfigurationError(f"backend {key!r} is already registered")
    if not isinstance(getattr(backend, "name", None), str) or not backend.name.strip():
        raise LocalConfigurationError("a backend must expose a non-empty string 'name' attribute")
    _REGISTERED[key] = backend


def resolve_backend(name: str) -> Any:
    """Return the backend registered under *name* (built-ins load lazily)."""
    key = _normalize_name(name)
    if key in _REGISTERED:
        return _REGISTERED[key]
    target = _BUILTIN.get(key)
    if target is None:
        available = sorted([*_BUILTIN, *_REGISTERED])
        raise LocalBackendNotSupportedError(
            f"unknown local backend {name!r}; available backends: {', '.join(available)}. "
            "Pass a backend object to use a custom implementation."
        )
    module_name, class_name = target
    module = importlib.import_module(module_name)
    return getattr(module, class_name)()
