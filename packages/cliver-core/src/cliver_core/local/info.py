"""Metadata reported by a local model provider."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["LocalModelInfo"]


@dataclass(frozen=True)
class LocalModelInfo:
    """Snapshot of a local provider's configuration and load state.

    ``resolved_model``, ``revision``, ``device``, ``dtype``, and ``dim`` are
    ``None`` until the model has been loaded.  ``revision`` is the resolved
    commit sha for Hugging Face models and ``None`` for local directories.
    """

    model: str
    backend: str
    loaded: bool
    closed: bool
    revision: str | None = None
    resolved_model: str | None = None
    device: str | None = None
    dtype: str | None = None
    dim: int | None = None
    max_length: int | None = None
    load_seconds: float | None = None
