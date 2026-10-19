"""Errors raised by in-process local model providers.

The hierarchy is intentionally small and dependency-free.  Backends classify
library-specific failures (missing optional package, bad revision, out-of-
memory, device unavailable) into these types so callers never need to catch
``ImportError``, ``huggingface_hub`` errors, or ``torch`` errors directly.

Every message names the model/backend and a concrete remediation.  Messages
never include input texts, vectors, or credentials.
"""

from __future__ import annotations

__all__ = [
    "LocalBackendNotSupportedError",
    "LocalConfigurationError",
    "LocalDependencyError",
    "LocalDeviceError",
    "LocalDimensionError",
    "LocalInferenceError",
    "LocalModelError",
    "LocalModelLoadError",
    "LocalProviderClosedError",
]


class LocalModelError(RuntimeError):
    """Base class for every local-model failure.

    Subclasses ``RuntimeError`` so applications that already treat provider
    failures as ``RuntimeError`` keep working.
    """


class LocalDependencyError(LocalModelError):
    """An optional local backend package is not installed."""


class LocalModelLoadError(LocalModelError):
    """The model could not be resolved, downloaded, or loaded."""


class LocalConfigurationError(LocalModelError):
    """An option is unknown, invalid, or set at the wrong layer."""


class LocalDeviceError(LocalModelError):
    """The requested device is unavailable, or the device ran out of memory."""


class LocalBackendNotSupportedError(LocalModelError):
    """A backend does not implement the requested capability."""


class LocalDimensionError(LocalModelError):
    """Embedding vectors do not match the expected dimension."""


class LocalInferenceError(LocalModelError):
    """The backend returned an unusable result or failed during inference."""


class LocalProviderClosedError(LocalModelError):
    """The provider was closed; create a new provider to run more calls."""
