"""Shared backend helpers.

Every ``torch`` import lives inside a function: importing this module (and the
backend modules that use it) must stay light.
"""

from __future__ import annotations

from typing import Any

from cliver_core.local.errors import (
    LocalConfigurationError,
    LocalDeviceError,
    LocalModelError,
    LocalModelLoadError,
)

__all__ = [
    "empty_device_cache",
    "resolve_device",
    "resolve_dtype",
    "sentence_transformers_dtype_kwarg",
    "torch_dtype",
    "wrap_load_error",
]


def resolve_device(requested: str) -> str:
    """Resolve ``auto`` and validate an explicit device against the runtime.

    ``auto`` prefers CUDA and otherwise uses CPU.  MPS is never selected
    automatically: CPU inference is the supported default on Apple Silicon,
    and MPS remains an explicit, per-model choice.
    """
    import torch

    device = requested.strip().lower()
    if device == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        return "cpu"
    if device == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise LocalDeviceError(
                "device 'mps' was requested but is not available in this torch build; use device='cpu' or device='auto'"
            )
        return "mps"
    if device == "cuda":
        device = "cuda:0"
    if device.startswith("cuda:"):
        _, _, index = device.partition(":")
        try:
            number = int(index)
        except ValueError as exc:
            raise LocalConfigurationError(f"invalid CUDA device {requested!r}; use 'cuda' or 'cuda:N'") from exc
        if not torch.cuda.is_available() or number < 0 or number >= torch.cuda.device_count():
            raise LocalDeviceError(
                f"device {requested!r} is not available; CUDA available={torch.cuda.is_available()}, "
                f"device count={torch.cuda.device_count() if torch.cuda.is_available() else 0}. "
                "Use device='cpu' or device='auto'."
            )
        return device
    raise LocalConfigurationError(f"unsupported device {requested!r}; use one of: auto, cpu, mps, cuda, cuda:N")


def resolve_dtype(requested: str, device: str, *, allow_bfloat16: bool = True) -> str:
    """Resolve ``auto`` and reject dtype/device combinations the backend cannot honor."""
    dtype = requested.strip().lower()
    if dtype == "auto":
        return "float32" if device == "cpu" else "float16"
    if dtype not in ("float32", "float16", "bfloat16"):
        raise LocalConfigurationError(f"unsupported dtype {requested!r}; use one of: auto, float32, float16, bfloat16")
    if dtype == "float16" and device == "cpu":
        raise LocalConfigurationError(
            "dtype 'float16' is not supported on device 'cpu' for local inference; use dtype='float32' or dtype='auto'"
        )
    if dtype == "bfloat16" and not allow_bfloat16:
        raise LocalConfigurationError(
            "dtype 'bfloat16' is not supported by this backend; use dtype='float32' or dtype='auto'"
        )
    return dtype


def torch_dtype(name: str) -> Any:
    """Return the ``torch`` dtype object for a resolved dtype name."""
    import torch

    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def sentence_transformers_dtype_kwarg() -> str:
    """Return the dtype keyword transformers expects at the installed major version.

    transformers 5 renamed ``torch_dtype`` to ``dtype`` but keeps a deprecated
    alias.  Passing the current name avoids a warning on newer releases while
    still loading correctly on the 4.x line.
    """
    try:
        import transformers

        major = int(transformers.__version__.split(".", 1)[0])
    except Exception:
        major = 4
    return "dtype" if major >= 5 else "torch_dtype"


def empty_device_cache(device: str) -> None:
    """Best-effort release of cached device memory after unloading a model."""
    try:
        import torch

        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif device == "mps":
            mps = getattr(torch, "mps", None)
            if mps is not None and hasattr(mps, "empty_cache"):
                mps.empty_cache()
    except Exception:
        # Unloading is best-effort; a failed cache flush must not mask close().
        pass


def wrap_load_error(exc: BaseException, *, model: str, backend: str) -> LocalModelError:
    """Classify a load failure into a local error with an actionable message."""
    if isinstance(exc, LocalModelError):
        return exc

    text = str(exc)
    lowered = text.lower()
    try:
        import torch

        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return LocalDeviceError(
                f"loading {model!r} with backend {backend!r} ran out of device memory. "
                "Use device='cpu', reduce max_length, or close other local models first."
            )
    except Exception:
        pass

    if "out of memory" in lowered:
        return LocalDeviceError(
            f"loading {model!r} with backend {backend!r} ran out of memory. "
            "Use device='cpu', reduce max_length, or close other local models first."
        )
    if "mps" in lowered and ("not available" in lowered or "not supported" in lowered):
        return LocalDeviceError(
            f"backend {backend!r} could not use MPS for {model!r}: {text}. Use device='cpu' or device='auto'."
        )
    return LocalModelLoadError(f"could not load model {model!r} with backend {backend!r}: {text}")
