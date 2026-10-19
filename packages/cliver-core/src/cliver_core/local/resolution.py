"""Resolve a local model reference to a local directory.

Hugging Face ids are resolved with ``snapshot_download`` **before** the model
library sees them.  This gives every backend the same semantics for
``revision``, ``cache_dir``, and ``local_files_only`` — FlagEmbedding, for
example, does not forward those options to ``from_pretrained``.

The download path deliberately ignores alternate-runtime copies (ONNX,
OpenVINO, TensorFlow, ...).  Some repositories ship the same weights twice;
only the runtime files the local backends use are fetched.

The Hugging Face token is never accepted here: ``huggingface_hub`` reads
``HF_TOKEN`` (and ``HF_ENDPOINT``) from the environment itself, and nothing in
this module logs credentials.
"""

from __future__ import annotations

import fnmatch
import logging
import os
from pathlib import Path

from cliver_core.local.errors import LocalConfigurationError, LocalModelLoadError

__all__ = ["resolve_local_model"]

_IGNORE_PATTERNS: tuple[str, ...] = (
    "onnx/*",
    "openvino/*",
    "*.onnx",
    "*.onnx_data",
    "*.h5",
    "*.msgpack",
    "*.tflite",
    "*.ot",
)

_REMOTE_SCHEMES = ("http", "https", "ftp", "file", "hf", "s3", "gs")


def _is_remote_reference(model: str) -> bool:
    lowered = model.lower()
    return any(lowered.startswith(f"{scheme}://") for scheme in _REMOTE_SCHEMES)


def _is_explicit_path(model: str) -> bool:
    """True when *model* is clearly meant as a filesystem path.

    A bare ``org/name`` is a valid Hugging Face repo id, so relative paths must
    be written as ``./models/name`` (or an absolute path) to be recognized as
    paths when they do not exist.
    """
    return model.startswith((".", "~", "/", os.sep)) or (os.altsep is not None and model.startswith(os.altsep))


def _ignored(filename: str) -> bool:
    return any(fnmatch.fnmatch(filename, pattern) for pattern in _IGNORE_PATTERNS)


def _estimate_size(model: str, revision: str | None) -> int | None:
    """Best-effort download size in bytes, excluding ignored patterns."""
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(model, revision=revision, files_metadata=True)
        total = 0
        for sibling in info.siblings or []:
            filename = getattr(sibling, "rfilename", None)
            size = getattr(sibling, "size", None)
            if not filename or _ignored(filename) or not isinstance(size, int):
                continue
            total += size
        return total or None
    except Exception:
        return None


def _format_size(size: int | None) -> str:
    if not size:
        return "size unknown"
    gb = size / (1024**3)
    if gb >= 1:
        return f"about {gb:.1f} GB"
    return f"about {size / (1024**2):.0f} MB"


def _cache_description(cache_dir: str | None) -> str:
    return f"cache_dir={cache_dir}" if cache_dir else "the Hugging Face cache (cache_dir not set)"


def resolve_local_model(
    model: str,
    *,
    revision: str | None,
    cache_dir: str | None,
    local_files_only: bool,
    logger: logging.Logger,
) -> tuple[str, str | None]:
    """Return ``(local_directory, resolved_revision)`` for *model*.

    A local directory is returned as an absolute path with revision ``None``.
    A Hugging Face id is resolved to its snapshot directory; the resolved
    revision is the snapshot's commit sha (the directory name).
    """
    reference = model.strip()
    if not reference:
        raise LocalConfigurationError("model is required")
    if _is_remote_reference(reference):
        raise LocalConfigurationError(
            f"model {reference!r} is a remote URL; local backends load a local directory or "
            "a Hugging Face repository id, not a URL"
        )

    path = Path(reference).expanduser()
    if path.exists():
        if not path.is_dir():
            raise LocalConfigurationError(f"model path {str(path)!r} is not a directory")
        if revision is not None:
            raise LocalConfigurationError(
                "revision is only valid for Hugging Face repository ids, not local directories"
            )
        return str(path.resolve()), None

    if _is_explicit_path(reference):
        raise LocalModelLoadError(f"model directory {str(path)!r} does not exist")

    return _resolve_hub_model(
        reference,
        revision=revision,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        logger=logger,
    )


def _resolve_hub_model(
    model: str,
    *,
    revision: str | None,
    cache_dir: str | None,
    local_files_only: bool,
    logger: logging.Logger,
) -> tuple[str, str | None]:
    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import LocalEntryNotFoundError, RevisionNotFoundError
    except ImportError as exc:  # pragma: no cover - extras ship huggingface_hub
        raise LocalModelLoadError(
            "resolving a Hugging Face model requires 'huggingface-hub'; install a local extra"
        ) from exc

    kwargs = {
        "repo_id": model,
        "revision": revision,
        "cache_dir": cache_dir,
        "ignore_patterns": list(_IGNORE_PATTERNS),
    }

    try:
        resolved = snapshot_download(local_files_only=True, **kwargs)
    except LocalEntryNotFoundError as exc:
        if local_files_only:
            raise LocalModelLoadError(
                f"model {model!r} (revision {revision or 'main'}) is not available in "
                f"{_cache_description(cache_dir)} and local_files_only=True was requested. "
                "Pre-download the model or unset local_files_only."
            ) from exc
        size = _estimate_size(model, revision)
        logger.info(
            "Downloading local model %r (revision %s, %s) — the first run can take several minutes",
            model,
            revision or "main",
            _format_size(size),
        )
        try:
            resolved = snapshot_download(local_files_only=False, **kwargs)
        except RevisionNotFoundError as exc:
            raise LocalModelLoadError(
                f"revision {revision!r} was not found for model {model!r}. "
                "Use a branch, tag, or commit sha that exists in the repository."
            ) from exc
        except Exception as exc:
            raise LocalModelLoadError(
                f"could not download model {model!r} (revision {revision or 'main'}) into "
                f"{_cache_description(cache_dir)}: {exc}. Check network access, or pre-download "
                "the model and set local_files_only=True."
            ) from exc
    except RevisionNotFoundError as exc:
        raise LocalModelLoadError(
            f"revision {revision!r} was not found for model {model!r}. "
            "Use a branch, tag, or commit sha that exists in the repository."
        ) from exc
    except Exception as exc:
        raise LocalModelLoadError(
            f"could not resolve model {model!r} from {_cache_description(cache_dir)}: {exc}. "
            "Check network access, or pre-download the model and set local_files_only=True."
        ) from exc

    resolved_path = Path(resolved)
    resolved_revision = resolved_path.name or None
    logger.info(
        "Resolved local model %r to %s (revision %s)",
        model,
        str(resolved_path),
        resolved_revision or "unknown",
    )
    return str(resolved_path), resolved_revision
