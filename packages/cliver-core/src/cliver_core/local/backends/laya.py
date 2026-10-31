"""laya decision backend — an in-process System One model.

Loads the open-source `laya` checkpoints from Hugging Face and answers typed
questions locally.  The library import happens inside :meth:`load_decision`,
so importing this module — and ``cliver_core.local`` — stays torch-free.

Checkpoints follow laya: ``english``, ``multilingual`` (the default; 100+
languages including Chinese) and ``typed-decisions``, with laya's usual
aliases.  ``model="auto"`` lets laya route each request by detected language
instead of pinning one checkpoint.

Laya chooses its own precision, so ``dtype`` must be ``auto``.  Pinned
checkpoints are resolved through the shared local-model resolver first, which
keeps ``revision`` / ``cache_dir`` / ``local_files_only`` uniform with the
other local backends and avoids downloading the bundle's sibling
checkpoints.  ``model="auto"`` leaves downloading to laya itself, so it does
not accept ``cache_dir`` or ``local_files_only`` — use ``HF_HUB_CACHE`` /
``HF_HUB_OFFLINE`` there.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cliver_core.local.backends.base import LoadedDecision, LoadSpec
from cliver_core.local.backends.common import (
    empty_device_cache,
    resolve_device,
    wrap_load_error,
)
from cliver_core.local.errors import (
    LocalConfigurationError,
    LocalDependencyError,
    LocalInferenceError,
)
from cliver_core.local.resolution import resolve_local_model

__all__ = ["LayaBackend"]

_AUTO = "auto"
_CUSTOM = "custom"

# laya's standalone checkpoint repositories, one per checkpoint.  Resolving
# them here (rather than handing laya the bundle repo) downloads only the
# requested checkpoint and honours revision / cache_dir / local_files_only.
_CHECKPOINT_REPOS: dict[str, str] = {
    "english": "convaiinnovations/laya",
    "multilingual": "convaiinnovations/laya-multilingual",
    "typed-decisions": "convaiinnovations/laya-typed-decisions",
}

# laya's documented checkpoint aliases (names, never repository ids).
_ALIASES: dict[str, str] = {
    "en": "english",
    "laya": "english",
    "default": "english",
    "multi": "multilingual",
    "ml": "multilingual",
    "laya-multilingual": "multilingual",
    "typed": "typed-decisions",
    "typed_decisions": "typed-decisions",
    "laya-typed-decisions": "typed-decisions",
    "decisions": "typed-decisions",
}


@dataclass
class _RouterHandle:
    """The Router plus the checkpoint to pin, or ``None`` to route per request."""

    router: Any
    checkpoint: str | None


class LayaBackend:
    """System One predictions from a local laya checkpoint."""

    name = "laya"

    def load_decision(self, spec: LoadSpec) -> LoadedDecision:
        try:
            from laya import Router
        except ImportError as exc:
            raise LocalDependencyError(
                "local backend 'laya' requires the optional package 'laya'. Install it with: "
                'pip install "cliver-core[local-laya]" (or "cliver-core[local]" for all local backends).'
            ) from exc
        if spec.dtype.strip().lower() != "auto":
            raise LocalConfigurationError(
                "the laya backend chooses its own precision; use dtype='auto' "
                "(laya applies mixed precision for the resolved device)"
            )
        device = resolve_device(spec.device)
        model = spec.model.strip()
        lowered = model.lower()

        try:
            if lowered == _AUTO:
                if spec.cache_dir is not None or spec.local_files_only:
                    raise LocalConfigurationError(
                        "model='auto' lets laya route between checkpoints, so laya downloads them "
                        "itself and cache_dir= / local_files_only= do not apply. Pin a checkpoint "
                        "(english, multilingual, typed-decisions) to use them, or point "
                        "huggingface_hub at the cache with HF_HUB_CACHE / HF_HUB_OFFLINE."
                    )
                # No preload: laya preloads *every* registered checkpoint, which
                # in routing mode is the full bundle.  The first predict loads
                # the checkpoint the router picks.
                router = Router(device=device, revision=spec.revision)
                checkpoint: str | None = None
                resolved_model = _AUTO
                revision = spec.revision
            else:
                checkpoint = _ALIASES.get(lowered, lowered)
                source = _CHECKPOINT_REPOS.get(checkpoint, model)
                resolved_model, revision = resolve_local_model(
                    source,
                    revision=spec.revision,
                    cache_dir=spec.cache_dir,
                    local_files_only=spec.local_files_only,
                    logger=spec.logger,
                )
                if checkpoint not in _CHECKPOINT_REPOS:
                    checkpoint = _CUSTOM
                router = Router(
                    models={checkpoint: resolved_model},
                    default=checkpoint,
                    device=device,
                )
                # Preload exactly the pinned checkpoint.  Router(preload=True)
                # would also build the bundle's other two checkpoints.
                router.preload([checkpoint])
        except LocalConfigurationError:
            raise
        except Exception as exc:
            raise wrap_load_error(exc, model=spec.model, backend=self.name) from exc

        return LoadedDecision(
            handle=_RouterHandle(router=router, checkpoint=checkpoint),
            resolved_model=resolved_model,
            revision=revision,
            device=device,
            dtype="auto",
            max_length=spec.max_length,
        )

    def predict(
        self,
        loaded: LoadedDecision,
        state: Any,
        questions: dict[str, Any],
        *,
        max_length: int | None = None,
    ) -> dict[str, Any]:
        handle: _RouterHandle = loaded.handle
        kwargs: dict[str, Any] = {}
        if handle.checkpoint is not None:
            kwargs["model"] = handle.checkpoint
        if max_length is not None:
            kwargs["max_len"] = max_length
        try:
            return handle.router.predict(state, questions, **kwargs)
        except Exception as exc:
            raise LocalInferenceError(
                f"system one inference with backend {self.name!r} failed for model {loaded.resolved_model!r}: {exc}"
            ) from exc

    def unload_decision(self, loaded: LoadedDecision) -> None:
        handle: _RouterHandle = loaded.handle
        try:
            handle.router.unload()
        finally:
            loaded.handle = None
            empty_device_cache(loaded.device)
