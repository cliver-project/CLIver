"""Run correlation ids shared by ``AgentCore`` and ``SkillPipeline``."""

from __future__ import annotations

import uuid


def resolve_run_id(run_id: str | None) -> str:
    """Return *run_id* unchanged, or generate a fresh 32-hex id.

    A caller-supplied id must be a non-empty string; an empty, whitespace-only,
    or non-string value raises ``ValueError``.
    """
    if run_id is None:
        return uuid.uuid4().hex
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError(f"run_id must be a non-empty string, got {run_id!r}")
    return run_id
