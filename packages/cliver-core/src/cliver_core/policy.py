"""Tool policy: the permission and execution-sandbox contract for AgentCore.

Two axes that multiply.  The **permission axis** decides whether a tool may
run at all — a category per tool (safe runs directly, ask goes through the
application-supplied callback) — and the **environment axis** decides what a
running tool may touch: subprocess env construction, a pinned working
directory, and read/write path scopes.

Deliberately opt-in: with no policy configured, behavior is exactly today's
(everything allowed, env inherited).  Security is supplied explicitly —
"explicit, not smart".  The application supplies the callback (CLI prompts,
gateway task config); this module only defines the shapes and the ambient
per-call context.
"""

from __future__ import annotations

import contextlib
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Awaitable, Callable, Literal

__all__ = [
    "Decision",
    "DecisionAction",
    "EnvSpec",
    "ExecutionPolicy",
    "PermissionCallback",
    "SAFE_ENV_VARS",
    "ToolCallContext",
    "ToolCategory",
    "current_policy",
    "policy_scope",
    "scope_violation",
]


# ── Permission axis ───────────────────────────────────────────


class DecisionAction(str, Enum):
    ALLOW = "allow"
    DENY = "deny"


@dataclass
class Decision:
    """A permission verdict; ``reason`` reaches the model on deny."""

    action: DecisionAction
    reason: str | None = None

    @classmethod
    def allow(cls) -> "Decision":
        return cls(action=DecisionAction.ALLOW)

    @classmethod
    def deny(cls, reason: str) -> "Decision":
        return cls(action=DecisionAction.DENY, reason=reason)


class ToolCategory(str, Enum):
    SAFE = "safe"  # runs without consulting the callback
    ASK = "ask"  # the callback decides (absent callback → allow, v1 compat)


@dataclass
class ToolCallContext:
    """What the permission callback sees of one tool call."""

    tool_name: str
    args: dict
    server: str | None  # MCP server name for mcp__ tools, else None
    category: ToolCategory


PermissionCallback = Callable[[ToolCallContext], Awaitable[Decision]]


# ── Environment axis ──────────────────────────────────────────


@dataclass
class EnvSpec:
    """How a subprocess environment is built — exact names, never patterns."""

    mode: Literal["inherit", "allowlist", "denylist"] = "inherit"
    names: set[str] = field(default_factory=set)

    def build(self, source: dict[str, str]) -> dict[str, str]:
        if self.mode == "allowlist":
            return {k: v for k, v in source.items() if k in self.names}
        if self.mode == "denylist":
            return {k: v for k, v in source.items() if k not in self.names}
        return dict(source)


# Convenience starter set for applications building an allowlist; a constant,
# not implicit behavior.
SAFE_ENV_VARS = frozenset(
    {
        "PATH",
        "HOME",
        "TMPDIR",
        "TEMP",
        "TMP",
        "LANG",
        "LC_ALL",
        "TZ",
        "SHELL",
        "USER",
        "TERM",
        "PWD",
    }
)


@dataclass
class ExecutionPolicy:
    """The environment a tool runs in when a policy is in effect."""

    env: EnvSpec = field(default_factory=EnvSpec)
    cwd: Path | None = None  # pinned working directory
    read_paths: list[Path] | None = None  # None = unrestricted
    write_paths: list[Path] | None = None  # None = unrestricted


# ── Ambient per-call context ──────────────────────────────────

_policy_var: ContextVar[ExecutionPolicy | None] = ContextVar("cliver_execution_policy", default=None)


def current_policy() -> ExecutionPolicy | None:
    """The execution policy ambient for this call, or None (today's behavior)."""
    return _policy_var.get()


@contextlib.contextmanager
def policy_scope(policy: ExecutionPolicy | None):
    """Set the ambient policy for a block; None is a no-op.

    ``AgentCore`` wraps each ``chat()``/``stream()`` loop in this scope; tools
    spawned through ``asyncio.to_thread`` inherit the context (tasks copy it,
    so concurrent chats on one agent stay isolated).
    """
    if policy is None:
        yield
        return
    token = _policy_var.set(policy)
    try:
        yield
    finally:
        _policy_var.reset(token)


def scope_violation(path: str | Path, *, write: bool) -> str | None:
    """Check *path* against the ambient policy's read/write scopes.

    Returns None when no policy is ambient, the matching scope list is None
    (unrestricted), or the resolved path stays inside a listed root; otherwise
    an error message naming the path and the scope.  Relative paths resolve
    against ``policy.cwd or Path.cwd()``; symlinks are followed before the
    comparison.
    """
    policy = current_policy()
    if policy is None:
        return None
    scopes = policy.write_paths if write else policy.read_paths
    if scopes is None:
        return None
    base = policy.cwd or Path.cwd()
    resolved = (base / path).resolve()
    if any(resolved.is_relative_to(root.resolve()) for root in scopes):
        return None
    kind = "write" if write else "read"
    roots = ", ".join(str(root) for root in scopes)
    return f"{resolved} is outside the allowed {kind} paths ({roots})"
