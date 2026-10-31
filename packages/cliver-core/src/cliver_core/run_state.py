"""Per-run tool state.

``AgentCore`` creates one :class:`ToolRunState` per ``chat()`` / ``stream()``
run and makes it ambient while that run's tools execute.  A tool that needs to
carry state across Re-Act iterations reads it with
:func:`current_tool_run_state`; the state is never stored on the agent, so
concurrent runs stay isolated.

``asyncio.to_thread`` copies the current context, so a tool body running in a
worker thread sees the same state object.  Mutate that object (or the
containers it holds) in place — assigning a new value to the context variable
inside the worker thread would not propagate back to the run.
"""

from __future__ import annotations

import contextlib
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Iterator

__all__ = ["ToolRunState", "current_tool_run_state", "tool_run_state_scope"]


@dataclass
class ToolRunState:
    """Mutable state shared by the tools of one AgentCore run."""

    data: dict[str, Any] = field(default_factory=dict)


_run_state_var: ContextVar[ToolRunState | None] = ContextVar("cliver_tool_run_state", default=None)


def current_tool_run_state() -> ToolRunState | None:
    """Return the state of the currently executing run, if any."""
    return _run_state_var.get()


@contextlib.contextmanager
def tool_run_state_scope(state: ToolRunState | None) -> Iterator[None]:
    """Make ``state`` ambient for the duration of the block."""
    if state is None:
        yield
        return
    token = _run_state_var.set(state)
    try:
        yield
    finally:
        _run_state_var.reset(token)
