"""Notebook runtime — execution context and variable scope."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

from cliver.notebook.ref_resolver import resolve_value

if TYPE_CHECKING:
    from cliver.agents.factory import AgentFactory


class RuntimeContext:
    """Context injected into code cells as the ctx parameter."""

    def __init__(
        self,
        variables: Dict[str, Any],
        agent_factory: Optional["AgentFactory"] = None,
        notebook_context: Optional[Dict[str, Any]] = None,
    ):
        self._variables = variables
        self._agent_factory = agent_factory
        self._notebook_context = notebook_context or {}
        self._logs: list[str] = []

    def refs(self, path: str) -> Any:
        """Access a cell output value by dot-path."""
        return resolve_value(path, self._variables)

    def log(self, msg: str) -> None:
        """Write to execution log."""
        self._logs.append(str(msg))

    @property
    def agent_factory(self) -> Optional["AgentFactory"]:
        return self._agent_factory

    @property
    def notebook_context(self) -> Dict[str, Any]:
        return self._notebook_context
