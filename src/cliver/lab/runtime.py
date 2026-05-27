"""Lab runtime — execution context and variable scope."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any, Dict, Optional

from cliver.lab.ref_resolver import resolve_value

if TYPE_CHECKING:
    from cliver.agents.factory import AgentFactory
    from cliver.lab.models import Lab

logger = logging.getLogger(__name__)


class RuntimeContext:
    """Context injected into code cells as the ctx parameter."""

    def __init__(
        self,
        variables: Dict[str, Any],
        agent_factory: Optional["AgentFactory"] = None,
        lab_context: Optional[Dict[str, Any]] = None,
    ):
        self._variables = variables
        self._agent_factory = agent_factory
        self._lab_context = lab_context or {}
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
    def lab_context(self) -> Dict[str, Any]:
        return self._lab_context


class LabRuntime:
    """Per-lab execution context with variable scope."""

    def __init__(self, lab: "Lab", agent_factory: "AgentFactory"):
        self.lab = lab
        self.lab_id = lab.id
        self.agent_factory = agent_factory
        self.variables: Dict[str, Any] = {}
        self.ctx = RuntimeContext(
            self.variables,
            agent_factory,
            lab.context if isinstance(lab.context, dict) else {},
        )
        from cliver.lab.executor import CellExecutor

        self._executor = CellExecutor()
        self._lock = asyncio.Lock()
        self.last_active = time.monotonic()

    def load_from_lab(self) -> None:
        """Rebuild variables from all completed cells' outputs."""
        self.variables.clear()
        for cell in self.lab.cells:
            if cell.status == "completed" and cell.outputs:
                self.variables[cell.id] = {"outputs": dict(cell.outputs)}

    async def execute_cell(self, cell_id: str) -> Dict[str, Any]:
        """Execute one cell. Updates variables, cell status, and outputs."""
        async with self._lock:
            self.last_active = time.monotonic()
            cell = self.lab.get_cell(cell_id)
            if not cell:
                raise ValueError(f"Cell '{cell_id}' not found in lab")

            cell.status = "running"
            cell.error = None
            start = time.monotonic()

            try:
                outputs = await self._executor.execute(cell, self)
                duration = int((time.monotonic() - start) * 1000)

                cell.outputs = outputs
                cell.status = "completed"
                cell.duration_ms = duration
                cell.error = None
                self.variables[cell.id] = {"outputs": dict(outputs)}

                return outputs
            except Exception as e:
                duration = int((time.monotonic() - start) * 1000)
                cell.status = "error"
                cell.error = str(e)
                cell.duration_ms = duration
                logger.warning("Cell '%s' failed: %s", cell_id, e)
                raise

    async def execute_all(self) -> None:
        """Execute all cells sequentially, stopping on error."""
        for cell in self.lab.cells:
            await self.execute_cell(cell.id)

    def get_available_refs(self, before_cell_id: str) -> list:
        """Return available references for cells before the given cell."""
        result = []
        for cell in self.lab.cells_before(before_cell_id):
            if cell.status != "completed" or not cell.outputs:
                continue
            fields = []
            for key, value in cell.outputs.items():
                path = f"{cell.id}.outputs.{key}"
                if isinstance(value, list):
                    preview = f"[Array({len(value)})]"
                    vtype = "array"
                elif isinstance(value, dict):
                    preview = f"{{Object({len(value)})}}"
                    vtype = "object"
                elif isinstance(value, str):
                    preview = value[:80] + "..." if len(value) > 80 else value
                    vtype = "string"
                else:
                    preview = str(value)
                    vtype = type(value).__name__
                fields.append({"path": path, "preview": preview, "type": vtype})
            result.append(
                {
                    "cell_id": cell.id,
                    "cell_title": cell.title,
                    "fields": fields,
                }
            )
        return result


class RuntimeManager:
    """Manages LabRuntime instances with idle timeout."""

    def __init__(self, timeout_s: int = 1800):
        self._runtimes: Dict[str, LabRuntime] = {}
        self._timeout_s = timeout_s

    def get_or_create(self, lab_id: str, lab: "Lab", agent_factory: "AgentFactory") -> LabRuntime:
        if lab_id in self._runtimes:
            rt = self._runtimes[lab_id]
            rt.last_active = time.monotonic()
            return rt

        rt = LabRuntime(lab, agent_factory)
        rt.load_from_lab()
        self._runtimes[lab_id] = rt
        return rt

    def remove(self, lab_id: str) -> None:
        self._runtimes.pop(lab_id, None)

    async def cleanup_idle(self) -> None:
        now = time.monotonic()
        expired = [nid for nid, rt in self._runtimes.items() if now - rt.last_active > self._timeout_s]
        for nid in expired:
            logger.info("Removing idle runtime for lab %s", nid)
            self._runtimes.pop(nid, None)

    async def shutdown_all(self) -> None:
        self._runtimes.clear()
