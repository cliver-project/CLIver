"""CellExecutor — type-specific cell execution."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any, Dict

from cliver.notebook.ref_resolver import resolve_refs

if TYPE_CHECKING:
    from cliver.notebook.models import Cell

logger = logging.getLogger(__name__)


class CellExecutor:
    """Dispatches cell execution by type."""

    async def execute(self, cell: "Cell", runtime) -> Dict[str, Any]:
        handlers = {
            "config": self._execute_config,
            "llm": self._execute_llm,
            "code": self._execute_code,
            "display": self._execute_display,
        }
        handler = handlers.get(cell.type)
        if not handler:
            raise ValueError(f"Unknown cell type: '{cell.type}'")
        return await handler(cell, runtime)

    async def _execute_config(self, cell: "Cell", runtime) -> Dict[str, Any]:
        return dict(cell.outputs)

    async def _execute_llm(self, cell: "Cell", runtime) -> Dict[str, Any]:
        prompt = resolve_refs(cell.inputs.get("prompt", ""), runtime.variables)
        agent_name = cell.inputs.get("agent", "")
        if agent_name and "${" in agent_name:
            agent_name = resolve_refs(agent_name, runtime.variables)
        agent_name = agent_name or getattr(runtime.notebook, "default_agent", None)

        agent = runtime.agent_factory.create(agent_name or None)
        ctx = getattr(runtime.notebook, "context", {})
        working_dir = ctx.get("working_dir") if isinstance(ctx, dict) else None
        await agent.initialize({"working_dir": working_dir})

        result = await agent.run(prompt)

        outputs: Dict[str, Any] = {"text": result.text}
        if result.artifacts:
            outputs["artifacts"] = [
                {"path": a.path, "media_type": a.media_type, "size": a.size} for a in result.artifacts
            ]
        if cell.inputs.get("output_format") == "json":
            try:
                outputs["data"] = json.loads(result.text)
            except (json.JSONDecodeError, TypeError):
                pass

        return outputs

    async def _execute_code(self, cell: "Cell", runtime) -> Dict[str, Any]:
        source = cell.inputs.get("source", "")
        if not source.strip():
            raise ValueError("Code cell has no source code")

        namespace: dict = {}
        compiled = compile(source, f"<cell:{cell.id}>", "exec")
        exec(compiled, namespace)  # noqa: S102

        run_fn = namespace.get("run")
        if not run_fn or not callable(run_fn):
            raise ValueError("Code cell must define a callable run(ctx) function")

        timeout = cell.inputs.get("timeout_s", 300)
        result = await asyncio.wait_for(
            asyncio.to_thread(run_fn, runtime.ctx),
            timeout=timeout,
        )

        if not isinstance(result, dict):
            raise TypeError(f"run() must return dict, got {type(result).__name__}")

        try:
            json.dumps(result)
        except (TypeError, ValueError) as e:
            raise TypeError(f"run() returned non-JSON-serializable value: {e}") from e

        return result

    async def _execute_display(self, cell: "Cell", runtime) -> Dict[str, Any]:
        content = cell.inputs.get("content", "")
        if content and "${" in content:
            resolve_refs(content, runtime.variables)
        return {}
