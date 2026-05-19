"""Lab API routes for the gateway."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Callable

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket

if TYPE_CHECKING:
    from cliver.agents.factory import AgentFactory
    from cliver.lab.runtime import RuntimeManager
    from cliver.lab.store import LabStore

logger = logging.getLogger(__name__)


def get_lab_routes(
    lab_store: "LabStore",
    runtime_manager: "RuntimeManager",
    agent_factory: "AgentFactory",
    require_auth: Callable,
) -> list:
    """Return lab API routes."""

    @require_auth
    async def handle_labs_list(request: Request):
        summaries = lab_store.list_all()
        return JSONResponse([s.model_dump() for s in summaries])

    @require_auth
    async def handle_labs_create(request: Request):
        data = await request.json()
        title = data.get("title", "").strip()
        if not title:
            return JSONResponse({"error": "title is required"}, status_code=400)
        lab = lab_store.create(
            title=title,
            description=data.get("description", ""),
            scenario_id=data.get("scenario_id"),
            default_agent=data.get("default_agent"),
            context=data.get("context"),
            cells=data.get("cells"),
        )
        return JSONResponse(lab.model_dump())

    @require_auth
    async def handle_lab_get(request: Request):
        lab_id = request.path_params["id"]
        lab = lab_store.get(lab_id)
        if not lab:
            return JSONResponse({"error": "not found"}, status_code=404)
        return JSONResponse(lab.model_dump())

    @require_auth
    async def handle_lab_update(request: Request):
        lab_id = request.path_params["id"]
        lab = lab_store.get(lab_id)
        if not lab:
            return JSONResponse({"error": "not found"}, status_code=404)
        data = await request.json()
        if "title" in data:
            lab.title = data["title"]
        if "description" in data:
            lab.description = data["description"]
        if "default_agent" in data:
            lab.default_agent = data["default_agent"]
        if "context" in data:
            lab.context = data["context"]
        if "cells" in data:
            from cliver.lab.models import Cell

            lab.cells = [Cell(**c) if isinstance(c, dict) else c for c in data["cells"]]
        lab_store.update(lab)
        return JSONResponse({"status": "ok"})

    @require_auth
    async def handle_lab_delete(request: Request):
        lab_id = request.path_params["id"]
        if lab_store.delete(lab_id):
            runtime_manager.remove(lab_id)
            return JSONResponse({"status": "ok"})
        return JSONResponse({"error": "not found"}, status_code=404)

    @require_auth
    async def handle_cell_execute(request: Request):
        lab_id = request.path_params["id"]
        cell_id = request.path_params["cell_id"]
        lab = lab_store.get(lab_id)
        if not lab:
            return JSONResponse({"error": "lab not found"}, status_code=404)
        cell = lab.get_cell(cell_id)
        if not cell:
            return JSONResponse({"error": "cell not found"}, status_code=404)

        runtime = runtime_manager.get_or_create(lab_id, lab, agent_factory)
        try:
            outputs = await runtime.execute_cell(cell_id)
            lab_store.save_cell_output(
                lab_id,
                cell_id,
                outputs,
                "completed",
                duration_ms=lab.get_cell(cell_id).duration_ms,
            )
            return JSONResponse(
                {
                    "cell_id": cell_id,
                    "status": "completed",
                    "outputs": outputs,
                }
            )
        except Exception as e:
            lab_store.save_cell_output(
                lab_id,
                cell_id,
                {},
                "error",
                error=str(e),
            )
            return JSONResponse(
                {
                    "cell_id": cell_id,
                    "status": "error",
                    "error": str(e),
                },
                status_code=500,
            )

    @require_auth
    async def handle_run_all(request: Request):
        lab_id = request.path_params["id"]
        lab = lab_store.get(lab_id)
        if not lab:
            return JSONResponse({"error": "lab not found"}, status_code=404)

        runtime = runtime_manager.get_or_create(lab_id, lab, agent_factory)
        try:
            await runtime.execute_all()
            for cell in lab.cells:
                lab_store.save_cell_output(
                    lab_id,
                    cell.id,
                    cell.outputs,
                    cell.status,
                    error=cell.error,
                    duration_ms=cell.duration_ms,
                )
            return JSONResponse({"status": "completed"})
        except Exception as e:
            return JSONResponse({"status": "error", "error": str(e)}, status_code=500)

    @require_auth
    async def handle_available_refs(request: Request):
        lab_id = request.path_params["id"]
        cell_id = request.path_params["cell_id"]
        lab = lab_store.get(lab_id)
        if not lab:
            return JSONResponse({"error": "lab not found"}, status_code=404)

        runtime = runtime_manager.get_or_create(lab_id, lab, agent_factory)
        refs = runtime.get_available_refs(cell_id)
        return JSONResponse(refs)

    async def ws_cell_execute(websocket: WebSocket):
        lab_id = websocket.path_params["id"]
        cell_id = websocket.path_params["cell_id"]
        await websocket.accept()

        try:
            data = await websocket.receive_json()
            if data.get("action") != "execute":
                await websocket.send_json({"type": "error", "message": "Unknown action"})
                await websocket.close()
                return

            lab = lab_store.get(lab_id)
            if not lab:
                await websocket.send_json({"type": "error", "message": "Lab not found"})
                await websocket.close()
                return

            cell = lab.get_cell(cell_id)
            if not cell or cell.type != "llm":
                await websocket.send_json({"type": "error", "message": "Cell not found or not an LLM cell"})
                await websocket.close()
                return

            await websocket.send_json({"type": "status", "status": "running"})

            runtime = runtime_manager.get_or_create(lab_id, lab, agent_factory)

            from cliver.lab.ref_resolver import resolve_refs

            prompt = resolve_refs(cell.inputs.get("prompt", ""), runtime.variables)
            agent_name = cell.inputs.get("agent", "") or lab.default_agent
            if agent_name and "${" in str(agent_name):
                agent_name = resolve_refs(str(agent_name), runtime.variables)

            agent = agent_factory.create(agent_name or None)
            ctx = lab.context if isinstance(lab.context, dict) else {}
            await agent.initialize({"working_dir": ctx.get("working_dir")})

            full_text = []
            async for chunk in agent.stream(prompt):
                if chunk.chunk_type == "text" and chunk.text:
                    full_text.append(chunk.text)
                    await websocket.send_json({"type": "text", "content": chunk.text})
                elif chunk.chunk_type == "thinking" and chunk.text:
                    await websocket.send_json({"type": "thinking", "content": chunk.text})
                elif chunk.chunk_type in ("tool_use", "tool_result") and chunk.text:
                    await websocket.send_json({"type": "tool", "content": chunk.text})
                elif chunk.chunk_type == "done" and chunk.final_result:
                    result = chunk.final_result
                    outputs = {"text": result.text}
                    if result.artifacts:
                        outputs["artifacts"] = [
                            {"path": a.path, "media_type": a.media_type, "size": a.size} for a in result.artifacts
                        ]
                    if cell.inputs.get("output_format") == "json":
                        try:
                            outputs["data"] = json.loads(result.text)
                        except (json.JSONDecodeError, TypeError):
                            pass

                    runtime.variables[cell_id] = {"outputs": outputs}
                    cell.outputs = outputs
                    cell.status = "completed"
                    lab_store.save_cell_output(
                        lab_id,
                        cell_id,
                        outputs,
                        "completed",
                        duration_ms=result.duration_ms,
                    )
                    await websocket.send_json(
                        {
                            "type": "done",
                            "outputs": outputs,
                            "status": "completed",
                            "duration_ms": result.duration_ms,
                        }
                    )

        except Exception as e:
            logger.warning("WebSocket cell execution error: %s", e)
            try:
                await websocket.send_json({"type": "error", "message": str(e), "status": "error"})
            except Exception:
                pass
        finally:
            try:
                await websocket.close()
            except Exception:
                pass

    return [
        Route("/admin/api/labs", handle_labs_list),
        Route("/admin/api/labs", handle_labs_create, methods=["POST"]),
        Route("/admin/api/labs/{id}", handle_lab_get),
        Route("/admin/api/labs/{id}", handle_lab_update, methods=["PUT"]),
        Route("/admin/api/labs/{id}", handle_lab_delete, methods=["DELETE"]),
        Route("/admin/api/labs/{id}/cells/{cell_id}/execute", handle_cell_execute, methods=["POST"]),
        Route("/admin/api/labs/{id}/run", handle_run_all, methods=["POST"]),
        Route("/admin/api/labs/{id}/cells/{cell_id}/available-refs", handle_available_refs),
        WebSocketRoute("/admin/ws/labs/{id}/cells/{cell_id}", ws_cell_execute),
    ]
