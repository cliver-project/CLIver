"""Lab API routes for the gateway."""

from __future__ import annotations

import asyncio
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
    from cliver.session_manager import SessionManager

logger = logging.getLogger(__name__)


async def _drain_queue(websocket: WebSocket, queue: asyncio.Queue) -> None:
    """Send all queued events to the WebSocket before closing."""
    while not queue.empty():
        try:
            event = queue.get_nowait()
            await websocket.send_json(event)
        except Exception:
            break


def get_lab_routes(
    lab_store: "LabStore",
    runtime_manager: RuntimeManager,
    agent_factory: "AgentFactory",
    session_manager: "SessionManager",
    require_auth: Callable,
) -> list:
    """Return lab API routes."""

    from starlette.responses import StreamingResponse

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
    async def handle_cell_save_output(request: Request):
        lab_id = request.path_params["id"]
        cell_id = request.path_params["cell_id"]

        data = await request.json()
        outputs = data.get("outputs", {})

        lab_store.save_cell_output(lab_id, cell_id, outputs, "completed")
        return JSONResponse({"status": "ok"})

    @require_auth
    async def handle_cell_chat(request: Request):
        lab_id = request.path_params["id"]
        cell_id = request.path_params["cell_id"]

        data = await request.json()
        message = data.get("message", "").strip()
        if not message:
            return JSONResponse({"error": "message is required"}, status_code=400)

        agent_name = data.get("agent", "")
        system_prompt = data.get("system_prompt", "")
        output_format = data.get("output_format", "text")

        lab = lab_store.get(lab_id)
        if not lab:
            return JSONResponse({"error": "lab not found"}, status_code=404)

        cell = lab.get_cell(cell_id)
        if not cell or cell.type != "llm":
            return JSONResponse({"error": "cell not found or not an LLM cell"}, status_code=404)

        # Get or create chat session
        try:
            session_id = session_manager.get_or_create_cell_session(lab_id, cell_id)
            session_manager.append_turn(session_id, "user", message)
        except Exception as e:
            logger.warning("Failed to save user turn: %s", e)
            return JSONResponse({"error": "Failed to save message"}, status_code=500)

        # Resolve agent
        agent_name = agent_name or lab.default_agent
        if agent_name and "${" in str(agent_name):
            from cliver.lab.ref_resolver import resolve_refs
            try:
                runtime = runtime_manager.get_or_create(lab_id, lab, agent_factory)
                agent_name = resolve_refs(str(agent_name), runtime.variables)
            except (ValueError, KeyError) as e:
                return JSONResponse(
                    {"error": f"Reference not found in agent: {e}"}, status_code=400
                )

        agent = agent_factory.create(agent_name or None)
        ctx = lab.context if isinstance(lab.context, dict) else {}
        await agent.initialize({"working_dir": ctx.get("working_dir")})

        # Build prompt with optional system prompt
        prompt = message
        if system_prompt:
            prompt = f"System: {system_prompt}\n\nUser: {message}"

        from cliver.lab.chat import stream_chat_response

        async def generate():
            async for event in stream_chat_response(
                agent, prompt, session_manager, session_id, output_format,
            ):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    @require_auth
    async def handle_cell_chat_history(request: Request):
        lab_id = request.path_params["id"]
        cell_id = request.path_params["cell_id"]

        try:
            session_id = session_manager.get_or_create_cell_session(lab_id, cell_id)
            turns = session_manager.load_turns(session_id)
        except Exception as e:
            logger.warning("Failed to load chat history: %s", e)
            return JSONResponse({"error": str(e)}, status_code=500)

        return JSONResponse({
            "session_id": session_id,
            "turns": turns,
        })

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

            # Queue for streaming events to WebSocket (created early so error
            # events can be sent before agent execution starts)
            event_queue: asyncio.Queue = asyncio.Queue()

            runtime = runtime_manager.get_or_create(lab_id, lab, agent_factory)
            # Rebuild variables from the latest saved lab state (config cells may
            # have been auto-saved since the runtime was created).
            runtime.load_from_lab()

            from cliver.lab.ref_resolver import resolve_refs

            try:
                prompt = resolve_refs(cell.inputs.get("prompt", ""), runtime.variables)
            except (ValueError, KeyError) as e:
                msg = f"Reference not found in prompt: {e}"
                await event_queue.put({"type": "error", "message": msg})
                cell.status = "error"
                cell.error = msg
                lab_store.save_cell_output(lab_id, cell_id, {}, "error", error=msg)
                await _drain_queue(websocket, event_queue)
                return

            agent_name = cell.inputs.get("agent", "") or lab.default_agent
            if agent_name and "${" in str(agent_name):
                try:
                    agent_name = resolve_refs(str(agent_name), runtime.variables)
                except (ValueError, KeyError) as e:
                    msg = f"Reference not found in agent: {e}"
                    await event_queue.put({"type": "error", "message": msg})
                    cell.status = "error"
                    cell.error = msg
                    lab_store.save_cell_output(lab_id, cell_id, {}, "error", error=msg)
                    await _drain_queue(websocket, event_queue)
                    return

            agent = agent_factory.create(agent_name or None)
            ctx = lab.context if isinstance(lab.context, dict) else {}
            await agent.initialize({"working_dir": ctx.get("working_dir")})

            async def run_agent():
                """Run agent in background — saves results regardless of WebSocket state."""
                full_text = []
                await event_queue.put({"type": "status", "content": f"Starting {agent.name} agent…"})
                try:
                    async for chunk in agent.stream(prompt):
                        # Forward chunk type from the agent directly to WebSocket
                        if chunk.chunk_type in ("thinking", "tool", "tool_use", "tool_result", "status"):
                            await event_queue.put({"type": chunk.chunk_type, "content": chunk.text})
                        elif chunk.chunk_type == "text" and chunk.text:
                            full_text.append(chunk.text)
                            await event_queue.put({"type": "text", "content": chunk.text})
                        elif chunk.chunk_type == "done" and chunk.final_result:
                            result = chunk.final_result
                            is_error = result.status == "error"
                            outputs = {"text": result.text}
                            if result.artifacts:
                                outputs["artifacts"] = [
                                    {"path": a.path, "media_type": a.media_type, "size": a.size}
                                    for a in result.artifacts
                                ]
                            if cell.inputs.get("output_format") == "json" and not is_error:
                                try:
                                    outputs["data"] = json.loads(result.text)
                                except (json.JSONDecodeError, TypeError):
                                    pass

                            runtime.variables[cell_id] = {"outputs": outputs}
                            cell.outputs = outputs
                            cell.status = "error" if is_error else "completed"
                            cell.error = result.error if is_error else None
                            lab_store.save_cell_output(
                                lab_id, cell_id, outputs, cell.status,
                                duration_ms=result.duration_ms, error=result.error if is_error else None,
                            )
                            await event_queue.put({
                                "type": "error" if is_error else "done",
                                "outputs": outputs,
                                "status": cell.status,
                                "duration_ms": result.duration_ms,
                                **({"message": result.error} if is_error and result.error else {}),
                            })
                            return
                    # If stream ends without "done", treat remaining text as result
                    if full_text:
                        text = "".join(full_text)
                        outputs = {"text": text}
                        cell.outputs = outputs
                        cell.status = "completed"
                        lab_store.save_cell_output(lab_id, cell_id, outputs, "completed")
                        await event_queue.put({"type": "done", "outputs": outputs, "status": "completed"})
                    else:
                        await event_queue.put({"type": "status", "content": "Agent finished with no output"})
                except Exception as e:
                    logger.warning("LLM cell execution error: %s", e)
                    cell.status = "error"
                    cell.error = str(e)
                    lab_store.save_cell_output(lab_id, cell_id, {}, "error", error=str(e))
                    await event_queue.put({"type": "error", "message": str(e), "status": "error"})

            # Start agent execution in background — it always saves results
            agent_task = asyncio.create_task(run_agent())

            # Stream events to WebSocket (drops if client disconnects)
            try:
                while not agent_task.done() or not event_queue.empty():
                    try:
                        event = await asyncio.wait_for(event_queue.get(), timeout=0.5)
                        await websocket.send_json(event)
                    except asyncio.TimeoutError:
                        continue
                    except Exception:
                        break
                # Agent finished — drain any remaining events
                while not event_queue.empty():
                    try:
                        await websocket.send_json(event_queue.get_nowait())
                    except Exception:
                        break
            finally:
                pass

        except Exception as e:
            logger.warning("WebSocket handshake error: %s", e)
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
        Route("/admin/api/labs/{id}/cells/{cell_id}/save-output", handle_cell_save_output, methods=["POST"]),
        Route("/admin/api/labs/{id}/cells/{cell_id}/chat", handle_cell_chat, methods=["POST"]),
        Route("/admin/api/labs/{id}/cells/{cell_id}/chat/history", handle_cell_chat_history),
        Route("/admin/api/labs/{id}/run", handle_run_all, methods=["POST"]),
        Route("/admin/api/labs/{id}/cells/{cell_id}/available-refs", handle_available_refs),
        WebSocketRoute("/admin/ws/labs/{id}/cells/{cell_id}", ws_cell_execute),
    ]
