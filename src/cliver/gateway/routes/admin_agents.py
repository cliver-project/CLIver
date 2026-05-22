"""Agent API routes for the admin portal."""

from __future__ import annotations

import logging
from typing import Callable

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

logger = logging.getLogger(__name__)


def get_agent_routes(agent_store, require_auth: Callable) -> list:
    """Return agent CRUD API routes."""
    from cliver.gateway.admin import _run_in_thread

    @require_auth
    async def handle_list_agents(request: Request):
        agents = await _run_in_thread(agent_store.list_agents)
        return JSONResponse([a.model_dump() for a in agents])

    @require_auth
    async def handle_create_agent(request: Request):
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "Invalid JSON"}, status_code=400)
        name = (body.get("name") or "").strip()
        if not name:
            return JSONResponse({"error": "name is required"}, status_code=400)
        agent = await _run_in_thread(
            agent_store.create_agent,
            name=name,
            type=body.get("type", "cliver"),
            description=body.get("description"),
            role=body.get("role"),
            model=body.get("model"),
            is_default=body.get("is_default", 0),
        )
        return JSONResponse(agent.model_dump())

    @require_auth
    async def handle_get_agent(request: Request):
        agent_id = request.path_params["agent_id"]
        agent = await _run_in_thread(agent_store.get_agent, agent_id)
        if agent is None:
            return JSONResponse({"error": "Agent not found"}, status_code=404)
        return JSONResponse(agent.model_dump())

    @require_auth
    async def handle_update_agent(request: Request):
        agent_id = request.path_params["agent_id"]
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "Invalid JSON"}, status_code=400)
        agent = await _run_in_thread(
            agent_store.update_agent,
            agent_id,
            name=body.get("name"),
            type=body.get("type"),
            description=body.get("description"),
            role=body.get("role"),
            model=body.get("model"),
            is_default=body.get("is_default"),
        )
        if agent is None:
            return JSONResponse({"error": "Agent not found"}, status_code=404)
        return JSONResponse(agent.model_dump())

    @require_auth
    async def handle_delete_agent(request: Request):
        agent_id = request.path_params["agent_id"]
        deleted = await _run_in_thread(agent_store.delete_agent, agent_id)
        if not deleted:
            return JSONResponse({"error": "Agent not found"}, status_code=404)
        return JSONResponse({"status": "deleted"})

    @require_auth
    async def handle_set_default_agent(request: Request):
        agent_id = request.path_params["agent_id"]
        ok = await _run_in_thread(agent_store.set_default, agent_id)
        if not ok:
            return JSONResponse({"error": "Agent not found"}, status_code=404)
        return JSONResponse({"status": "ok"})

    return [
        Route("/admin/api/agents", handle_list_agents),
        Route("/admin/api/agents", handle_create_agent, methods=["POST"]),
        Route("/admin/api/agents/{agent_id}", handle_get_agent),
        Route("/admin/api/agents/{agent_id}", handle_update_agent, methods=["PATCH"]),
        Route("/admin/api/agents/{agent_id}", handle_delete_agent, methods=["DELETE"]),
        Route("/admin/api/agents/{agent_id}/default", handle_set_default_agent, methods=["POST"]),
    ]
