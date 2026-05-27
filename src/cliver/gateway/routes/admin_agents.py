"""Agent API routes for the admin portal (config.yaml backed)."""

from __future__ import annotations

import logging
from typing import Callable

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

logger = logging.getLogger(__name__)


def get_agent_routes(config_manager, require_auth: Callable) -> list:
    """Return agent CRUD API routes backed by config.yaml."""
    from cliver.gateway.admin import _run_in_thread

    def _get_agents_dict(config_manager):
        """Lazily ensure config.agents dict exists."""
        cfg = config_manager.config
        if cfg.agents is None:
            cfg.agents = {}
            config_manager._save_config()
        return cfg.agents

    @require_auth
    async def handle_list_agents(request: Request):
        agents = await _run_in_thread(_get_agents_dict, config_manager)
        result = []
        for name, a in agents.items():
            result.append(
                {
                    "id": name,
                    "name": name,
                    "description": a.description,
                    "role": a.role,
                    "system_prompt": a.system_prompt,
                    "model": a.model,
                    "skills": a.skills,
                    "toolsets": a.toolsets,
                    "is_default": 1 if config_manager.config.default_agent == name else 0,
                }
            )
        return JSONResponse(result)

    @require_auth
    async def handle_create_agent(request: Request):
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "Invalid JSON"}, status_code=400)
        name = (body.get("name") or "").strip()
        if not name:
            return JSONResponse({"error": "name is required"}, status_code=400)

        def _create():
            agents = _get_agents_dict(config_manager)
            if name in agents:
                raise ValueError(f"Agent '{name}' already exists")
            from cliver.config import AgentConfig

            agents[name] = AgentConfig(
                name=name,
                description=body.get("description"),
                role=body.get("role"),
                system_prompt=body.get("system_prompt"),
                model=body.get("model"),
                skills=body.get("skills"),
                toolsets=body.get("toolsets"),
            )
            if body.get("is_default"):
                config_manager.config.default_agent = name
            config_manager._save_config()
            return {"id": name, "name": name}

        try:
            result = await _run_in_thread(_create)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        return JSONResponse(result)

    @require_auth
    async def handle_get_agent(request: Request):
        agent_id = request.path_params["agent_id"]
        agents = await _run_in_thread(_get_agents_dict, config_manager)
        a = agents.get(agent_id)
        if a is None:
            return JSONResponse({"error": "Agent not found"}, status_code=404)
        return JSONResponse(
            {
                "id": agent_id,
                "name": agent_id,
                "description": a.description,
                "system_prompt": a.system_prompt,
                "model": a.model,
                "skills": a.skills,
                "toolsets": a.toolsets,
                "is_default": 1 if config_manager.config.default_agent == agent_id else 0,
            }
        )

    @require_auth
    async def handle_update_agent(request: Request):
        agent_id = request.path_params["agent_id"]

        def _update():
            agents = _get_agents_dict(config_manager)
            a = agents.get(agent_id)
            if a is None:
                return None
            return a

        existing = await _run_in_thread(_update)
        if existing is None:
            return JSONResponse({"error": "Agent not found"}, status_code=404)

        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "Invalid JSON"}, status_code=400)

        def _apply():
            agents = _get_agents_dict(config_manager)
            a = agents[agent_id]
            for field in ("description", "role", "system_prompt", "model", "skills", "toolsets"):
                if body.get(field) is not None:
                    setattr(a, field, body[field])
            if "is_default" in body:
                config_manager.config.default_agent = agent_id if body["is_default"] else None
            config_manager._save_config()
            return {"id": agent_id, "name": agent_id}

        result = await _run_in_thread(_apply)
        return JSONResponse(result)

    @require_auth
    async def handle_delete_agent(request: Request):
        agent_id = request.path_params["agent_id"]

        def _delete():
            agents = _get_agents_dict(config_manager)
            if agent_id not in agents:
                return False
            del agents[agent_id]
            config_manager._save_config()
            return True

        deleted = await _run_in_thread(_delete)
        if not deleted:
            return JSONResponse({"error": "Agent not found"}, status_code=404)
        return JSONResponse({"status": "deleted"})

    return [
        Route("/admin/api/agents", handle_list_agents),
        Route("/admin/api/agents", handle_create_agent, methods=["POST"]),
        Route("/admin/api/agents/{agent_id}", handle_get_agent),
        Route("/admin/api/agents/{agent_id}", handle_update_agent, methods=["PATCH"]),
        Route("/admin/api/agents/{agent_id}", handle_delete_agent, methods=["DELETE"]),
    ]
