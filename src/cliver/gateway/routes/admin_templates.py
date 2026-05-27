"""Chat template CRUD routes for the admin portal."""

from __future__ import annotations

import logging
from typing import Callable

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from cliver.chat_templates import ChatTemplate

logger = logging.getLogger(__name__)


def get_template_routes(
    template_store: "ChatTemplateStore",
    require_auth: Callable,
) -> list:
    """Return template CRUD routes."""

    from cliver.chat_templates import ChatTemplateStore

    @require_auth
    async def handle_list(request: Request):
        templates = template_store.list_all()
        return JSONResponse([t.model_dump() for t in templates])

    @require_auth
    async def handle_create(request: Request):
        try:
            data = await request.json()
        except Exception:
            return JSONResponse({"error": "Invalid JSON"}, status_code=400)

        template_id = data.get("id", "").strip()
        if not template_id:
            return JSONResponse({"error": "id is required"}, status_code=400)

        template = ChatTemplate(
            id=template_id,
            system_prompt=data.get("system_prompt", ""),
            skills=data.get("skills", []),
            agent=data.get("agent"),
            knowledge_base=data.get("knowledge_base"),
            description=data.get("description"),
        )
        template_store.save_to_user(template)
        return JSONResponse(template.model_dump())

    @require_auth
    async def handle_update(request: Request):
        template_id = request.path_params["id"]
        try:
            data = await request.json()
        except Exception:
            return JSONResponse({"error": "Invalid JSON"}, status_code=400)

        existing = template_store.get(template_id)
        if not existing:
            return JSONResponse({"error": "Template not found"}, status_code=404)

        if "system_prompt" in data:
            existing.system_prompt = data["system_prompt"]
        if "skills" in data:
            existing.skills = data["skills"]
        if "agent" in data:
            existing.agent = data["agent"]
        if "knowledge_base" in data:
            existing.knowledge_base = data["knowledge_base"]
        if "description" in data:
            existing.description = data["description"]

        template_store.save_to_user(existing)
        return JSONResponse(existing.model_dump())

    @require_auth
    async def handle_delete(request: Request):
        template_id = request.path_params["id"]
        if template_store.delete_from_user(template_id):
            return JSONResponse({"status": "ok"})
        return JSONResponse({"error": "Template not found"}, status_code=404)

    return [
        Route("/admin/api/templates", handle_list),
        Route("/admin/api/templates", handle_create, methods=["POST"]),
        Route("/admin/api/templates/{id}", handle_update, methods=["PUT"]),
        Route("/admin/api/templates/{id}", handle_delete, methods=["DELETE"]),
    ]
