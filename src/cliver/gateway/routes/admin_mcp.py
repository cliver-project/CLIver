"""MCP Server API routes for the admin portal."""

from __future__ import annotations

import logging
from typing import Callable

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

logger = logging.getLogger(__name__)


def get_mcp_routes(mcp_store, require_auth: Callable) -> list:
    """Return MCP server CRUD API routes."""
    from cliver.gateway.admin import _run_in_thread

    @require_auth
    async def handle_list_servers(request: Request):
        servers = await _run_in_thread(mcp_store.list_servers)
        return JSONResponse([s.model_dump() for s in servers])

    @require_auth
    async def handle_create_server(request: Request):
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "Invalid JSON"}, status_code=400)
        name = (body.get("name") or "").strip()
        if not name:
            return JSONResponse({"error": "name is required"}, status_code=400)
        server = await _run_in_thread(
            mcp_store.create_server,
            name,
            body.get("transport", "stdio"),
            body.get("url"),
            body.get("auth"),
            body.get("headers"),
            body.get("command"),
            body.get("args"),
            body.get("envs"),
        )
        return JSONResponse(server.model_dump())

    @require_auth
    async def handle_get_server(request: Request):
        server_id = request.path_params["server_id"]
        server = await _run_in_thread(mcp_store.get_server, server_id)
        if server is None:
            return JSONResponse({"error": "MCP server not found"}, status_code=404)
        return JSONResponse(server.model_dump())

    @require_auth
    async def handle_update_server(request: Request):
        server_id = request.path_params["server_id"]
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "Invalid JSON"}, status_code=400)
        server = await _run_in_thread(
            mcp_store.update_server,
            server_id,
            body.get("name"),
            body.get("transport"),
            body.get("url"),
            body.get("auth"),
            body.get("headers"),
            body.get("command"),
            body.get("args"),
            body.get("envs"),
        )
        if server is None:
            return JSONResponse({"error": "MCP server not found"}, status_code=404)
        return JSONResponse(server.model_dump())

    @require_auth
    async def handle_delete_server(request: Request):
        server_id = request.path_params["server_id"]
        deleted = await _run_in_thread(mcp_store.delete_server, server_id)
        if not deleted:
            return JSONResponse({"error": "MCP server not found"}, status_code=404)
        return JSONResponse({"status": "deleted"})

    return [
        Route("/admin/api/mcp-servers", handle_list_servers),
        Route("/admin/api/mcp-servers", handle_create_server, methods=["POST"]),
        Route("/admin/api/mcp-servers/{server_id}", handle_get_server),
        Route("/admin/api/mcp-servers/{server_id}", handle_update_server, methods=["PATCH"]),
        Route("/admin/api/mcp-servers/{server_id}", handle_delete_server, methods=["DELETE"]),
    ]
