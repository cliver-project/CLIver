"""MCP Server API routes for the admin portal (config.yaml backed)."""

from __future__ import annotations

import logging
from typing import Callable

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

logger = logging.getLogger(__name__)


def get_mcp_routes(config_manager, require_auth: Callable) -> list:
    """Return MCP server CRUD API routes backed by config.yaml."""
    from cliver.gateway.admin import _run_in_thread

    @require_auth
    async def handle_list_servers(request: Request):
        servers = await _run_in_thread(config_manager.list_mcp_servers)
        result = []
        for name, srv in servers.items():
            d = {"id": name, "name": name, "transport": srv.transport}
            if srv.transport == "stdio":
                d["command"] = getattr(srv, "command", None)
                d["args"] = srv.args if hasattr(srv, "args") else None
                d["env"] = srv.env if hasattr(srv, "env") else None
            else:
                d["url"] = getattr(srv, "url", None)
                d["headers"] = srv.headers if hasattr(srv, "headers") else None
            result.append(d)
        return JSONResponse(result)

    @require_auth
    async def handle_create_server(request: Request):
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "Invalid JSON"}, status_code=400)
        name = (body.get("name") or "").strip()
        if not name:
            return JSONResponse({"error": "name is required"}, status_code=400)
        try:
            await _run_in_thread(
                config_manager.add_or_update_mcp_server,
                name=name,
                transport=body.get("transport", "stdio"),
                command=body.get("command"),
                args=body.get("args"),
                env=body.get("env"),
                url=body.get("url"),
                headers=body.get("headers"),
            )
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        return JSONResponse({"id": name, "name": name})

    @require_auth
    async def handle_get_server(request: Request):
        server_id = request.path_params["server_id"]
        servers = await _run_in_thread(config_manager.list_mcp_servers)
        srv = servers.get(server_id)
        if srv is None:
            return JSONResponse({"error": "MCP server not found"}, status_code=404)
        d = {"id": server_id, "name": server_id, "transport": srv.transport}
        if srv.transport == "stdio":
            d["command"] = getattr(srv, "command", None)
            d["args"] = srv.args if hasattr(srv, "args") else None
            d["env"] = srv.env if hasattr(srv, "env") else None
        else:
            d["url"] = getattr(srv, "url", None)
            d["headers"] = srv.headers if hasattr(srv, "headers") else None
        return JSONResponse(d)

    @require_auth
    async def handle_update_server(request: Request):
        server_id = request.path_params["server_id"]
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "Invalid JSON"}, status_code=400)
        servers = await _run_in_thread(config_manager.list_mcp_servers)
        existing = servers.get(server_id)
        if existing is None:
            return JSONResponse({"error": "MCP server not found"}, status_code=404)
        try:
            await _run_in_thread(
                config_manager.add_or_update_mcp_server,
                name=server_id,
                transport=existing.transport,
                command=body.get("command") or getattr(existing, "command", None),
                args=body.get("args")
                if body.get("args") is not None
                else (existing.args if hasattr(existing, "args") else None),
                env=body.get("env")
                if body.get("env") is not None
                else (existing.env if hasattr(existing, "env") else None),
                url=body.get("url") if body.get("url") is not None else getattr(existing, "url", None),
                headers=body.get("headers")
                if body.get("headers") is not None
                else (existing.headers if hasattr(existing, "headers") else None),
            )
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        return JSONResponse({"id": server_id, "name": server_id})

    @require_auth
    async def handle_delete_server(request: Request):
        server_id = request.path_params["server_id"]
        deleted = await _run_in_thread(config_manager.remove_mcp_server, server_id)
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
