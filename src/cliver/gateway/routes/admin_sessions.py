"""Session management routes for the admin portal."""

from __future__ import annotations

import logging
from typing import Callable

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

logger = logging.getLogger(__name__)


def _get_session_manager(ctx, source):
    if source == "gateway":
        gw = ctx.get("gateway")
        return gw._session_manager if gw else None
    elif source == "cli":
        return ctx.get("cli_session_manager")
    return None


def _get_sessions_by_source(ctx, source):
    sm = _get_session_manager(ctx, source)
    if not sm:
        return []
    try:
        sessions = sm.list_sessions()
        if source == "gateway":
            for s in sessions:
                title = s.get("title") or ""
                parts = title.split(":", 2)
                if len(parts) >= 2:
                    s["platform"] = parts[0]
                    turns = sm.load_turns(s["id"])
                    first_user = next((t["content"] for t in turns if t["role"] == "user"), None)
                    if first_user:
                        s["display_title"] = first_user[:80]
        return sessions
    except Exception as e:
        logger.warning("Failed to get %s sessions: %s", source, e)
        return []


def _get_session_turns(ctx, source, session_id):
    sm = _get_session_manager(ctx, source)
    if not sm:
        return []
    try:
        return sm.load_turns(session_id)
    except Exception as e:
        logger.warning("Failed to get session turns: %s", e)
        return []


def _delete_session(ctx, source, session_id):
    sm = _get_session_manager(ctx, source)
    if not sm:
        return False
    try:
        return sm.delete_session(session_id)
    except Exception as e:
        logger.warning("Failed to delete session: %s", e)
        return False


def get_session_routes(context: dict, require_auth: Callable) -> list:
    """Return session management API routes."""

    @require_auth
    async def handle_sessions_by_source(request: Request):
        from cliver.gateway.admin import _run_in_thread

        source = request.path_params["source"]
        if source not in ("cli", "gateway"):
            return JSONResponse({"error": f"Invalid source '{source}'. Use 'cli' or 'gateway'."}, status_code=400)
        sessions = await _run_in_thread(_get_sessions_by_source, context, source)
        return JSONResponse(sessions)

    @require_auth
    async def handle_session_turns(request: Request):
        from cliver.gateway.admin import _run_in_thread

        source = request.path_params["source"]
        session_id = request.path_params["id"]
        if source not in ("cli", "gateway"):
            return JSONResponse({"error": f"Invalid source '{source}'."}, status_code=400)
        turns = await _run_in_thread(_get_session_turns, context, source, session_id)
        return JSONResponse(turns)

    @require_auth
    async def handle_delete_session(request: Request):
        source = request.path_params["source"]
        session_id = request.path_params["id"]
        if source not in ("cli", "gateway"):
            return JSONResponse({"error": f"Invalid source '{source}'."}, status_code=400)
        logger.info("[admin] Session '%s' (%s) deleted via admin portal", session_id, source)
        deleted = _delete_session(context, source, session_id)
        if deleted:
            return JSONResponse({"status": "deleted"})
        return JSONResponse({"error": "session not found"}, status_code=404)

    return [
        Route("/admin/api/sessions/{source}", handle_sessions_by_source),
        Route("/admin/api/sessions/{source}/{id}", handle_session_turns),
        Route("/admin/api/sessions/{source}/{id}", handle_delete_session, methods=["DELETE"]),
    ]
