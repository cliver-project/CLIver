"""
Admin portal for CLIver gateway.

Provides a web-based admin interface with cookie-based session auth
for monitoring and managing the gateway: tasks, sessions,
skills, adapters, agent info, and configuration.

Route implementations are split into:
- gateway/routes/admin_auth.py   — login / logout
- gateway/routes/admin_tasks.py  — task CRUD + run
- gateway/routes/admin_sessions.py — session management
- gateway/routes/admin_browse.py — file browser
- gateway/routes/admin_info.py   — status, skills, adapters, agent, config, keys, models
- gateway/routes/admin_chat.py   — streaming chat (SSE)
- gateway/routes/admin_spa.py    — SPA static file serving

Usage:
    from cliver.gateway.admin import get_admin_routes
    routes = get_admin_routes(username="admin", password="secret", context={...})
"""

from __future__ import annotations

import asyncio
import base64
import functools
import hashlib
import hmac
import logging
import secrets
from pathlib import Path
from typing import Optional

from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse
from starlette.routing import Route

logger = logging.getLogger(__name__)

_TEMPLATE_DIR = Path(__file__).parent / "templates"
_PAGES_DIR = _TEMPLATE_DIR / "pages"


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------


def _check_basic_auth(request, username: str, password: str) -> bool:
    """Decode Basic Auth header and compare credentials."""
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Basic "):
        return False
    try:
        decoded = base64.b64decode(auth[6:]).decode("utf-8")
    except Exception:
        return False
    parts = decoded.split(":", 1)
    if len(parts) != 2:
        return False
    return parts[0] == username and parts[1] == password


def _make_session_token(username: str, secret: str) -> str:
    """Create an HMAC-signed session token."""
    sig = hmac.new(secret.encode(), username.encode(), hashlib.sha256).hexdigest()
    return f"{username}:{sig}"


def _check_session_cookie(request, username: str, secret: str) -> bool:
    """Validate the session cookie."""
    cookie = request.cookies.get("cliver_session", "")
    if not cookie or ":" not in cookie:
        return False
    expected = _make_session_token(username, secret)
    return hmac.compare_digest(cookie, expected)


# ---------------------------------------------------------------------------
# Page rendering
# ---------------------------------------------------------------------------


def _render_page(page_name, context, extra_replacements=None):
    """Assemble base.html + pages/{page_name}.html with substitutions."""
    base_path = _TEMPLATE_DIR / "base.html"
    page_path = _PAGES_DIR / f"{page_name}.html"
    if not page_path.exists():
        return None
    base_html = base_path.read_text(encoding="utf-8")
    page_html = page_path.read_text(encoding="utf-8")
    html = base_html.replace("{{CONTENT}}", page_html)
    html = html.replace("{{AGENT_NAME}}", context.get("profile_name", "CLIver"))
    html = html.replace("{{BASE_URL}}", "/admin")
    html = html.replace("{{CURRENT_PAGE}}", page_name.split("_")[0])
    if extra_replacements:
        for key, value in extra_replacements.items():
            html = html.replace(key, value)
    return html


def _render_login_page(context, error=None):
    """Render the standalone login page."""
    page_path = _PAGES_DIR / "login.html"
    if not page_path.exists():
        return "<html><body><h1>Login page not found</h1></body></html>"
    html = page_path.read_text(encoding="utf-8")
    html = html.replace("{{AGENT_NAME}}", context.get("profile_name", "CLIver"))
    html = html.replace("{{ERROR}}", error or "")
    html = html.replace("{{ERROR_DISPLAY}}", "block" if error else "none")
    return html


# ---------------------------------------------------------------------------
# Thread helper (used by task/session route modules)
# ---------------------------------------------------------------------------


async def _run_in_thread(fn, *args):
    """Run a blocking function in a thread executor to avoid blocking the event loop."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, fn, *args)


# ---------------------------------------------------------------------------
# Route assembly
# ---------------------------------------------------------------------------


def get_admin_routes(
    username: Optional[str],
    password: Optional[str],
    context: dict,
):
    """Return admin portal routes assembled from sub-modules.

    Args:
        username: Admin username (None = portal disabled)
        password: Admin password (None = portal disabled)
        context: Dict with keys:
            - get_status: async callable or dict returning status info
            - agent_name: str
            - config_dir: Path or None
            - gateway: Gateway instance (optional)
            - cli_session_manager: SessionManager for CLI sessions (optional)

    Returns:
        (api_routes, spa_routes, require_auth)
    """
    session_secret = secrets.token_hex(32)

    # --- Auth decorator (cookie-first, Basic Auth fallback) ---

    def require_auth(handler):
        @functools.wraps(handler)
        async def wrapper(request: Request):
            if username is None or password is None:
                return JSONResponse({"error": "Admin portal is disabled"}, status_code=403)
            if _check_session_cookie(request, username, session_secret):
                return await handler(request)
            if _check_basic_auth(request, username, password):
                return await handler(request)
            is_api = request.url.path.startswith("/admin/api/")
            if is_api:
                return JSONResponse({"error": "Unauthorized"}, status_code=401)
            return RedirectResponse("/admin/login", status_code=302)

        return wrapper

    # --- SPA dist directory ---

    spa_dist_dir = Path(__file__).parent.parent.parent.parent / "admin" / "dist"
    if not spa_dist_dir.exists():
        spa_dist_dir = Path(__file__).parent / "admin_dist"

    # --- Login page route (unauthenticated) ---

    async def handle_login_page(request: Request):
        html = _render_login_page(context)
        return HTMLResponse(html)

    # --- Assemble routes from sub-modules ---

    from cliver.gateway.routes.admin_auth import get_auth_routes
    from cliver.gateway.routes.admin_browse import get_browse_routes
    from cliver.gateway.routes.admin_chat import get_chat_routes
    from cliver.gateway.routes.admin_conversations import get_conversations_routes
    from cliver.gateway.routes.admin_info import get_info_routes
    from cliver.gateway.routes.admin_sessions import get_session_routes
    from cliver.gateway.routes.admin_spa import get_spa_routes
    from cliver.gateway.routes.admin_tasks import get_task_routes

    async def handle_root(request: Request):
        if username and password:
            return RedirectResponse("/admin", status_code=302)
        return JSONResponse({"status": "ok", "service": "CLIver Gateway"})

    api_routes = [
        Route("/", handle_root),
        Route("/admin/login", handle_login_page),
        *get_auth_routes(username, password, session_secret),
        *get_task_routes(context, require_auth),
        *get_session_routes(context, require_auth),
        *get_browse_routes(require_auth),
        *get_info_routes(context, require_auth),
        *get_chat_routes(context, require_auth),
        *get_conversations_routes(context, require_auth),
    ]

    spa_routes = get_spa_routes(spa_dist_dir)

    return api_routes, spa_routes, require_auth
