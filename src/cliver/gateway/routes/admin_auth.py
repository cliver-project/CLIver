"""Auth routes for the admin portal (login, logout)."""

from __future__ import annotations

from starlette.requests import Request
from starlette.responses import JSONResponse, RedirectResponse
from starlette.routing import Route


def get_auth_routes(username: str, password: str, session_secret: str) -> list:
    """Return login and logout routes."""

    async def handle_login_submit(request: Request):
        try:
            data = await request.json()
        except Exception:
            return JSONResponse({"error": "Invalid request"}, status_code=400)
        u = data.get("username", "")
        p = data.get("password", "")
        if u == username and p == password:
            from cliver.gateway.admin import _make_session_token

            token = _make_session_token(username, session_secret)
            resp = JSONResponse({"status": "ok"})
            resp.set_cookie("cliver_session", token, httponly=True, samesite="lax", path="/admin")
            return resp
        return JSONResponse({"error": "Invalid credentials"}, status_code=401)

    async def handle_logout(request: Request):
        resp = RedirectResponse("/admin/login", status_code=302)
        resp.delete_cookie("cliver_session", path="/admin")
        return resp

    return [
        Route("/admin/api/login", handle_login_submit, methods=["POST"]),
        Route("/admin/logout", handle_logout),
    ]
