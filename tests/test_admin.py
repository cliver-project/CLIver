"""Tests for the admin portal — multi-page routing, auth, and APIs."""

import base64

from starlette.applications import Starlette
from starlette.testclient import TestClient

from cliver.gateway.admin import get_admin_routes


def _auth_header(username="admin", password="secret"):
    creds = base64.b64encode(f"{username}:{password}".encode()).decode()
    return {"Authorization": f"Basic {creds}"}


def _make_admin_app(username="admin", password="secret"):
    async def mock_get_status():
        return {"uptime": 42, "tasks_run": 3, "platforms": ["slack"], "adapters": []}

    ctx = {
        "get_status": mock_get_status,
        "agent_name": "TestAgent",
        "config_dir": None,
    }
    api_routes, spa_routes, _auth = get_admin_routes(username=username, password=password, context=ctx)
    app = Starlette(routes=api_routes + spa_routes)
    return app


class TestAdminAuth:
    def test_admin_root_serves_spa(self):
        """Test that /admin now serves the React SPA instead of redirecting."""
        client = TestClient(_make_admin_app(), follow_redirects=False)
        resp = client.get("/admin", headers=_auth_header())
        assert resp.status_code == 200
        # Either the SPA index.html or the 503 "portal not built" message
        assert "<!doctype html>" in resp.text.lower() or "Admin portal not built" in resp.text

    def test_admin_page_serves_spa_without_auth(self):
        """Test that unauthenticated users get the SPA (which handles login client-side)."""
        client = TestClient(_make_admin_app(), follow_redirects=False)
        resp = client.get("/admin/gateway")
        assert resp.status_code == 200
        assert "<!doctype html>" in resp.text.lower() or "Admin portal not built" in resp.text

    def test_admin_api_requires_auth(self):
        """Test that API routes require authentication."""
        client = TestClient(_make_admin_app(), follow_redirects=False)
        resp = client.get("/admin/api/status")
        assert resp.status_code == 401

    def test_admin_page_succeeds_with_correct_auth(self):
        """Test that authenticated users get the SPA."""
        client = TestClient(_make_admin_app())
        resp = client.get("/admin/gateway", headers=_auth_header())
        assert resp.status_code == 200
        # Check for React SPA or the 503 fallback
        assert "<!doctype html>" in resp.text.lower() or "Admin portal not built" in resp.text

    def test_admin_unknown_page_serves_spa(self):
        """Test that unknown routes now serve the SPA (React Router handles 404s)."""
        client = TestClient(_make_admin_app())
        resp = client.get("/admin/nonexistent", headers=_auth_header())
        assert resp.status_code == 200
        assert "<!doctype html>" in resp.text.lower() or "Admin portal not built" in resp.text

    def test_api_status_requires_auth(self):
        client = TestClient(_make_admin_app())
        resp = client.get("/admin/api/status")
        assert resp.status_code == 401

    def test_api_status_returns_json(self):
        client = TestClient(_make_admin_app())
        resp = client.get("/admin/api/status", headers=_auth_header())
        assert resp.status_code == 200
        data = resp.json()
        assert data["uptime"] == 42


class TestAdminDisabled:
    def test_admin_disabled_api_returns_403(self):
        """Test that when admin is disabled, API calls return 403."""
        client = TestClient(_make_admin_app(username=None, password=None))
        resp = client.get("/admin/api/status")
        assert resp.status_code == 403


class TestAdminApiEndpoints:
    def test_tasks_endpoint(self):
        client = TestClient(_make_admin_app())
        resp = client.get("/admin/api/tasks", headers=_auth_header())
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_sessions_cli_endpoint(self):
        client = TestClient(_make_admin_app())
        resp = client.get("/admin/api/sessions/cli", headers=_auth_header())
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_sessions_gateway_endpoint(self):
        client = TestClient(_make_admin_app())
        resp = client.get("/admin/api/sessions/gateway", headers=_auth_header())
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_sessions_invalid_source_returns_400(self):
        client = TestClient(_make_admin_app())
        resp = client.get("/admin/api/sessions/invalid", headers=_auth_header())
        assert resp.status_code == 400

    def test_skills_endpoint(self):
        client = TestClient(_make_admin_app())
        resp = client.get("/admin/api/skills", headers=_auth_header())
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_adapters_endpoint(self):
        client = TestClient(_make_admin_app())
        resp = client.get("/admin/api/adapters", headers=_auth_header())
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_agent_endpoint(self):
        client = TestClient(_make_admin_app())
        resp = client.get("/admin/api/agent", headers=_auth_header())
        assert resp.status_code == 200
        assert "name" in resp.json()

    def test_config_endpoint(self):
        client = TestClient(_make_admin_app())
        resp = client.get("/admin/api/config", headers=_auth_header())
        assert resp.status_code == 200
        assert isinstance(resp.json(), dict)

    def test_task_detail_endpoint(self):
        client = TestClient(_make_admin_app())
        resp = client.get("/admin/api/tasks/nonexistent", headers=_auth_header())
        assert resp.status_code == 200
