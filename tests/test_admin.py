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
    routes = get_admin_routes(username=username, password=password, context=ctx)
    app = Starlette(routes=routes)
    return app


class TestAdminAuth:
    def test_admin_root_redirects_to_gateway(self):
        client = TestClient(_make_admin_app(), follow_redirects=False)
        resp = client.get("/admin", headers=_auth_header())
        assert resp.status_code == 302
        assert resp.headers["Location"] == "/admin/gateway"

    def test_admin_page_requires_auth(self):
        client = TestClient(_make_admin_app(), follow_redirects=False)
        resp = client.get("/admin/gateway")
        assert resp.status_code == 302
        assert "/admin/login" in resp.headers["Location"]

    def test_admin_page_wrong_password(self):
        client = TestClient(_make_admin_app(), follow_redirects=False)
        resp = client.get("/admin/gateway", headers=_auth_header("admin", "wrong"))
        assert resp.status_code == 302

    def test_admin_page_succeeds_with_correct_auth(self):
        client = TestClient(_make_admin_app())
        resp = client.get("/admin/gateway", headers=_auth_header())
        assert resp.status_code == 200
        assert "Admin Portal" in resp.text

    def test_admin_unknown_page_returns_404(self):
        client = TestClient(_make_admin_app())
        resp = client.get("/admin/nonexistent", headers=_auth_header())
        assert resp.status_code == 404

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
    def test_admin_disabled_returns_403(self):
        client = TestClient(_make_admin_app(username=None, password=None))
        resp = client.get("/admin/gateway")
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

    def test_workflows_endpoint(self):
        client = TestClient(_make_admin_app())
        resp = client.get("/admin/api/workflows", headers=_auth_header())
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

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

    def test_workflow_detail_endpoint(self):
        client = TestClient(_make_admin_app())
        resp = client.get("/admin/api/workflows/nonexistent", headers=_auth_header())
        assert resp.status_code == 200
