"""Tests for the admin portal API and auth."""

import base64

from aiohttp import web
from aiohttp.test_utils import AioHTTPTestCase

from cliver.gateway.admin import register_admin_routes


def _auth_header(username="admin", password="secret"):
    creds = base64.b64encode(f"{username}:{password}".encode()).decode()
    return {"Authorization": f"Basic {creds}"}


def _make_admin_app(username="admin", password="secret"):
    app = web.Application()

    async def mock_get_status():
        return {"uptime": 42, "tasks_run": 3, "platforms": ["slack"], "adapters": []}

    ctx = {
        "get_status": mock_get_status,
        "agent_name": "TestAgent",
        "config_dir": None,
    }
    register_admin_routes(app, username=username, password=password, context=ctx)
    return app


class TestAdminAuth(AioHTTPTestCase):
    async def get_application(self):
        return _make_admin_app()

    async def test_admin_page_requires_auth(self):
        resp = await self.client.request("GET", "/admin")
        assert resp.status == 401

    async def test_admin_page_wrong_password(self):
        resp = await self.client.request("GET", "/admin", headers=_auth_header("admin", "wrong"))
        assert resp.status == 401

    async def test_admin_page_succeeds_with_correct_auth(self):
        resp = await self.client.request("GET", "/admin", headers=_auth_header())
        assert resp.status == 200
        text = await resp.text()
        assert "CLIver" in text

    async def test_api_status_requires_auth(self):
        resp = await self.client.request("GET", "/admin/api/status")
        assert resp.status == 401

    async def test_api_status_returns_json(self):
        resp = await self.client.request("GET", "/admin/api/status", headers=_auth_header())
        assert resp.status == 200
        data = await resp.json()
        assert data["uptime"] == 42
        assert data["tasks_run"] == 3


class TestAdminDisabled(AioHTTPTestCase):
    async def get_application(self):
        return _make_admin_app(username=None, password=None)

    async def test_admin_disabled_returns_403(self):
        resp = await self.client.request("GET", "/admin")
        assert resp.status == 403


class TestAdminApiEndpoints(AioHTTPTestCase):
    async def get_application(self):
        return _make_admin_app()

    async def test_tasks_endpoint(self):
        resp = await self.client.request("GET", "/admin/api/tasks", headers=_auth_header())
        assert resp.status == 200
        data = await resp.json()
        assert isinstance(data, list)

    async def test_sessions_endpoint(self):
        resp = await self.client.request("GET", "/admin/api/sessions", headers=_auth_header())
        assert resp.status == 200
        data = await resp.json()
        assert isinstance(data, list)

    async def test_workflows_endpoint(self):
        resp = await self.client.request("GET", "/admin/api/workflows", headers=_auth_header())
        assert resp.status == 200
        data = await resp.json()
        assert isinstance(data, list)

    async def test_skills_endpoint(self):
        resp = await self.client.request("GET", "/admin/api/skills", headers=_auth_header())
        assert resp.status == 200
        data = await resp.json()
        assert isinstance(data, list)

    async def test_adapters_endpoint(self):
        resp = await self.client.request("GET", "/admin/api/adapters", headers=_auth_header())
        assert resp.status == 200
        data = await resp.json()
        assert isinstance(data, list)

    async def test_agent_endpoint(self):
        resp = await self.client.request("GET", "/admin/api/agent", headers=_auth_header())
        assert resp.status == 200
        data = await resp.json()
        assert "name" in data

    async def test_config_endpoint(self):
        resp = await self.client.request("GET", "/admin/api/config", headers=_auth_header())
        assert resp.status == 200
        data = await resp.json()
        assert isinstance(data, dict)
