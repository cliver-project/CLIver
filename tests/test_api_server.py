"""Tests for OpenAI-compatible API server."""

from unittest.mock import MagicMock

import pytest
from aiohttp import web
from aiohttp.test_utils import AioHTTPTestCase

from cliver.gateway.api_server import (
    _build_completion_response,
    _build_stream_chunk,
    _build_system_appender,
    _configure_server_mode,
    _parse_chat_request,
    register_routes,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_executor(models=None, default_model="qwen"):
    """Create a mock AgentCore with typical defaults."""
    executor = MagicMock()
    executor.llm_models = models or {"qwen": MagicMock(), "deepseek": MagicMock()}
    executor.default_model = default_model
    executor.permission_manager = None
    return executor


def _mock_status():
    """Return a status dict like the Gateway would."""
    return {"uptime": 42, "tasks_run": 3, "platforms": ["slack"]}


def _make_app(executor=None, api_key=None, get_status=None):
    """Create an aiohttp app with routes registered."""
    app = web.Application()
    register_routes(
        app,
        executor or _mock_executor(),
        get_status or _mock_status,
        api_key=api_key,
    )
    return app


# ---------------------------------------------------------------------------
# Unit tests for module-level helpers
# ---------------------------------------------------------------------------


class TestParseHelpers:
    """Test request parsing and response building (no HTTP needed)."""

    def test_parse_chat_request_basic(self):
        executor = _mock_executor()
        body = {"model": "qwen", "messages": [{"role": "user", "content": "hello"}]}
        parsed = _parse_chat_request(body, executor)
        assert parsed["user_input"] == "hello"
        assert parsed["model"] == "qwen"
        assert parsed["stream"] is False

    def test_parse_chat_request_with_history(self):
        executor = _mock_executor()
        body = {
            "model": "qwen",
            "messages": [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "answer"},
                {"role": "user", "content": "second"},
            ],
        }
        parsed = _parse_chat_request(body, executor)
        assert parsed["user_input"] == "second"
        assert len(parsed["conversation_history"]) == 2
        assert parsed["system_message"] == "You are helpful"

    def test_parse_chat_request_stream(self):
        executor = _mock_executor()
        body = {"model": "qwen", "messages": [{"role": "user", "content": "hello"}], "stream": True}
        parsed = _parse_chat_request(body, executor)
        assert parsed["stream"] is True

    def test_parse_chat_request_options(self):
        executor = _mock_executor()
        body = {
            "model": "qwen",
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0.5,
            "max_tokens": 100,
        }
        parsed = _parse_chat_request(body, executor)
        assert parsed["options"]["temperature"] == 0.5
        assert parsed["options"]["max_tokens"] == 100

    def test_parse_chat_request_default_model(self):
        executor = _mock_executor()
        body = {"messages": [{"role": "user", "content": "hello"}]}
        parsed = _parse_chat_request(body, executor)
        assert parsed["model"] == "qwen"

    def test_parse_chat_request_missing_messages(self):
        executor = _mock_executor()
        with pytest.raises(ValueError, match="messages"):
            _parse_chat_request({}, executor)

    def test_build_completion_response(self):
        result = _build_completion_response("chatcmpl-123", "qwen", "Hello!", 10, 5)
        assert result["model"] == "qwen"
        assert result["choices"][0]["message"]["content"] == "Hello!"
        assert result["choices"][0]["finish_reason"] == "stop"
        assert result["usage"]["total_tokens"] == 15

    def test_build_stream_chunk(self):
        chunk = _build_stream_chunk("chatcmpl-123", "qwen", "Hello")
        assert chunk["choices"][0]["delta"]["content"] == "Hello"
        assert chunk["choices"][0]["finish_reason"] is None


class TestServerMode:
    """Test server mode configuration."""

    def test_configure_server_mode_with_permission_manager(self):
        from cliver.permissions import PermissionMode

        executor = _mock_executor()
        mock_pm = MagicMock()
        executor.permission_manager = mock_pm

        _configure_server_mode(executor)
        mock_pm.set_mode.assert_called_once_with(PermissionMode.YOLO)
        assert executor.on_permission_prompt is not None
        assert executor.on_permission_prompt("any_tool", {}) == "allow"

    def test_configure_server_mode_without_permission_manager(self):
        executor = _mock_executor()
        executor.permission_manager = None

        _configure_server_mode(executor)
        assert executor.on_permission_prompt is not None

    def test_system_appender_with_user_message(self):
        appender = _build_system_appender("Custom system message")
        result = appender()
        assert "Custom system message" in result
        assert "Server Mode" in result

    def test_system_appender_without_user_message(self):
        appender = _build_system_appender(None)
        result = appender()
        assert "Server Mode" in result
        assert "Do NOT use Ask" in result


# ---------------------------------------------------------------------------
# HTTP-level tests using aiohttp test client
# ---------------------------------------------------------------------------


class TestHealthEndpoint(AioHTTPTestCase):
    async def get_application(self):
        return _make_app()

    async def test_health_returns_status(self):
        resp = await self.client.request("GET", "/health")
        assert resp.status == 200
        data = await resp.json()
        assert data["status"] == "ok"
        assert data["uptime"] == 42
        assert data["tasks_run"] == 3
        assert data["platforms"] == ["slack"]


class TestModelsEndpoint(AioHTTPTestCase):
    async def get_application(self):
        return _make_app()

    async def test_list_models(self):
        resp = await self.client.request("GET", "/v1/models")
        assert resp.status == 200
        data = await resp.json()
        assert len(data["data"]) == 2
        model_ids = [m["id"] for m in data["data"]]
        assert "qwen" in model_ids


class TestAuthEndpoints(AioHTTPTestCase):
    async def get_application(self):
        return _make_app(api_key="secret")

    async def test_models_unauthorized_without_key(self):
        resp = await self.client.request("GET", "/v1/models")
        assert resp.status == 401

    async def test_models_unauthorized_wrong_key(self):
        resp = await self.client.request("GET", "/v1/models", headers={"Authorization": "Bearer wrong"})
        assert resp.status == 401

    async def test_models_authorized(self):
        resp = await self.client.request("GET", "/v1/models", headers={"Authorization": "Bearer secret"})
        assert resp.status == 200

    async def test_health_no_auth_required(self):
        resp = await self.client.request("GET", "/health")
        assert resp.status == 200

    async def test_chat_unauthorized(self):
        resp = await self.client.request(
            "POST",
            "/v1/chat/completions",
            json={"model": "qwen", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status == 401


class TestChatCompletionsEndpoint(AioHTTPTestCase):
    async def get_application(self):
        return _make_app()

    async def test_invalid_json(self):
        resp = await self.client.request(
            "POST",
            "/v1/chat/completions",
            data=b"not json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status == 400

    async def test_missing_messages(self):
        resp = await self.client.request(
            "POST",
            "/v1/chat/completions",
            json={"model": "qwen"},
        )
        assert resp.status == 400

    async def test_unknown_model(self):
        resp = await self.client.request(
            "POST",
            "/v1/chat/completions",
            json={"model": "nonexistent", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status == 404
