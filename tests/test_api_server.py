"""Tests for OpenAI-compatible API server."""

from unittest.mock import MagicMock

from cliver.config import APIServerConfig, GatewayConfig


class TestAPIServerConfig:
    def test_defaults(self):
        cfg = APIServerConfig()
        assert cfg.enabled is False
        assert cfg.host == "0.0.0.0"
        assert cfg.port == 8000
        assert cfg.api_key is None

    def test_custom_config(self):
        cfg = APIServerConfig(enabled=True, port=9000, api_key="test-key")
        assert cfg.enabled is True
        assert cfg.port == 9000
        assert cfg.api_key == "test-key"

    def test_gateway_config_has_api_server(self):
        gw = GatewayConfig()
        assert gw.api_server is None

    def test_gateway_config_with_api_server(self):
        gw = GatewayConfig(api_server=APIServerConfig(enabled=True, port=8080))
        assert gw.api_server.enabled is True
        assert gw.api_server.port == 8080


class TestAPIServerEndpoints:
    """Test API server request handling logic (without starting HTTP server)."""

    def _make_server(self):
        from cliver.gateway.api_server import APIServer

        mock_executor = MagicMock()
        mock_executor.llm_models = {"qwen": MagicMock(), "deepseek": MagicMock()}
        mock_executor.default_model = "qwen"
        mock_executor.permission_manager = None
        config = APIServerConfig(enabled=True, port=8000)
        return APIServer(task_executor=mock_executor, config=config), mock_executor

    def test_list_models(self):
        server, _ = self._make_server()
        result = server._build_models_response()
        assert len(result["data"]) == 2
        model_ids = [m["id"] for m in result["data"]]
        assert "qwen" in model_ids

    def test_parse_chat_request_basic(self):
        server, _ = self._make_server()
        body = {"model": "qwen", "messages": [{"role": "user", "content": "hello"}]}
        parsed = server._parse_chat_request(body)
        assert parsed["user_input"] == "hello"
        assert parsed["model"] == "qwen"
        assert parsed["stream"] is False

    def test_parse_chat_request_with_history(self):
        server, _ = self._make_server()
        body = {
            "model": "qwen",
            "messages": [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "answer"},
                {"role": "user", "content": "second"},
            ],
        }
        parsed = server._parse_chat_request(body)
        assert parsed["user_input"] == "second"
        assert len(parsed["conversation_history"]) == 2
        assert parsed["system_message"] == "You are helpful"

    def test_parse_chat_request_stream(self):
        server, _ = self._make_server()
        body = {"model": "qwen", "messages": [{"role": "user", "content": "hello"}], "stream": True}
        parsed = server._parse_chat_request(body)
        assert parsed["stream"] is True

    def test_parse_chat_request_options(self):
        server, _ = self._make_server()
        body = {
            "model": "qwen",
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0.5,
            "max_tokens": 100,
        }
        parsed = server._parse_chat_request(body)
        assert parsed["options"]["temperature"] == 0.5
        assert parsed["options"]["max_tokens"] == 100

    def test_parse_chat_request_default_model(self):
        server, _ = self._make_server()
        body = {"messages": [{"role": "user", "content": "hello"}]}
        parsed = server._parse_chat_request(body)
        assert parsed["model"] == "qwen"

    def test_build_completion_response(self):
        server, _ = self._make_server()
        result = server._build_completion_response("chatcmpl-123", "qwen", "Hello!", 10, 5)
        assert result["model"] == "qwen"
        assert result["choices"][0]["message"]["content"] == "Hello!"
        assert result["choices"][0]["finish_reason"] == "stop"
        assert result["usage"]["total_tokens"] == 15

    def test_build_stream_chunk(self):
        server, _ = self._make_server()
        chunk = server._build_stream_chunk("chatcmpl-123", "qwen", "Hello")
        assert chunk["choices"][0]["delta"]["content"] == "Hello"
        assert chunk["choices"][0]["finish_reason"] is None

    def test_auth_no_key_configured(self):
        server, _ = self._make_server()
        assert server._check_auth(None) is True
        assert server._check_auth("anything") is True

    def test_auth_with_key(self):
        from cliver.gateway.api_server import APIServer

        config = APIServerConfig(enabled=True, api_key="secret")
        server = APIServer(task_executor=MagicMock(), config=config)
        assert server._check_auth("Bearer secret") is True
        assert server._check_auth("Bearer wrong") is False
        assert server._check_auth(None) is False


class TestServerMode:
    """Test server mode configuration."""

    def test_configure_server_mode_with_permission_manager(self):
        from cliver.gateway.api_server import APIServer
        from cliver.permissions import PermissionMode

        mock_executor = MagicMock()
        mock_pm = MagicMock()
        mock_executor.permission_manager = mock_pm
        config = APIServerConfig(enabled=True)
        server = APIServer(task_executor=mock_executor, config=config)

        server._configure_server_mode()
        mock_pm.set_mode.assert_called_once_with(PermissionMode.YOLO)
        # on_permission_prompt should be set to auto-allow
        assert mock_executor.on_permission_prompt is not None
        assert mock_executor.on_permission_prompt("any_tool", {}) == "allow"

    def test_configure_server_mode_without_permission_manager(self):
        from cliver.gateway.api_server import APIServer

        mock_executor = MagicMock()
        mock_executor.permission_manager = None
        config = APIServerConfig(enabled=True)
        server = APIServer(task_executor=mock_executor, config=config)

        # Should not raise
        server._configure_server_mode()
        assert mock_executor.on_permission_prompt is not None

    def test_system_appender_with_user_message(self):
        from cliver.gateway.api_server import APIServer

        config = APIServerConfig(enabled=True)
        server = APIServer(task_executor=MagicMock(), config=config)
        appender = server._build_system_appender("Custom system message")
        result = appender()
        assert "Custom system message" in result
        assert "Server Mode" in result

    def test_system_appender_without_user_message(self):
        from cliver.gateway.api_server import APIServer

        config = APIServerConfig(enabled=True)
        server = APIServer(task_executor=MagicMock(), config=config)
        appender = server._build_system_appender(None)
        result = appender()
        assert "Server Mode" in result
        assert "Do NOT use Ask" in result
