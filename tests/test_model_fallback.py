"""Tests for smart model routing: error classification, fallback selection, and Re-Act loop integration."""

from click.testing import CliRunner

from cliver.llm.error_classifier import ErrorAction, classify_error
from cliver.model_capabilities import ModelCapability


class TestErrorClassification:
    """Error classifier categorizes exceptions into retry/failover/fatal."""

    def test_rate_limit_is_retry(self):
        from unittest.mock import MagicMock

        from openai import RateLimitError

        response = MagicMock()
        response.status_code = 429
        response.headers = {}
        err = RateLimitError("rate limit exceeded", response=response, body=None)
        result = classify_error(err)
        assert result.action == ErrorAction.RETRY
        assert result.reason == "rate_limit"
        assert result.should_compress is False

    def test_timeout_is_retry(self):
        from httpx import ReadTimeout

        err = ReadTimeout("read timed out")
        result = classify_error(err)
        assert result.action == ErrorAction.RETRY
        assert result.reason == "timeout"

    def test_connection_error_is_retry(self):
        err = ConnectionError("connection refused")
        result = classify_error(err)
        assert result.action == ErrorAction.RETRY
        assert result.reason == "connection_error"

    def test_server_error_is_retry(self):
        from unittest.mock import MagicMock

        from openai import InternalServerError

        response = MagicMock()
        response.status_code = 500
        response.headers = {}
        err = InternalServerError("internal server error", response=response, body=None)
        result = classify_error(err)
        assert result.action == ErrorAction.RETRY
        assert result.reason == "server_error"

    def test_auth_error_is_failover(self):
        from unittest.mock import MagicMock

        from openai import AuthenticationError

        response = MagicMock()
        response.status_code = 401
        response.headers = {}
        err = AuthenticationError("invalid api key", response=response, body=None)
        result = classify_error(err)
        assert result.action == ErrorAction.FAILOVER
        assert result.reason == "auth"
        assert result.should_compress is False

    def test_context_overflow_is_failover_with_compress(self):
        from unittest.mock import MagicMock

        from openai import BadRequestError

        response = MagicMock()
        response.status_code = 400
        response.headers = {}
        err = BadRequestError(
            "This model's maximum context length is 8192 tokens",
            response=response,
            body=None,
        )
        result = classify_error(err)
        assert result.action == ErrorAction.FAILOVER
        assert result.reason == "context_overflow"
        assert result.should_compress is True

    def test_model_not_found_is_failover(self):
        from unittest.mock import MagicMock

        from openai import NotFoundError

        response = MagicMock()
        response.status_code = 404
        response.headers = {}
        err = NotFoundError("model not found", response=response, body=None)
        result = classify_error(err)
        assert result.action == ErrorAction.FAILOVER
        assert result.reason == "model_not_found"

    def test_unknown_error_is_retry(self):
        err = RuntimeError("something unexpected")
        result = classify_error(err)
        assert result.action == ErrorAction.RETRY
        assert result.reason == "unknown"

    def test_classified_error_carries_original(self):
        err = RuntimeError("test")
        result = classify_error(err)
        assert result.original_error is err


class TestFallbackSelection:
    """Capability-based fallback model selection."""

    def _make_executor(self, models_dict):
        from unittest.mock import MagicMock

        from cliver.llm.llm import AgentCore

        mock_models = {}
        for name, caps in models_dict.items():
            config = MagicMock()
            config.name = name
            config.get_capabilities.return_value = caps
            mock_models[name] = config

        return AgentCore(llm_models=mock_models, mcp_servers={})

    def test_finds_model_with_matching_capabilities(self):
        executor = self._make_executor(
            {
                "qwen": {ModelCapability.TEXT_TO_TEXT, ModelCapability.TOOL_CALLING},
                "deepseek": {ModelCapability.TEXT_TO_TEXT, ModelCapability.TOOL_CALLING},
            }
        )
        result = executor._find_fallback_model(
            "qwen",
            {ModelCapability.TEXT_TO_TEXT, ModelCapability.TOOL_CALLING},
            {"qwen"},
        )
        assert result == "deepseek"

    def test_skips_models_without_required_capabilities(self):
        executor = self._make_executor(
            {
                "qwen": {ModelCapability.TEXT_TO_TEXT, ModelCapability.TOOL_CALLING},
                "simple": {ModelCapability.TEXT_TO_TEXT},
                "deepseek": {ModelCapability.TEXT_TO_TEXT, ModelCapability.TOOL_CALLING},
            }
        )
        result = executor._find_fallback_model(
            "qwen",
            {ModelCapability.TEXT_TO_TEXT, ModelCapability.TOOL_CALLING},
            {"qwen"},
        )
        assert result == "deepseek"

    def test_skips_already_tried_models(self):
        executor = self._make_executor(
            {
                "qwen": {ModelCapability.TEXT_TO_TEXT, ModelCapability.TOOL_CALLING},
                "deepseek": {ModelCapability.TEXT_TO_TEXT, ModelCapability.TOOL_CALLING},
                "gpt4": {ModelCapability.TEXT_TO_TEXT, ModelCapability.TOOL_CALLING},
            }
        )
        result = executor._find_fallback_model(
            "qwen",
            {ModelCapability.TEXT_TO_TEXT, ModelCapability.TOOL_CALLING},
            {"qwen", "deepseek"},
        )
        assert result == "gpt4"

    def test_returns_none_when_no_fallback_available(self):
        executor = self._make_executor(
            {
                "qwen": {ModelCapability.TEXT_TO_TEXT, ModelCapability.TOOL_CALLING},
            }
        )
        result = executor._find_fallback_model(
            "qwen",
            {ModelCapability.TEXT_TO_TEXT, ModelCapability.TOOL_CALLING},
            {"qwen"},
        )
        assert result is None

    def test_requires_image_capability_when_images_used(self):
        executor = self._make_executor(
            {
                "qwen": {ModelCapability.TEXT_TO_TEXT, ModelCapability.IMAGE_TO_TEXT},
                "deepseek": {ModelCapability.TEXT_TO_TEXT, ModelCapability.TOOL_CALLING},
                "llava": {ModelCapability.TEXT_TO_TEXT, ModelCapability.IMAGE_TO_TEXT},
            }
        )
        result = executor._find_fallback_model(
            "qwen",
            {ModelCapability.TEXT_TO_TEXT, ModelCapability.IMAGE_TO_TEXT},
            {"qwen"},
        )
        assert result == "llava"


class TestComputeRequiredCapabilities:
    def test_text_only(self):
        from cliver.llm.llm import AgentCore

        caps = AgentCore._compute_required_capabilities(
            images=None,
            audio_files=None,
            video_files=None,
            has_tools=False,
        )
        assert caps == {ModelCapability.TEXT_TO_TEXT}

    def test_with_images(self):
        from cliver.llm.llm import AgentCore

        caps = AgentCore._compute_required_capabilities(
            images=["img.png"],
            audio_files=None,
            video_files=None,
            has_tools=False,
        )
        assert ModelCapability.IMAGE_TO_TEXT in caps

    def test_with_tools(self):
        from cliver.llm.llm import AgentCore

        caps = AgentCore._compute_required_capabilities(
            images=None,
            audio_files=None,
            video_files=None,
            has_tools=True,
        )
        assert ModelCapability.TOOL_CALLING in caps

    def test_with_audio(self):
        from cliver.llm.llm import AgentCore

        caps = AgentCore._compute_required_capabilities(
            images=None,
            audio_files=["audio.mp3"],
            video_files=None,
            has_tools=False,
        )
        assert ModelCapability.AUDIO_TO_TEXT in caps

    def test_with_video(self):
        from cliver.llm.llm import AgentCore

        caps = AgentCore._compute_required_capabilities(
            images=None,
            audio_files=None,
            video_files=["vid.mp4"],
            has_tools=False,
        )
        assert ModelCapability.VIDEO_TO_TEXT in caps

    def test_combined(self):
        from cliver.llm.llm import AgentCore

        caps = AgentCore._compute_required_capabilities(
            images=["a.png"],
            audio_files=None,
            video_files=None,
            has_tools=True,
        )
        assert caps == {
            ModelCapability.TEXT_TO_TEXT,
            ModelCapability.IMAGE_TO_TEXT,
            ModelCapability.TOOL_CALLING,
        }


class TestReActLoopFallback:
    """Re-Act loop should retry, compress, and failover on errors."""

    def _make_executor_with_engines(self):
        from unittest.mock import AsyncMock, MagicMock

        from langchain_core.messages import AIMessage

        from cliver.llm.llm import AgentCore

        model_a = MagicMock()
        model_a.name = "model_a"
        model_a.get_capabilities.return_value = {
            ModelCapability.TEXT_TO_TEXT,
            ModelCapability.TOOL_CALLING,
        }
        model_a.context_window = 8192

        model_b = MagicMock()
        model_b.name = "model_b"
        model_b.get_capabilities.return_value = {
            ModelCapability.TEXT_TO_TEXT,
            ModelCapability.TOOL_CALLING,
        }
        model_b.context_window = 32768

        executor = AgentCore(
            llm_models={"model_a": model_a, "model_b": model_b},
            mcp_servers={},
        )

        engine_a = MagicMock()
        engine_a.parse_tool_calls = MagicMock(return_value=None)
        engine_b = MagicMock()
        engine_b.parse_tool_calls = MagicMock(return_value=None)
        engine_b.infer = AsyncMock(return_value=AIMessage(content="fallback answer"))

        executor.llm_engines = {"model_a": engine_a, "model_b": engine_b}

        return executor, engine_a, engine_b

    def test_failover_on_auth_error(self):
        import asyncio
        from unittest.mock import AsyncMock, MagicMock

        from langchain_core.messages import SystemMessage
        from openai import AuthenticationError

        executor, engine_a, engine_b = self._make_executor_with_engines()

        response = MagicMock()
        response.status_code = 401
        response.headers = {}
        engine_a.infer = AsyncMock(side_effect=AuthenticationError("bad key", response=response, body=None))

        messages = [SystemMessage(content="test")]
        result = asyncio.run(
            executor._process_messages(
                engine_a,
                "model_a",
                messages,
                50,
                0,
                [],
                None,
                None,
                options={},
                required_capabilities={
                    ModelCapability.TEXT_TO_TEXT,
                    ModelCapability.TOOL_CALLING,
                },
                auto_fallback=True,
                tried_models={"model_a"},
            )
        )
        assert result.content == "fallback answer"

    def test_no_fallback_when_disabled(self):
        import asyncio
        from unittest.mock import AsyncMock, MagicMock

        from langchain_core.messages import SystemMessage
        from openai import AuthenticationError

        executor, engine_a, engine_b = self._make_executor_with_engines()

        response = MagicMock()
        response.status_code = 401
        response.headers = {}
        engine_a.infer = AsyncMock(side_effect=AuthenticationError("bad key", response=response, body=None))

        messages = [SystemMessage(content="test")]
        result = asyncio.run(
            executor._process_messages(
                engine_a,
                "model_a",
                messages,
                50,
                0,
                [],
                None,
                None,
                options={},
                required_capabilities={ModelCapability.TEXT_TO_TEXT},
                auto_fallback=False,
                tried_models={"model_a"},
            )
        )
        assert "error" in result.content.lower() or "auth" in result.content.lower()

    def test_retry_on_rate_limit(self):
        import asyncio
        from unittest.mock import AsyncMock, MagicMock, patch

        from langchain_core.messages import AIMessage, SystemMessage
        from openai import RateLimitError

        executor, engine_a, engine_b = self._make_executor_with_engines()

        response_429 = MagicMock()
        response_429.status_code = 429
        response_429.headers = {}

        engine_a.infer = AsyncMock(
            side_effect=[
                RateLimitError("rate limited", response=response_429, body=None),
                RateLimitError("rate limited", response=response_429, body=None),
                AIMessage(content="success after retry"),
            ]
        )

        messages = [SystemMessage(content="test")]
        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = asyncio.run(
                executor._process_messages(
                    engine_a,
                    "model_a",
                    messages,
                    50,
                    0,
                    [],
                    None,
                    None,
                    options={},
                    required_capabilities={ModelCapability.TEXT_TO_TEXT},
                    auto_fallback=True,
                    tried_models={"model_a"},
                )
            )
        assert result.content == "success after retry"
        assert engine_a.infer.call_count == 3


class TestNoFallbackFlag:
    """--no-fallback CLI flag."""

    def test_no_fallback_flag_accepted(self, load_cliver, init_config):
        result = CliRunner().invoke(
            load_cliver,
            ["chat", "--no-fallback", "hello"],
            catch_exceptions=False,
        )
        assert "no such option: --no-fallback" not in result.output

    def test_no_fallback_on_top_level(self, load_cliver, init_config):
        result = CliRunner().invoke(
            load_cliver,
            ["-p", "hello", "--no-fallback"],
            catch_exceptions=False,
        )
        assert "no such option: --no-fallback" not in result.output


class TestModelFallbackIntegration:
    """Integration tests combining fallback with CLI flags."""

    def test_no_fallback_with_json_output(self, load_cliver, init_config):
        """--no-fallback + --output json should work together."""
        import json

        result = CliRunner().invoke(
            load_cliver,
            ["-p", "hello", "--output", "json", "--no-fallback"],
            catch_exceptions=False,
        )
        output = result.output.strip()
        data = json.loads(output)
        assert "success" in data

    def test_full_ci_with_no_fallback(self, load_cliver, init_config):
        """All CI flags + --no-fallback should work together."""
        import json

        result = CliRunner().invoke(
            load_cliver,
            [
                "-p", "hello",
                "--output", "json",
                "--timeout", "60",
                "--permission-mode", "yolo",
                "--no-fallback",
            ],
            catch_exceptions=False,
        )
        output = result.output.strip()
        data = json.loads(output)
        assert "success" in data
