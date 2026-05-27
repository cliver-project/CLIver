import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest

from cliver.llm.media_generation import get_image_helper
from cliver.llm.media_generation.minimax import MiniMaxImageHelper
from cliver.llm.media_generation.openai_compat import OpenAIImageHelper
from cliver.media import MediaContent, MediaType


class TestHelperRegistry:
    def test_minimax_url_returns_minimax_helper(self):
        helper = get_image_helper("https://api.minimaxi.com/v1/image_generation")
        assert isinstance(helper, MiniMaxImageHelper)

    def test_minimax_url_variant(self):
        helper = get_image_helper("https://api.minimax.io/v1/image_generation")
        assert isinstance(helper, MiniMaxImageHelper)

    def test_unknown_url_returns_openai_helper(self):
        helper = get_image_helper("https://api.openai.com/v1/images/generations")
        assert isinstance(helper, OpenAIImageHelper)

    def test_other_url_returns_openai_fallback(self):
        helper = get_image_helper("https://ark.cn-beijing.volces.com/api/v3/images/generations")
        assert isinstance(helper, OpenAIImageHelper)


class TestMiniMaxImageHelper:
    def test_build_request_default(self):
        helper = MiniMaxImageHelper()
        body = helper.build_request("a sunset over mountains")
        assert body["prompt"] == "a sunset over mountains"
        assert body["model"] == "image-01"
        assert body["response_format"] == "base64"

    def test_build_request_custom_model(self):
        helper = MiniMaxImageHelper()
        body = helper.build_request("a cat", model_name="image-02")
        assert body["model"] == "image-02"

    def test_build_request_base64_format(self):
        helper = MiniMaxImageHelper()
        body = helper.build_request("a cat", response_format="base64")
        assert body["response_format"] == "base64"

    def test_build_request_with_params(self):
        helper = MiniMaxImageHelper()
        body = helper.build_request("a cat", aspect_ratio="16:9", n=3)
        assert body["aspect_ratio"] == "16:9"
        assert body["n"] == 3

    def test_parse_response_url_format(self):
        helper = MiniMaxImageHelper()
        response = {
            "base_resp": {"status_code": 0, "status_msg": "success"},
            "data": {"image_urls": ["https://cdn.minimax.com/img1.png", "https://cdn.minimax.com/img2.png"]},
        }
        result = helper.parse_response(response)
        assert len(result) == 2
        assert all(isinstance(m, MediaContent) for m in result)
        assert all(m.type == MediaType.IMAGE for m in result)
        assert result[0].data == "https://cdn.minimax.com/img1.png"
        assert result[0].source == "minimax_image_generation"

    def test_parse_response_base64_format(self):
        helper = MiniMaxImageHelper()
        response = {
            "base_resp": {"status_code": 0, "status_msg": "success"},
            "data": {"image_base64": ["aW1nMQ=="]},
        }
        result = helper.parse_response(response)
        assert len(result) == 1
        assert result[0].data == "aW1nMQ=="

    def test_parse_response_api_error(self):
        helper = MiniMaxImageHelper()
        response = {
            "base_resp": {"status_code": 1001, "status_msg": "invalid prompt"},
            "data": {},
        }
        with pytest.raises(RuntimeError, match="invalid prompt"):
            helper.parse_response(response)


class TestOpenAIImageHelper:
    def test_build_request_default(self):
        helper = OpenAIImageHelper()
        body = helper.build_request("a cat")
        assert body["prompt"] == "a cat"
        assert body["model"] == "dall-e-3"
        assert body["response_format"] == "url"

    def test_build_request_with_params(self):
        helper = OpenAIImageHelper()
        body = helper.build_request("a cat", size="1024x1024", quality="hd")
        assert body["size"] == "1024x1024"
        assert body["quality"] == "hd"

    def test_parse_response_url(self):
        helper = OpenAIImageHelper()
        response = {"data": [{"url": "https://oai.com/img1.png"}, {"url": "https://oai.com/img2.png"}]}
        result = helper.parse_response(response)
        assert len(result) == 2
        assert result[0].data == "https://oai.com/img1.png"
        assert result[0].source == "openai_image_generation"

    def test_parse_response_b64(self):
        helper = OpenAIImageHelper()
        response = {"data": [{"b64_json": "aW1n"}]}
        result = helper.parse_response(response)
        assert result[0].data == "aW1n"


class TestAgentCoreGenerateImage:
    """Tests for AgentCore.generate_image() with the new model-centric API.

    Image generation is now handled by ModelConfig objects with
    category='image' and api_url set on the model, rather than via
    ProviderConfig.image_url / image_model (which were removed).
    """

    def _make_image_model(self, api_model_name="image-01", api_url="https://api.minimaxi.com/v1/image_generation"):
        from cliver.config import ModelConfig, ProviderConfig

        prov = ProviderConfig(
            name="minimax",
            type="openai",
            api_url="http://x",
            api_key="sk-test",
        )
        mc = ModelConfig(
            name="minimax/image",
            provider="minimax",
            model=api_model_name,
            category="image",
            api_url=api_url,
        )
        mc._provider_config = prov
        return mc

    @pytest.mark.asyncio
    async def test_generate_with_explicit_model(self):
        from langchain_core.messages import AIMessage

        from cliver.llm.call_context import CallContext
        from cliver.llm.llm import AgentCore

        image_model = self._make_image_model()
        executor = AgentCore(llm_models={"minimax/image": image_model}, mcp_servers={})

        mock_api_response = {
            "base_resp": {"status_code": 0, "status_msg": "success"},
            "data": {"image_urls": ["https://img.png"]},
        }
        with patch.object(executor, "_call_generation_api", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = mock_api_response
            ctx = CallContext()
            result = await executor.generate_image("a sunset", model="minimax/image", ctx=ctx)

            assert isinstance(result, AIMessage)
            assert "Generated 1 image" in result.content
            assert len(ctx.generated_media) == 1
            mock_call.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_uses_image_model_name(self):
        """The API model name (ModelConfig.model) is used in the request body."""
        from cliver.llm.call_context import CallContext
        from cliver.llm.llm import AgentCore

        image_model = self._make_image_model(api_model_name="image-02")
        executor = AgentCore(llm_models={"minimax/image": image_model}, mcp_servers={})

        mock_api_response = {
            "base_resp": {"status_code": 0, "status_msg": "success"},
            "data": {"image_urls": ["https://img.png"]},
        }
        with patch.object(executor, "_call_generation_api", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = mock_api_response
            await executor.generate_image("a cat", ctx=CallContext())

            request_body = mock_call.call_args[0][2]
            assert request_body["model"] == "image-02"

    @pytest.mark.asyncio
    async def test_generate_discovers_image_model(self):
        """Auto-discovery of image model when none is specified explicitly."""
        from langchain_core.messages import AIMessage

        from cliver.llm.call_context import CallContext
        from cliver.llm.llm import AgentCore

        image_model = self._make_image_model()
        executor = AgentCore(llm_models={"minimax/image": image_model}, mcp_servers={})

        mock_api_response = {
            "base_resp": {"status_code": 0, "status_msg": "success"},
            "data": {"image_urls": ["https://img.png"]},
        }
        with patch.object(executor, "_call_generation_api", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = mock_api_response
            result = await executor.generate_image("a cat", ctx=CallContext())

            assert isinstance(result, AIMessage)
            mock_call.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_no_image_support(self):
        """Error returned when no image-category model is configured."""
        from langchain_core.messages import AIMessage

        from cliver.config import ModelConfig
        from cliver.llm.call_context import CallContext
        from cliver.llm.llm import AgentCore

        chat_model = ModelConfig(name="openai/chat", provider="openai", model="test")
        executor = AgentCore(llm_models={"openai/chat": chat_model}, mcp_servers={})

        result = await executor.generate_image("a cat", ctx=CallContext())
        assert isinstance(result, AIMessage)
        assert "error" in result.content.lower() or "no model" in result.content.lower()


class TestCapabilityRouting:
    @pytest.mark.asyncio
    async def test_all_models_use_react_loop(self):
        """All models go through the Re-Act loop. Image generation is handled
        by the ImageGenerateTool, not by capability-based routing."""
        from langchain_core.messages import AIMessage

        from cliver.config import ModelConfig, ProviderConfig
        from cliver.llm.llm import AgentCore

        prov = ProviderConfig(
            name="minimax",
            type="openai",
            api_url="http://x",
            api_key="sk-test",
        )
        model = ModelConfig(name="minimax/chat", provider="minimax", model="mm-2.7")
        model._provider_config = prov

        executor = AgentCore(llm_models={"minimax/chat": model}, mcp_servers={})

        mock_engine = Mock()
        mock_response = AIMessage(content="Here is your image")
        mock_engine.infer = AsyncMock(return_value=mock_response)
        mock_engine.parse_tool_calls.return_value = None
        mock_engine.system_message.return_value = "system"
        mock_engine.supports_capability.return_value = False
        mock_engine.config = model
        executor.llm_engines["minimax/chat"] = mock_engine

        with patch.object(executor, "_prepare_messages_and_tools", new_callable=AsyncMock) as mock_prep:
            mock_prep.return_value = (mock_engine, [], [])
            await asyncio.to_thread(executor.process_user_input, "describe a sunset", model="minimax/chat")
            mock_engine.infer.assert_called_once()
