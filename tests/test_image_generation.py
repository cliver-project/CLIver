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
        assert body["response_format"] == "b64_json"

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
    def _make_provider(self, image_url="https://api.minimaxi.com/v1/image_generation", image_model="image-01"):
        from cliver.config import ProviderConfig

        return ProviderConfig(
            name="mm",
            type="openai",
            api_url="http://x",
            api_key="sk-test",
            image_url=image_url,
            image_model=image_model,
        )

    def _make_chat_model(self, provider_name="mm"):
        from cliver.config import ModelConfig

        mc = ModelConfig(name=f"{provider_name}/chat", provider=provider_name)
        return mc

    @pytest.mark.asyncio
    async def test_generate_with_explicit_model(self):
        from langchain_core.messages import AIMessage

        from cliver.llm.call_context import CallContext
        from cliver.llm.llm import AgentCore

        prov = self._make_provider()
        chat_model = self._make_chat_model()
        chat_model._provider_config = prov

        executor = AgentCore(llm_models={"mm/chat": chat_model}, mcp_servers={})

        mock_api_response = {
            "base_resp": {"status_code": 0, "status_msg": "success"},
            "data": {"image_urls": ["https://img.png"]},
        }
        with patch.object(executor, "_call_generation_api", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = mock_api_response
            result = await executor.generate_image("a sunset", model="mm/chat", ctx=CallContext())

            assert isinstance(result, AIMessage)
            assert "media_content" in result.additional_kwargs
            assert len(result.additional_kwargs["media_content"]) == 1
            mock_call.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_uses_provider_image_model(self):
        from cliver.llm.call_context import CallContext
        from cliver.llm.llm import AgentCore

        prov = self._make_provider(image_model="image-02")
        chat_model = self._make_chat_model()
        chat_model._provider_config = prov

        executor = AgentCore(llm_models={"mm/chat": chat_model}, mcp_servers={})

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
    async def test_generate_discovers_provider_with_image_url(self):
        from langchain_core.messages import AIMessage

        from cliver.llm.call_context import CallContext
        from cliver.llm.llm import AgentCore

        prov = self._make_provider()
        chat_model = self._make_chat_model()
        chat_model._provider_config = prov

        executor = AgentCore(llm_models={"mm/chat": chat_model}, mcp_servers={})

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
        from langchain_core.messages import AIMessage

        from cliver.config import ModelConfig
        from cliver.llm.call_context import CallContext
        from cliver.llm.llm import AgentCore

        chat_model = ModelConfig(name="openai/chat", provider="openai")
        executor = AgentCore(llm_models={"openai/chat": chat_model}, mcp_servers={})

        result = await executor.generate_image("a cat", ctx=CallContext())
        assert isinstance(result, AIMessage)
        assert "error" in result.content.lower() or "no model" in result.content.lower()


class TestCapabilityRouting:
    @pytest.mark.asyncio
    async def test_text_to_image_only_model_routes_to_generate(self):
        from langchain_core.messages import AIMessage

        from cliver.config import ModelConfig, ProviderConfig
        from cliver.llm.llm import AgentCore
        from cliver.model_capabilities import ModelCapability

        prov = ProviderConfig(
            name="mm",
            type="openai",
            api_url="http://x",
            api_key="sk-test",
            image_url="https://api.minimaxi.com/v1/image_generation",
            image_model="image-01",
        )
        img_model = ModelConfig(name="mm/image-01", provider="mm")
        img_model.capabilities = {ModelCapability.TEXT_TO_IMAGE}
        img_model._provider_config = prov

        executor = AgentCore(llm_models={"mm/image-01": img_model}, mcp_servers={})

        mock_api_response = {
            "base_resp": {"status_code": 0, "status_msg": "success"},
            "data": {"image_urls": ["https://img.png"]},
        }
        with patch.object(executor, "_call_generation_api", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = mock_api_response
            result = await executor.process_user_input("a sunset", model="mm/image-01")

            assert isinstance(result, AIMessage)
            assert "media_content" in result.additional_kwargs
            mock_call.assert_called_once()

    @pytest.mark.asyncio
    async def test_multimodal_model_uses_react_loop(self):
        from langchain_core.messages import AIMessage

        from cliver.config import ModelConfig, ProviderConfig
        from cliver.llm.llm import AgentCore
        from cliver.model_capabilities import ModelCapability

        prov = ProviderConfig(
            name="mm",
            type="openai",
            api_url="http://x",
            api_key="sk-test",
        )
        multi_model = ModelConfig(name="mm/mm-2.7", provider="mm")
        multi_model.capabilities = {
            ModelCapability.TEXT_TO_TEXT,
            ModelCapability.TEXT_TO_IMAGE,
            ModelCapability.TOOL_CALLING,
        }
        multi_model._provider_config = prov

        executor = AgentCore(llm_models={"mm/mm-2.7": multi_model}, mcp_servers={})

        mock_engine = Mock()
        mock_response = AIMessage(content="Here is your image")
        mock_engine.infer = AsyncMock(return_value=mock_response)
        mock_engine.parse_tool_calls.return_value = None
        mock_engine.system_message.return_value = "system"
        mock_engine.supports_capability.return_value = False
        mock_engine.config = multi_model
        executor.llm_engines["mm/mm-2.7"] = mock_engine

        with patch.object(executor, "_prepare_messages_and_tools", new_callable=AsyncMock) as mock_prep:
            mock_prep.return_value = (mock_engine, [], [])
            await executor.process_user_input("describe a sunset", model="mm/mm-2.7")
            mock_engine.infer.assert_called_once()
