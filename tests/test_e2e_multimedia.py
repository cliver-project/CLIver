"""
End-to-end test for multimedia support in CLIver.
"""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk

from cliver.config import ModelConfig
from cliver.llm.llm import AgentCore
from cliver.llm.unified_engine import UnifiedInferenceEngine
from cliver.media import MediaContent, MediaType
from cliver.model_capabilities import ModelCapability


class TestE2EMultimedia:
    """End-to-end tests for multimedia support."""

    @patch("cliver.llm.unified_engine.init_chat_model")
    def test_process_multimedia_user_input_with_image(self, mock_init):
        """Test processing user input with an image."""
        mock_init.return_value = AsyncMock()

        config = Mock(spec=ModelConfig)
        config.name = "openai/gpt-4-vision"
        config.provider = "openai"
        config.api_model_name = "gpt-4-vision"
        config.get_api_key = Mock(return_value="test-key")
        config.get_resolved_url = Mock(return_value="https://api.openai.com/v1")
        config.get_provider_type = Mock(return_value="openai")
        config.options = None

        config.get_capabilities = Mock(
            return_value={
                ModelCapability.TEXT_TO_TEXT,
                ModelCapability.IMAGE_TO_TEXT,
                ModelCapability.TOOL_CALLING,
            }
        )

        engine = UnifiedInferenceEngine(config)
        engine.llm = AsyncMock()

        mock_response = AIMessage(content="The image shows a beautiful landscape with mountains and a lake.")
        engine.infer = AsyncMock(return_value=mock_response)

        llm_models = {"openai/gpt-4-vision": config}
        mcp_servers = {}
        executor = AgentCore(llm_models, mcp_servers, config)
        executor.llm_engines["openai/gpt-4-vision"] = engine

        image_files = ["test_image.jpg"]

        with patch("cliver.media.load_media_file") as mock_load_media:
            mock_load_media.return_value = MediaContent(
                type=MediaType.IMAGE,
                data="base64image",
                mime_type="image/jpeg",
                filename="test_image.jpg",
            )

            response = asyncio.run(
                executor.process_user_input(
                    user_input="What's in this image?",
                    images=image_files,
                    model="openai/gpt-4-vision",
                )
            )

            assert response.content == "The image shows a beautiful landscape with mountains and a lake."
            engine.infer.assert_called_once()

    @patch("cliver.llm.unified_engine.init_chat_model")
    def test_stream_multimedia_user_input_with_image(self, mock_init):
        """Test streaming user input with an image."""
        mock_init.return_value = AsyncMock()

        config = Mock(spec=ModelConfig)
        config.name = "openai/gpt-4-vision"
        config.provider = "openai"
        config.api_model_name = "gpt-4-vision"
        config.get_api_key = Mock(return_value="test-key")
        config.get_resolved_url = Mock(return_value="https://api.openai.com/v1")
        config.get_provider_type = Mock(return_value="openai")
        config.options = None

        config.get_capabilities = Mock(
            return_value={
                ModelCapability.TEXT_TO_TEXT,
                ModelCapability.IMAGE_TO_TEXT,
                ModelCapability.TOOL_CALLING,
            }
        )

        engine = UnifiedInferenceEngine(config)
        engine.llm = AsyncMock()

        mock_chunks = [
            AIMessageChunk(content="The image shows "),
            AIMessageChunk(content="a beautiful landscape"),
            AIMessageChunk(content=" with mountains and a lake."),
        ]

        async def mock_stream(messages, tools, *, callbacks=None, **kwargs):
            if kwargs is None:
                kwargs = {}
            for chunk in mock_chunks:
                yield chunk

        engine.stream = mock_stream

        llm_models = {"openai/gpt-4-vision": config}
        mcp_servers = {}
        executor = AgentCore(llm_models, mcp_servers, config)
        executor.llm_engines["openai/gpt-4-vision"] = engine

        image_files = ["test_image.jpg"]

        with patch("cliver.media.load_media_file") as mock_load_media:
            mock_load_media.return_value = MediaContent(
                type=MediaType.IMAGE,
                data="base64image",
                mime_type="image/jpeg",
                filename="test_image.jpg",
            )

            chunks = []

            async def collect_chunks():
                async for chunk in executor.stream_user_input(
                    user_input="What's in this image?",
                    images=image_files,
                    model="openai/gpt-4-vision",
                ):
                    chunks.append(chunk)

            asyncio.run(collect_chunks())

            assert len(chunks) == 3
            assert chunks[0].content == "The image shows "
            assert chunks[1].content == "a beautiful landscape"
            assert chunks[2].content == " with mountains and a lake."


if __name__ == "__main__":
    pytest.main([__file__])
