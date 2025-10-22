"""
End-to-end test for multimedia support in CLIver.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, Mock, patch
from cliver.llm.openai_engine import OpenAICompatibleInferenceEngine
from cliver.llm.llm import TaskExecutor
from cliver.media import MediaContent, MediaType
from cliver.config import ModelConfig
from cliver.model_capabilities import ModelCapability
from langchain_core.messages import AIMessage, AIMessageChunk


class TestE2EMultimedia:
    """End-to-end tests for multimedia support."""

    def test_process_multimedia_user_input_with_image(self):
        """Test processing user input with an image."""
        # Create a mock OpenAI engine
        config = Mock(spec=ModelConfig)
        config.name = "gpt-4-vision"
        config.provider = "openai"
        config.name_in_provider = "gpt-4-vision"
        config.url = "https://api.openai.com/v1"
        config.api_key = "test-key"
        config.options = None

        # Mock capabilities
        config.get_capabilities = Mock(
            return_value={
                ModelCapability.TEXT_TO_TEXT,
                ModelCapability.IMAGE_TO_TEXT,
                ModelCapability.TOOL_CALLING,
            }
        )

        engine = OpenAICompatibleInferenceEngine(config)
        engine.llm = AsyncMock()

        # Mock the OpenAI response
        mock_response = AIMessage(
            content="The image shows a beautiful landscape with mountains and a lake."
        )
        # We're using the regular infer method now since we've enhanced it to handle multimedia
        engine.infer = AsyncMock(return_value=mock_response)

        # Create a TaskExecutor with mock OpenAI engine
        llm_models = {"gpt-4-vision": config}
        mcp_servers = {}
        executor = TaskExecutor(llm_models, mcp_servers, config)
        executor.llm_engines["gpt-4-vision"] = engine

        # Create mock media content
        image_files = ["test_image.jpg"]

        with patch("cliver.media.load_media_file") as mock_load_media:
            mock_load_media.return_value = MediaContent(
                type=MediaType.IMAGE,
                data="base64image",
                mime_type="image/jpeg",
                filename="test_image.jpg",
            )

            # Process multimedia input (run async function in event loop)
            response = asyncio.run(
                executor.process_user_input(
                    user_input="What's in this image?",
                    images=image_files,
                    model="gpt-4-vision",
                )
            )

            # Verify the response
            assert (
                response.content
                == "The image shows a beautiful landscape with mountains and a lake."
            )
            # Verify that infer was called
            engine.infer.assert_called_once()

    def test_stream_multimedia_user_input_with_image(self):
        """Test streaming user input with an image."""
        # Create a mock OpenAI engine
        config = Mock(spec=ModelConfig)
        config.name = "gpt-4-vision"
        config.provider = "openai"
        config.name_in_provider = "gpt-4-vision"
        config.url = "https://api.openai.com/v1"
        config.api_key = "test-key"
        config.options = None

        # Mock capabilities
        config.get_capabilities = Mock(
            return_value={
                ModelCapability.TEXT_TO_TEXT,
                ModelCapability.IMAGE_TO_TEXT,
                ModelCapability.TOOL_CALLING,
            }
        )

        engine = OpenAICompatibleInferenceEngine(config)
        engine.llm = AsyncMock()

        # Mock the OpenAI streaming response
        mock_chunks = [
            AIMessageChunk(content="The image shows "),
            AIMessageChunk(content="a beautiful landscape"),
            AIMessageChunk(content=" with mountains and a lake."),
        ]

        async def mock_stream(messages, tools, *, callbacks=None, **kwargs):
            # Handle the case where kwargs might be None or empty
            if kwargs is None:
                kwargs = {}
            for chunk in mock_chunks:
                yield chunk

        # We're using the regular stream method now since we've enhanced it to handle multimedia
        engine.stream = mock_stream

        # Create a TaskExecutor with mock OpenAI engine
        llm_models = {"gpt-4-vision": config}
        mcp_servers = {}
        executor = TaskExecutor(llm_models, mcp_servers, config)
        executor.llm_engines["gpt-4-vision"] = engine

        # Create mock media content
        image_files = ["test_image.jpg"]

        with patch("cliver.media.load_media_file") as mock_load_media:
            mock_load_media.return_value = MediaContent(
                type=MediaType.IMAGE,
                data="base64image",
                mime_type="image/jpeg",
                filename="test_image.jpg",
            )

            # Stream multimedia input (run async function in event loop)
            chunks = []

            async def collect_chunks():
                async for chunk in executor.stream_user_input(
                    user_input="What's in this image?",
                    images=image_files,
                    model="gpt-4-vision",
                ):
                    chunks.append(chunk)

            asyncio.run(collect_chunks())

            # Verify we got the expected chunks
            assert len(chunks) == 3
            assert chunks[0].content == "The image shows "
            assert chunks[1].content == "a beautiful landscape"
            assert chunks[2].content == " with mountains and a lake."


if __name__ == "__main__":
    pytest.main([__file__])
