"""
Test module for multi-media support in CLIver.
"""

import pytest
from unittest.mock import AsyncMock, Mock
from cliver.llm.openai_engine import OpenAICompatibleInferenceEngine
from cliver.llm.ollama_engine import OllamaLlamaInferenceEngine
from cliver.llm.llm import TaskExecutor
from cliver.media import MediaContent, MediaType, load_media_file
from cliver.config import ModelConfig
from cliver.model_capabilities import ModelCapability
from langchain_core.messages import HumanMessage


class TestMediaContent:
    """Test media content functionality."""

    def test_media_content_creation(self):
        """Test creating MediaContent objects."""
        media = MediaContent(
            type=MediaType.IMAGE,
            data="base64data",
            mime_type="image/jpeg",
            filename="test.jpg",
        )
        assert media.type == MediaType.IMAGE
        assert media.data == "base64data"
        assert media.mime_type == "image/jpeg"
        assert media.filename == "test.jpg"

    def test_load_media_file(self):
        """Test loading media files."""
        # Just test that the function exists and can be imported
        assert load_media_file is not None


class TestOpenAIEngine:
    """Test OpenAI engine multimedia support."""

    @pytest.fixture
    def openai_engine(self):
        """Create a mock OpenAI engine."""
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
        return engine

    def test_convert_messages_to_openai_format(self, openai_engine):
        """Test converting messages to OpenAI format."""
        # Create a human message with media content
        media = MediaContent(
            type=MediaType.IMAGE,
            data="base64image",
            mime_type="image/jpeg",
            filename="test.jpg",
        )

        message = HumanMessage(content="What's in this image?")
        message.media_content = [media]

        converted = openai_engine.convert_messages_to_engine_specific([message])

        assert len(converted) == 1
        converted_message = converted[0]
        assert isinstance(converted_message, HumanMessage)
        assert isinstance(converted_message.content, list)
        assert len(converted_message.content) == 2  # text + image
        assert converted_message.content[0]["type"] == "text"
        assert converted_message.content[1]["type"] == "image_url"


class TestOllamaEngine:
    """Test Ollama engine multimedia support."""

    @pytest.fixture
    def ollama_engine(self):
        """Create a mock Ollama engine."""
        config = Mock(spec=ModelConfig)
        config.name = "llava"
        config.provider = "ollama"
        config.name_in_provider = "llava"
        config.url = "http://localhost:11434"
        config.options = None  # Add the missing options attribute
        config.model_dump = Mock(return_value={})

        # Mock capabilities
        config.get_capabilities = Mock(
            return_value={
                ModelCapability.TEXT_TO_TEXT,
                ModelCapability.IMAGE_TO_TEXT,
                ModelCapability.TOOL_CALLING,
            }
        )

        engine = OllamaLlamaInferenceEngine(config)
        engine.llm = AsyncMock()
        return engine

    def test_convert_messages_to_ollama_format(self, ollama_engine):
        """Test converting messages to Ollama format."""
        # Create a human message with media content
        media = MediaContent(
            type=MediaType.IMAGE,
            data="base64image",
            mime_type="image/jpeg",
            filename="test.jpg",
        )

        message = HumanMessage(content="What's in this image?")
        message.media_content = [media]

        converted = ollama_engine.convert_messages_to_engine_specific([message])

        assert len(converted) == 1
        converted_message = converted[0]
        assert isinstance(converted_message, HumanMessage)
        assert converted_message.content == "What's in this image?"
        assert "images" in converted_message.additional_kwargs
        assert converted_message.additional_kwargs["images"] == ["base64image"]


class TestTaskExecutor:
    """Test TaskExecutor multi-media support."""

    @pytest.fixture
    def task_executor(self):
        """Create a mock TaskExecutor."""
        llm_models = {}
        mcp_servers = {}
        executor = TaskExecutor(llm_models, mcp_servers)
        return executor

    def test_process_user_input_accepts_specific_media_types(self, task_executor):
        """Test that process_user_input method accepts specific media type parameters."""
        # Check that the method signature includes specific media type parameters
        import inspect

        sig = inspect.signature(task_executor.process_user_input)
        assert "images" in sig.parameters
        assert "audio_files" in sig.parameters
        assert "video_files" in sig.parameters

    def test_stream_user_input_accepts_specific_media_types(self, task_executor):
        """Test that stream_user_input method accepts specific media type parameters."""
        # Check that the method signature includes specific media type parameters
        import inspect

        sig = inspect.signature(task_executor.stream_user_input)
        assert "images" in sig.parameters
        assert "audio_files" in sig.parameters
        assert "video_files" in sig.parameters

    def test_prepare_messages_and_tools_accepts_specific_media_types(self, task_executor):
        """Test that _prepare_messages_and_tools accepts specific media type parameters."""
        # Check that the method signature includes specific media type parameters
        import inspect

        sig = inspect.signature(task_executor._prepare_messages_and_tools)
        assert "images" in sig.parameters
        assert "audio_files" in sig.parameters
        assert "video_files" in sig.parameters


if __name__ == "__main__":
    pytest.main([__file__])
