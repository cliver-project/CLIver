"""
Test module for multi-media support in CLIver.
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from langchain_core.messages import HumanMessage

from cliver.config import ModelConfig
from cliver.llm.llm import AgentCore
from cliver.llm.unified_engine import UnifiedInferenceEngine
from cliver.media import MediaContent, MediaType, load_media_file


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
    @patch("cliver.llm.unified_engine.init_chat_model")
    def openai_engine(self, mock_init):
        """Create a mock OpenAI engine."""
        mock_init.return_value = AsyncMock()

        config = Mock(spec=ModelConfig)
        config.name = "openai/gpt-4-vision"
        config.provider = "openai"
        config.api_model_name = "gpt-4-vision"
        config.get_api_key = Mock(return_value="test-key")
        config.get_resolved_url = Mock(return_value="https://api.openai.com/v1")
        config.get_provider_type = Mock(return_value="openai")
        config.options = None

        engine = UnifiedInferenceEngine(config)
        engine.llm = AsyncMock()
        return engine

    def test_convert_messages_preserves_multimodal(self, openai_engine):
        """Multimodal content (already in OpenAI format) passes through."""
        message = HumanMessage(
            content=[
                {"type": "text", "text": "What's in this image?"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,base64image"}},
            ]
        )

        converted = openai_engine.convert_messages_to_engine_specific([message])

        assert len(converted) == 1
        converted_message = converted[0]
        assert isinstance(converted_message.content, list)
        assert len(converted_message.content) == 2
        assert converted_message.content[0]["type"] == "text"
        assert converted_message.content[1]["type"] == "image_url"


class TestOllamaEngine:
    """Test Ollama engine multimedia support."""

    @pytest.fixture
    @patch("cliver.llm.unified_engine.init_chat_model")
    def ollama_engine(self, mock_init):
        """Create a mock Ollama engine."""
        mock_init.return_value = AsyncMock()

        config = Mock(spec=ModelConfig)
        config.name = "ollama/llava"
        config.provider = "ollama"
        config.api_model_name = "llava"
        config.get_resolved_url = Mock(return_value="http://localhost:11434")
        config.get_provider_type = Mock(return_value="openai")
        config.options = None
        config.model_dump = Mock(return_value={})

        engine = UnifiedInferenceEngine(config)
        engine.llm = AsyncMock()
        return engine

    def test_convert_messages_pass_through(self, ollama_engine):
        """Messages pass through unchanged (no provider-specific conversion)."""
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
        # Media is consumed during _prepare_user_messages, not during
        # convert_messages_to_engine_specific — the message passes through.
        assert "images" not in converted_message.additional_kwargs


class TestAgentCore:
    """Test AgentCore multi-media support."""

    @pytest.fixture
    def agent_core(self):
        """Create a mock AgentCore."""
        llm_models = {}
        mcp_servers = {}
        executor = AgentCore(llm_models, mcp_servers)
        return executor

    def test_process_user_input_accepts_specific_media_types(self, agent_core):
        """Test that process_user_input method accepts specific media type parameters."""
        import inspect

        sig = inspect.signature(agent_core.process_user_input)
        assert "images" in sig.parameters
        assert "audio_files" in sig.parameters
        assert "video_files" in sig.parameters

    def test_stream_user_input_accepts_specific_media_types(self, agent_core):
        """Test that stream_user_input method accepts specific media type parameters."""
        import inspect

        sig = inspect.signature(agent_core.stream_user_input)
        assert "images" in sig.parameters
        assert "audio_files" in sig.parameters
        assert "video_files" in sig.parameters

    def test_prepare_messages_and_tools_accepts_specific_media_types(self, agent_core):
        """Test that _prepare_messages_and_tools accepts specific media type parameters."""
        import inspect

        sig = inspect.signature(agent_core._prepare_messages_and_tools)
        assert "images" in sig.parameters
        assert "audio_files" in sig.parameters
        assert "video_files" in sig.parameters


if __name__ == "__main__":
    pytest.main([__file__])
