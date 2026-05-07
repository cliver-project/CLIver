"""
Test module for LLM-specific media extraction in CLIver.
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from langchain_core.messages import AIMessage

from cliver.config import ModelConfig
from cliver.llm.unified_engine import UnifiedInferenceEngine
from cliver.media import MediaType
from cliver.model_capabilities import ModelCapability


class TestLLMMediaExtraction:
    """Test LLM-specific media extraction functionality."""

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

        config.get_capabilities = Mock(
            return_value={
                ModelCapability.TEXT_TO_TEXT,
                ModelCapability.IMAGE_TO_TEXT,
                ModelCapability.TOOL_CALLING,
            }
        )

        return UnifiedInferenceEngine(config)

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
        config.get_provider_type = Mock(return_value="ollama")
        config.options = None
        config.model_dump = Mock(return_value={})

        config.get_capabilities = Mock(
            return_value={
                ModelCapability.TEXT_TO_TEXT,
                ModelCapability.IMAGE_TO_TEXT,
                ModelCapability.TOOL_CALLING,
            }
        )

        return UnifiedInferenceEngine(config)

    def test_openai_extract_media_from_data_url(self, openai_engine):
        """Test OpenAI engine extracts media from data URL."""
        content = "Here's an image: data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        response = AIMessage(content=content)

        media_content = openai_engine.extract_media_from_response(response)
        assert len(media_content) == 1
        assert media_content[0].type == MediaType.IMAGE
        assert media_content[0].mime_type == "image/png"
        assert "openai_generated" in media_content[0].filename

    def test_ollama_extract_media_from_data_url(self, ollama_engine):
        """Test Ollama engine extracts media from data URL."""
        content = "Here's an image: data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAAMCAgMCAgMDAwMEAwMEBQgFBAQE"
        response = AIMessage(content=content)

        media_content = ollama_engine.extract_media_from_response(response)
        assert len(media_content) == 1
        assert media_content[0].type == MediaType.IMAGE
        assert media_content[0].mime_type == "image/jpeg"
        assert "ollama_generated" in media_content[0].filename

    def test_engines_inherit_extract_method(self, openai_engine, ollama_engine):
        """Test that both engines inherit the extract_media_from_response method."""
        assert hasattr(openai_engine, "extract_media_from_response")
        assert hasattr(ollama_engine, "extract_media_from_response")

        empty_response = AIMessage(content="")
        openai_media = openai_engine.extract_media_from_response(empty_response)
        ollama_media = ollama_engine.extract_media_from_response(empty_response)

        assert openai_media == []
        assert ollama_media == []


if __name__ == "__main__":
    pytest.main([__file__])
