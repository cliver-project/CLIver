"""
Test module for LLM-specific media extraction in CLIver.
"""

import pytest
from unittest.mock import Mock

from cliver.llm.openai_engine import OpenAICompatibleInferenceEngine
from cliver.llm.ollama_engine import OllamaLlamaInferenceEngine
from cliver.media import MediaType
from cliver.config import ModelConfig
from cliver.model_capabilities import ModelCapability
from langchain_core.messages import AIMessage


class TestLLMMediaExtraction:
    """Test LLM-specific media extraction functionality."""

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
        return engine

    @pytest.fixture
    def ollama_engine(self):
        """Create a mock Ollama engine."""
        config = Mock(spec=ModelConfig)
        config.name = "llava"
        config.provider = "ollama"
        config.name_in_provider = "llava"
        config.url = "http://localhost:11434"
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
        return engine

    def test_openai_extract_media_from_data_url(self, openai_engine):
        """Test OpenAI engine extracts media from data URL."""
        # Create a response with a data URL
        content = "Here's an image: data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        response = AIMessage(content=content)

        media_content = openai_engine.extract_media_from_response(response)
        assert len(media_content) == 1
        assert media_content[0].type == MediaType.IMAGE
        assert media_content[0].mime_type == "image/png"
        assert "openai_generated" in media_content[0].filename

    def test_ollama_extract_media_from_data_url(self, ollama_engine):
        """Test Ollama engine extracts media from data URL."""
        # Create a response with a data URL
        content = "Here's an image: data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAAMCAgMCAgMDAwMEAwMEBQgFBAQE"
        response = AIMessage(content=content)

        media_content = ollama_engine.extract_media_from_response(response)
        assert len(media_content) == 1
        assert media_content[0].type == MediaType.IMAGE
        assert media_content[0].mime_type == "image/jpeg"
        assert "ollama_generated" in media_content[0].filename

    def test_engines_inherit_extract_method(self, openai_engine, ollama_engine):
        """Test that both engines inherit the extract_media_from_response method."""
        # Both engines should have the method from the base class
        assert hasattr(openai_engine, 'extract_media_from_response')
        assert hasattr(ollama_engine, 'extract_media_from_response')

        # Test with empty response
        empty_response = AIMessage(content="")
        openai_media = openai_engine.extract_media_from_response(empty_response)
        ollama_media = ollama_engine.extract_media_from_response(empty_response)

        # Both should return empty lists for empty content
        assert openai_media == []
        assert ollama_media == []


if __name__ == "__main__":
    pytest.main([__file__])