"""
Test module for Ollama-specific media extraction in CLIver.
"""

import json
from unittest.mock import AsyncMock, Mock, patch

import pytest
from langchain_core.messages import AIMessage

from cliver.config import ModelConfig
from cliver.llm.unified_engine import UnifiedInferenceEngine
from cliver.media import MediaType


class TestOllamaSpecificMediaExtraction:
    """Test Ollama-specific media extraction functionality."""

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

        return UnifiedInferenceEngine(config)

    def test_ollama_text_response_no_media(self, ollama_engine):
        """Test Ollama engine with text-only response (most common case)."""
        content = "The image shows a beautiful landscape with mountains and a lake."
        response = AIMessage(content=content)

        media_content = ollama_engine.extract_media_from_response(response)
        assert len(media_content) == 0

    def test_ollama_data_url_in_text(self, ollama_engine):
        """Test Ollama engine extracts data URL from text content."""
        content = "Here's the image I generated: data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        response = AIMessage(content=content)

        media_content = ollama_engine.extract_media_from_response(response)
        assert len(media_content) == 1
        assert media_content[0].type == MediaType.IMAGE
        assert media_content[0].mime_type == "image/png"
        assert "openai_generated" in media_content[0].filename

    def test_ollama_image_generation_response(self, ollama_engine):
        """Test engine handles image generation response with base64 data."""
        image_response = {
            "data": [
                {
                    "b64_json": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
                },
                {"b64_json": "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAAMCAgMCAgMDAwMEAwMEBQgFBAQE"},
            ]
        }
        response = AIMessage(content=json.dumps(image_response))

        media_content = ollama_engine.extract_media_from_response(response)
        assert len(media_content) == 2
        assert media_content[0].type == MediaType.IMAGE
        assert media_content[1].type == MediaType.IMAGE
        assert "openai_image_0.png" == media_content[0].filename
        assert "openai_image_1.png" == media_content[1].filename

    def test_ollama_additional_kwargs_with_images(self, ollama_engine):
        """Test engine handles images in additional_kwargs."""
        response = AIMessage(
            content="Here are the images you requested",
            additional_kwargs={
                "image_urls": [
                    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==",
                    "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAAMCAgMCAgMDAwMEAwMEBQgFBAQE",
                ]
            },
        )

        media_content = ollama_engine.extract_media_from_response(response)
        assert len(media_content) == 2
        assert media_content[0].type == MediaType.IMAGE
        assert media_content[1].type == MediaType.IMAGE
        assert "openai_tool_image_0.png" == media_content[0].filename
        assert "openai_tool_image_1.png" == media_content[1].filename

    def test_ollama_structured_content_response(self, ollama_engine):
        """Test Ollama engine handles structured content responses."""
        structured_content = [
            {"type": "text", "text": "What's in this image?"},
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAAMCAgMCAgMDAwMEAwMEBQgFBAQE"
                },
            },
        ]
        response = AIMessage(content=structured_content)

        media_content = ollama_engine.extract_media_from_response(response)
        assert len(media_content) == 1
        assert media_content[0].type == MediaType.IMAGE
        assert "openai_structured_image" in media_content[0].filename

    def test_ollama_structured_content_with_http_url(self, ollama_engine):
        """Test Ollama engine handles structured content with HTTP URLs."""
        structured_content = [
            {"type": "text", "text": "What's in this image?"},
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/image.jpg"},
            },
        ]
        response = AIMessage(content=structured_content)

        media_content = ollama_engine.extract_media_from_response(response)
        assert len(media_content) == 1
        assert media_content[0].type == MediaType.IMAGE
        assert "openai image URL" in media_content[0].data
        assert "openai_image_from_url.png" == media_content[0].filename


if __name__ == "__main__":
    pytest.main([__file__])
