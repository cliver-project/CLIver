"""
Test module for Ollama-specific media extraction in CLIver.
"""

import pytest
import json
from unittest.mock import Mock

from cliver.llm.ollama_engine import OllamaLlamaInferenceEngine
from cliver.media import MediaType
from cliver.config import ModelConfig
from cliver.model_capabilities import ModelCapability
from langchain_core.messages import AIMessage


class TestOllamaSpecificMediaExtraction:
    """Test Ollama-specific media extraction functionality."""

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
        return engine

    def test_ollama_text_response_no_media(self, ollama_engine):
        """Test Ollama engine with text-only response (most common case)."""
        # LLM models typically return text responses
        content = "The image shows a beautiful landscape with mountains and a lake."
        response = AIMessage(content=content)

        media_content = ollama_engine.extract_media_from_response(response)
        assert len(media_content) == 0

    def test_ollama_data_url_in_text(self, ollama_engine):
        """Test Ollama engine extracts data URL from text content."""
        # Response containing a data URL
        content = "Here's the image I generated: data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        response = AIMessage(content=content)

        media_content = ollama_engine.extract_media_from_response(response)
        assert len(media_content) == 1
        assert media_content[0].type == MediaType.IMAGE
        assert media_content[0].mime_type == "image/png"
        assert "ollama_generated" in media_content[0].filename

    def test_ollama_image_generation_response(self, ollama_engine):
        """Test Ollama engine handles image generation response with base64 data."""
        # Ollama image generation response format with base64 data
        image_response = {
            "images": [
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==",
                "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAAMCAgMCAgMDAwMEAwMEBQgFBAQE"
            ]
        }
        response = AIMessage(content=json.dumps(image_response))

        media_content = ollama_engine.extract_media_from_response(response)
        assert len(media_content) == 2
        assert media_content[0].type == MediaType.IMAGE
        assert media_content[1].type == MediaType.IMAGE
        assert "ollama_image_0.png" == media_content[0].filename
        assert "ollama_image_1.png" == media_content[1].filename

    def test_ollama_additional_kwargs_with_images(self, ollama_engine):
        """Test Ollama engine handles images in additional_kwargs."""
        # Response with images in additional_kwargs (similar to input format)
        response = AIMessage(
            content="Here are the images you requested",
            additional_kwargs={
                "images": [
                    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==",
                    "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAAMCAgMCAgMDAwMEAwMEBQgFBAQE"
                ]
            }
        )

        media_content = ollama_engine.extract_media_from_response(response)
        assert len(media_content) == 2
        assert media_content[0].type == MediaType.IMAGE
        assert media_content[1].type == MediaType.IMAGE
        assert "ollama_tool_image_0.png" == media_content[0].filename
        assert "ollama_tool_image_1.png" == media_content[1].filename

    def test_ollama_structured_content_response(self, ollama_engine):
        """Test Ollama engine handles structured content responses."""
        # Structured content response (more for input messages, but we handle it)
        structured_content = [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAAMCAgMCAgMDAwMEAwMEBQgFBAQE"}}
        ]
        response = AIMessage(content=structured_content)

        media_content = ollama_engine.extract_media_from_response(response)
        assert len(media_content) == 1
        assert media_content[0].type == MediaType.IMAGE
        assert "ollama_structured_image" in media_content[0].filename

    def test_ollama_structured_content_with_http_url(self, ollama_engine):
        """Test Ollama engine handles structured content with HTTP URLs."""
        # Structured content with HTTP URL
        structured_content = [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
        ]
        response = AIMessage(content=structured_content)

        media_content = ollama_engine.extract_media_from_response(response)
        assert len(media_content) == 1
        assert media_content[0].type == MediaType.IMAGE
        assert "Ollama image URL" in media_content[0].data
        assert "ollama_image_from_url.png" == media_content[0].filename


if __name__ == "__main__":
    pytest.main([__file__])