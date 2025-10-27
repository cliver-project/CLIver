"""
Test module for OpenAI-specific media extraction in CLIver.
"""

import json
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage

from cliver.config import ModelConfig
from cliver.llm.openai_engine import OpenAICompatibleInferenceEngine
from cliver.media import MediaType
from cliver.model_capabilities import ModelCapability


class TestOpenAISpecificMediaExtraction:
    """Test OpenAI-specific media extraction functionality."""

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

    def test_openai_text_response_no_media(self, openai_engine):
        """Test OpenAI engine with text-only response (most common case)."""
        # GPT models typically return text responses
        content = "The image shows a beautiful landscape with mountains and a lake."
        response = AIMessage(content=content)

        media_content = openai_engine.extract_media_from_response(response)
        assert len(media_content) == 0

    def test_openai_data_url_in_text(self, openai_engine):
        """Test OpenAI engine extracts data URL from text content."""
        # Response containing a data URL (less common but possible)
        content = "Here's the image I generated: data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        response = AIMessage(content=content)

        media_content = openai_engine.extract_media_from_response(response)
        assert len(media_content) == 1
        assert media_content[0].type == MediaType.IMAGE
        assert media_content[0].mime_type == "image/png"
        assert "openai_generated" in media_content[0].filename

    def test_openai_dalle_url_response(self, openai_engine):
        """Test OpenAI engine handles DALL-E URL-based responses."""
        # DALL-E response format with image URLs (actual OpenAI format)
        dalle_response = {
            "data": [
                {"url": "https://oaidalleprodscus.blob.core.windows.net/private/org-X/blah1.png"},
                {"url": "https://oaidalleprodscus.blob.core.windows.net/private/org-X/blah2.png"},
            ]
        }
        response = AIMessage(content=json.dumps(dalle_response))

        media_content = openai_engine.extract_media_from_response(response)
        assert len(media_content) == 2
        assert media_content[0].type == MediaType.IMAGE
        assert media_content[1].type == MediaType.IMAGE
        assert "OpenAI generated image URL" in media_content[0].data
        assert "openai_image_0.png" == media_content[0].filename
        assert "openai_image_1.png" == media_content[1].filename

    def test_openai_dalle_b64_response(self, openai_engine):
        """Test OpenAI engine handles DALL-E base64 responses."""
        # DALL-E response with base64 encoded images
        dalle_response = {
            "data": [
                {
                    "b64_json": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
                }
            ]
        }
        response = AIMessage(content=json.dumps(dalle_response))

        media_content = openai_engine.extract_media_from_response(response)
        assert len(media_content) == 1
        assert media_content[0].type == MediaType.IMAGE
        assert media_content[0].mime_type == "image/png"
        assert "openai_image_0.png" == media_content[0].filename

    def test_openai_additional_kwargs_with_image_urls(self, openai_engine):
        """Test OpenAI engine handles image URLs in additional_kwargs."""
        # Response with image URLs in additional_kwargs (tool responses)
        response = AIMessage(
            content="Here are the images you requested",
            additional_kwargs={
                "image_urls": [
                    "https://example.com/image1.png",
                    "https://example.com/image2.jpg",
                ]
            },
        )

        media_content = openai_engine.extract_media_from_response(response)
        assert len(media_content) == 2
        assert media_content[0].type == MediaType.IMAGE
        assert media_content[1].type == MediaType.IMAGE
        assert "OpenAI tool response image URL" in media_content[0].data
        assert "openai_tool_image_0.png" == media_content[0].filename

    def test_openai_structured_content_response(self, openai_engine):
        """Test OpenAI engine handles structured content responses."""
        # Structured content response (more for input messages, but we handle it)
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

        media_content = openai_engine.extract_media_from_response(response)
        assert len(media_content) == 1
        assert media_content[0].type == MediaType.IMAGE
        assert "openai_structured_image" in media_content[0].filename

    def test_openai_structured_content_with_http_url(self, openai_engine):
        """Test OpenAI engine handles structured content with HTTP URLs."""
        # Structured content with HTTP URL
        structured_content = [
            {"type": "text", "text": "What's in this image?"},
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/image.jpg"},
            },
        ]
        response = AIMessage(content=structured_content)

        media_content = openai_engine.extract_media_from_response(response)
        assert len(media_content) == 1
        assert media_content[0].type == MediaType.IMAGE
        assert "OpenAI image URL" in media_content[0].data
        assert "openai_image_from_url.png" == media_content[0].filename


if __name__ == "__main__":
    pytest.main([__file__])
