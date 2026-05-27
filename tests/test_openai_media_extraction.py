"""
Test module for OpenAI-specific media extraction in CLIver.
"""

import json
from unittest.mock import AsyncMock, Mock, patch

import pytest
from langchain_core.messages import AIMessage

from cliver.config import ModelConfig
from cliver.llm.unified_engine import UnifiedInferenceEngine
from cliver.media import MediaType


class TestOpenAISpecificMediaExtraction:
    """Test OpenAI-specific media extraction functionality."""

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

        return UnifiedInferenceEngine(config)

    def test_openai_text_response_no_media(self, openai_engine):
        """Test OpenAI engine with text-only response (most common case)."""
        content = "The image shows a beautiful landscape with mountains and a lake."
        response = AIMessage(content=content)

        media_content = openai_engine.extract_media_from_response(response)
        assert len(media_content) == 0

    def test_openai_data_url_in_text(self, openai_engine):
        """Test OpenAI engine extracts data URL from text content."""
        content = "Here's the image I generated: data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        response = AIMessage(content=content)

        media_content = openai_engine.extract_media_from_response(response)
        assert len(media_content) == 1
        assert media_content[0].type == MediaType.IMAGE
        assert media_content[0].mime_type == "image/png"
        assert "openai_generated" in media_content[0].filename

    def test_openai_dalle_url_response(self, openai_engine):
        """Test OpenAI engine handles DALL-E URL-based responses."""
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
        assert "openai generated image URL" in media_content[0].data
        assert "openai_image_0.png" == media_content[0].filename
        assert "openai_image_1.png" == media_content[1].filename

    def test_openai_dalle_b64_response(self, openai_engine):
        """Test OpenAI engine handles DALL-E base64 responses."""
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
        assert "openai tool response image URL" in media_content[0].data
        assert "openai_tool_image_0.png" == media_content[0].filename

    def test_openai_structured_content_response(self, openai_engine):
        """Test OpenAI engine handles structured content responses."""
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
        assert "openai image URL" in media_content[0].data
        assert "openai_image_from_url.png" == media_content[0].filename


if __name__ == "__main__":
    pytest.main([__file__])
