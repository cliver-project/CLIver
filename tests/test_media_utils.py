"""
Test module for media utilities in CLIver.
"""

import pytest
from cliver.llm.media_utils import (
    extract_data_urls,
    data_url_to_media_content,
    extract_media_from_json,
    get_file_extension
)
from cliver.media import MediaType


class TestMediaUtils:
    """Test media utilities functionality."""

    def test_extract_data_urls(self):
        """Test extracting data URLs from text."""
        text = "Here's an image: data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        urls = extract_data_urls(text)
        assert len(urls) == 1
        assert urls[0].startswith("data:image/png;base64,")

        # Test with no data URLs
        text = "Just plain text with no data URLs"
        urls = extract_data_urls(text)
        assert len(urls) == 0

    def test_data_url_to_media_content(self):
        """Test converting data URL to MediaContent."""
        data_url = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="

        media = data_url_to_media_content(data_url, "test_image")
        assert media is not None
        assert media.type == MediaType.IMAGE
        assert media.mime_type == "image/png"
        assert media.filename == "test_image.png"
        assert "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==" in media.data

        # Test with invalid data URL
        invalid_url = "not a data url"
        media = data_url_to_media_content(invalid_url, "test")
        assert media is None

    def test_get_file_extension(self):
        """Test getting file extension from MIME type."""
        assert get_file_extension("image/png") == ".png"
        assert get_file_extension("image/jpeg") == ".jpg"
        assert get_file_extension("audio/wav") == ".wav"
        assert get_file_extension("unknown/type") == ".bin"

    def test_extract_media_from_json(self):
        """Test extracting media from JSON content."""
        # Test with media content in expected format
        json_content = {
            "media_content": [
                {
                    "mime_type": "image/png",
                    "data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
                }
            ]
        }

        media_list = extract_media_from_json(json_content, "test")
        assert len(media_list) == 1
        assert media_list[0].type == MediaType.IMAGE
        assert media_list[0].mime_type == "image/png"

        # Test with empty content
        empty_content = {}
        media_list = extract_media_from_json(empty_content, "test")
        assert len(media_list) == 0


if __name__ == "__main__":
    pytest.main([__file__])