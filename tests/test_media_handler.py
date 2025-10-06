"""
Test module for multimedia response handler in CLIver.
"""

import pytest
import tempfile

from cliver.media_handler import MultimediaResponseHandler, MultimediaResponse
from cliver.media import MediaContent, MediaType


class TestMultimediaResponseHandler:
    """Test multimedia response handler functionality."""

    def test_handler_initialization(self):
        """Test MultimediaResponseHandler initialization."""
        # Test with default directory
        handler = MultimediaResponseHandler()
        assert handler.save_directory.exists()

        # Test with custom directory
        with tempfile.TemporaryDirectory() as temp_dir:
            handler = MultimediaResponseHandler(temp_dir)
            assert str(handler.save_directory) == temp_dir

    def test_process_text_response(self):
        """Test processing text response."""
        handler = MultimediaResponseHandler()

        # Test string response
        response = "Hello, world!"
        result = handler.process_response(response)
        assert isinstance(result, MultimediaResponse)
        assert result.text_content == "Hello, world!"
        assert result.has_text()
        assert not result.has_media()

    def test_process_aimessage_response(self):
        """Test processing AIMessage response."""
        from langchain_core.messages import AIMessage

        handler = MultimediaResponseHandler()

        # Test AIMessage with text content
        response = AIMessage(content="Hello, world!")
        result = handler.process_response(response)
        assert isinstance(result, MultimediaResponse)
        assert result.text_content == "Hello, world!"
        assert result.has_text()
        assert not result.has_media()

    def test_display_response(self):
        """Test displaying response."""
        handler = MultimediaResponseHandler()

        # Test text-only response
        response = MultimediaResponse(text_content="Hello, world!")
        output = handler.display_response(response)
        assert output == "Hello, world!"

        # Test response with media
        media = MediaContent(
            type=MediaType.IMAGE,
            data="base64data",
            mime_type="image/jpeg",
            filename="test.jpg"
        )
        response = MultimediaResponse(
            text_content="Here's an image:",
            media_content=[media]
        )
        output = handler.display_response(response)
        assert "Here's an image:" in output
        assert "[Media Content: 1 items]" in output
        assert "1. image (test.jpg) [image/jpeg]" in output

    def test_response_summary(self):
        """Test getting response summary."""
        handler = MultimediaResponseHandler()

        # Test text-only response
        response = MultimediaResponse(text_content="Hello, world!")
        summary = handler.get_response_summary(response)
        assert summary['has_text'] == True
        assert summary['text_length'] == 13
        assert summary['has_media'] == False
        assert summary['media_count'] == 0

        # Test response with media
        media = MediaContent(
            type=MediaType.IMAGE,
            data="base64data",
            mime_type="image/jpeg",
            filename="test.jpg"
        )
        response = MultimediaResponse(
            text_content="Here's an image:",
            media_content=[media]
        )
        summary = handler.get_response_summary(response)
        assert summary['has_text'] == True
        assert summary['has_media'] == True
        assert summary['media_count'] == 1
        assert summary['media_types'] == ['image']
        assert summary['media_filenames'] == ['test.jpg']


if __name__ == "__main__":
    pytest.main([__file__])