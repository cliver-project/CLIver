"""Tests for PlatformAdapter ABC, MessageEvent, MediaAttachment."""

from cliver.gateway.platform_adapter import (
    MediaAttachment,
    MessageEvent,
    PlatformAdapter,
    split_message,
)


class TestMediaAttachment:
    def test_image_attachment(self):
        att = MediaAttachment(type="image", data=b"\x89PNG", mime_type="image/png")
        assert att.type == "image"
        assert att.data == b"\x89PNG"

    def test_file_attachment_with_filename(self):
        att = MediaAttachment(type="file", filename="doc.pdf", mime_type="application/pdf")
        assert att.filename == "doc.pdf"

    def test_voice_attachment(self):
        att = MediaAttachment(type="voice", data=b"audio-data", mime_type="audio/ogg")
        assert att.type == "voice"

    def test_url_attachment(self):
        att = MediaAttachment(type="image", url="https://example.com/img.png")
        assert att.url == "https://example.com/img.png"
        assert att.data is None


class TestMessageEvent:
    def test_text_message(self):
        event = MessageEvent(
            platform="telegram",
            channel_id="123",
            user_id="456",
            text="hello",
            media=[],
        )
        assert event.platform == "telegram"
        assert event.text == "hello"
        assert event.is_group is False

    def test_group_message(self):
        event = MessageEvent(
            platform="telegram",
            channel_id="-100123",
            user_id="456",
            text="hello",
            media=[],
            is_group=True,
        )
        assert event.is_group is True

    def test_message_with_media(self):
        img = MediaAttachment(type="image", data=b"img-data")
        event = MessageEvent(
            platform="discord",
            channel_id="789",
            user_id="101",
            text="look at this",
            media=[img],
        )
        assert len(event.media) == 1
        assert event.media[0].type == "image"


class TestSplitMessage:
    def test_short_message_not_split(self):
        chunks = split_message("hello world", max_length=100)
        assert chunks == ["hello world"]

    def test_long_message_split_at_newlines(self):
        text = "line1\nline2\nline3\nline4"
        chunks = split_message(text, max_length=12)
        assert len(chunks) >= 2
        # Each chunk should be <= max_length
        for chunk in chunks:
            assert len(chunk) <= 12

    def test_very_long_line_force_split(self):
        text = "a" * 200
        chunks = split_message(text, max_length=50)
        assert len(chunks) >= 4
        for chunk in chunks:
            assert len(chunk) <= 50

    def test_empty_message(self):
        chunks = split_message("", max_length=100)
        assert chunks == [""]

    def test_split_preserves_content(self):
        text = "hello\nworld\nfoo\nbar"
        chunks = split_message(text, max_length=12)
        reassembled = "\n".join(chunks)
        assert reassembled == text


class TestPlatformAdapterABC:
    def test_cannot_instantiate_abc(self):
        import pytest

        with pytest.raises(TypeError):
            PlatformAdapter()
