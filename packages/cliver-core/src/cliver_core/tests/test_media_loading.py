"""``load_media_file`` URL loading uses the stdlib (no ``requests`` dependency)."""

import base64
import urllib.error

import pytest

from cliver_core.media import MediaType, load_media_file

_PAYLOAD = b"\x89PNG\r\n\x1a\n"


class _Response:
    def __init__(self, headers: dict[str, str]) -> None:
        self.headers = headers

    def read(self) -> bytes:
        return _PAYLOAD

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *_exc) -> bool:
        return False


def test_downloads_url_with_stdlib(monkeypatch) -> None:
    seen: dict = {}

    def fake_urlopen(request, timeout=None):
        seen["url"] = request.full_url
        seen["timeout"] = timeout
        return _Response(
            {
                "content-type": "image/png; charset=binary",
                "content-disposition": 'attachment; filename="remote.png"',
            }
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    media = load_media_file("https://example.com/download")

    assert seen == {"url": "https://example.com/download", "timeout": 120}
    assert media.type is MediaType.IMAGE
    assert media.mime_type == "image/png"
    assert media.filename == "remote.png"
    assert media.source == "url"
    assert media.data == base64.b64encode(_PAYLOAD).decode()


def test_download_uses_headerless_filename_and_guessed_mime(monkeypatch) -> None:
    def fake_urlopen(request, timeout=None):
        return _Response({})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    media = load_media_file("https://example.com/media/clip.mp3")

    assert media.type is MediaType.AUDIO
    assert media.mime_type == "audio/mpeg"
    assert media.filename == "clip.mp3"


def test_download_failure_is_a_value_error(monkeypatch) -> None:
    def fake_urlopen(request, timeout=None):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    with pytest.raises(ValueError, match="Failed to download media from URL"):
        load_media_file("https://example.com/broken")
