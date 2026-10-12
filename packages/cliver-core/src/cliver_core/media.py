"""
Media content module for Cliver client.
Defines data structures for handling multi-media content in LLM interactions.
"""

import base64
import logging
import mimetypes
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

from cliver_core.util import get_app_user_agent, url_request

logger = logging.getLogger(__name__)


def _download_url(url: str) -> bytes:
    """Download URL content using url_request (stdlib, with User-Agent)."""
    import urllib.request

    req = url_request(url, timeout=120)
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read()


class MediaType(Enum):
    """Enumeration of media types."""

    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    TEXT = "text"
    FILE = "file"  # anything the types above don't classify (custom media kinds)


# Shared MIME type to file extension mapping
_MIME_TO_EXTENSION = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/gif": ".gif",
    "image/webp": ".webp",
    "audio/wav": ".wav",
    "audio/mp3": ".mp3",
    "audio/mpeg": ".mp3",
    "video/mp4": ".mp4",
    "video/quicktime": ".mov",
    "video/x-msvideo": ".avi",
    "model/gltf+json": ".gltf",
    "model/gltf-binary": ".glb",
}


def get_file_extension(mime_type: str) -> str:
    """
    Get file extension from MIME type.

    Args:
        mime_type: MIME type string

    Returns:
        File extension including the dot (e.g., '.jpg')
    """
    return _MIME_TO_EXTENSION.get(mime_type, ".bin")


#: Image magic bytes → MIME — the containers vendors actually return.
_IMAGE_MAGIC = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
)


def sniff_image_mime(payload: bytes) -> Optional[str]:
    """MIME of an image payload from its magic bytes, or ``None`` if unknown."""
    if payload[:4] == b"RIFF" and payload[8:12] == b"WEBP":
        return "image/webp"
    for magic, mime in _IMAGE_MAGIC:
        if payload.startswith(magic):
            return mime
    return None


def resolve_media_mime(default: str, *, url: Optional[str] = None, payload: Optional[bytes] = None) -> str:
    """Best-effort real MIME for a vendor-generated media item.

    Vendors generate what they generate — MiniMax image-01 returns JPEG, not
    the flow's PNG default — so the label follows the evidence: the payload's
    magic bytes when the caller already holds them, then the format the URL
    names for itself (stdlib ``mimetypes``, query/fragment stripped), then
    the caller's default.
    """
    if payload:
        sniffed = sniff_image_mime(payload)
        if sniffed:
            return sniffed
    if url:
        named, _ = mimetypes.guess_type(url)
        if named and named.split("/", 1)[0] in ("image", "audio", "video"):
            return named
    return default


#: MIME top-level types → media kind; everything else is ``MediaType.FILE``.
_MIME_KIND_PREFIXES = {
    "image": MediaType.IMAGE,
    "audio": MediaType.AUDIO,
    "video": MediaType.VIDEO,
    "text": MediaType.TEXT,
}


def media_type_from_mime(mime_type: str) -> MediaType:
    """Media kind for one MIME type — the top-level type decides, anything
    unclassifiable (``application/...``, ``model/...``, ...) is ``FILE``."""
    prefix = mime_type.split(";", 1)[0].strip().lower().split("/", 1)[0]
    return _MIME_KIND_PREFIXES.get(prefix, MediaType.FILE)


def extension_from_mime(mime_type: str) -> str:
    """Dotless file extension for one MIME type (parameters stripped).

    The module's own table decides first; otherwise the subtype is used
    (``model/gltf+json`` → ``gltf``), with ``octet-stream``/``binary``/empty
    falling back to ``bin``.  Deliberately NOT ``mimetypes.guess_extension``
    — its platform database returns surprises like ``.jpe``.
    """
    mime = mime_type.split(";", 1)[0].strip().lower()
    known = _MIME_TO_EXTENSION.get(mime)
    if known:
        return known.removeprefix(".")
    subtype = mime.split("/", 1)[-1].split("+", 1)[0].strip()
    if not subtype or subtype in ("octet-stream", "binary"):
        return "bin"
    return subtype


@dataclass
class MediaContent:
    """Represents a media content item."""

    type: MediaType
    data: str  # base64 encoded data or URL
    mime_type: str  # e.g., "image/jpeg", "audio/wav"
    filename: Optional[str] = None
    source: str = "local"  # local, url, etc.
    saved_path: Optional[str] = None  # set after saving to disk

    def is_url(self) -> bool:
        """Check if data is a URL rather than base64-encoded content."""
        return self.data.startswith("http://") or self.data.startswith("https://")

    def save(self, file_path: Path) -> bool:
        """
        Save the media content to a file.

        Handles three data formats:
        - URL (http/https): downloads the content
        - Data URL (data:mime;base64,...): extracts and decodes base64
        - Raw base64: decodes directly

        Args:
            file_path: Path where to save the file

        Returns:
            True if successful, False otherwise

        Raises:
            Exception: If failed to save the file
        """
        try:
            if self.is_url():
                binary_data = _download_url(self.data)
            elif self.data.startswith("data:"):
                base64_data = self.data.split(",", 1)[1] if "," in self.data else self.data
                binary_data = base64.b64decode(base64_data)
            else:
                binary_data = base64.b64decode(self.data)

            with open(file_path, "wb") as f:
                f.write(binary_data)

            self.saved_path = str(file_path)
            return True
        except Exception as e:
            logger.error("Error saving media to %s, exception: %s", file_path, e, exc_info=True)
            raise e

    def to_bytes(self) -> bytes:
        """Resolve data to raw bytes (download URL, decode base64)."""
        if self.is_url():
            return _download_url(self.data)
        elif self.data.startswith("data:"):
            base64_data = self.data.split(",", 1)[1] if "," in self.data else self.data
            return base64.b64decode(base64_data)
        else:
            return base64.b64decode(self.data)

    def get_file_extension(self) -> str:
        """
        Get file extension from MIME type.

        Returns:
            File extension including the dot (e.g., '.jpg')
        """
        return get_file_extension(self.mime_type)


def load_media_file(source: str) -> MediaContent:
    """
    Load a media file from either a local file path or a URL and convert it to MediaContent.

    Args:
        source: Path to the local media file or URL to the media file

    Returns:
        MediaContent object with base64 encoded data
    """
    from urllib.parse import urlparse

    # Check if source is a URL
    parsed_url = urlparse(source)
    is_url = bool(parsed_url.scheme and parsed_url.netloc)

    file_data = None
    filename = None

    if is_url:
        # Load from URL — requests is only needed for the remote branch,
        # so local files work without it (keeps cliver-core pydantic-only).
        import requests

        try:
            response = requests.get(source, headers={"User-Agent": get_app_user_agent()})
            response.raise_for_status()
            file_data = response.content

            # Try to get filename from URL or Content-Disposition header
            if "content-disposition" in response.headers:
                content_disposition = response.headers["content-disposition"]
                if "filename=" in content_disposition:
                    filename = content_disposition.split("filename=")[1].strip('"')

            if not filename:
                # Extract filename from URL path
                url_path = parsed_url.path
                filename = url_path.split("/")[-1] if "/" in url_path else source.split("/")[-1]

            # Get mime type from response headers or guess from filename
            if "content-type" in response.headers:
                mime_type = response.headers["content-type"].split(";")[0]  # Remove charset info
            else:
                mime_type, _ = mimetypes.guess_type(filename)

        except requests.RequestException as e:
            raise ValueError(f"Failed to download media from URL {source}: {e}") from e
    else:
        # Load from local file
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Media file not found: {source}")

        # Read file data
        with open(path, "rb") as f:
            file_data = f.read()

        filename = path.name
        mime_type, _ = mimetypes.guess_type(source)

    # Validate mime type
    if mime_type is None:
        raise ValueError(f"Unable to determine media type for source: {source}")

    # Determine media type category
    if mime_type.startswith("image/"):
        media_type = MediaType.IMAGE
    elif mime_type.startswith("audio/"):
        media_type = MediaType.AUDIO
    elif mime_type.startswith("video/"):
        media_type = MediaType.VIDEO
    else:
        raise ValueError(f"Unsupported media type: {mime_type}")

    # Encode as base64
    encoded_data = base64.b64encode(file_data).decode("utf-8")

    return MediaContent(
        type=media_type,
        data=encoded_data,
        mime_type=mime_type,
        filename=filename,
        source="url" if is_url else "local",
    )


def load_media_files(file_paths: List[str]) -> List[MediaContent]:
    """
    Load multiple media files from either local file paths or URLs.

    Args:
        file_paths: List of paths to local media files or URLs to media files

    Returns:
        List of MediaContent objects
    """
    media_content = []
    for file_path in file_paths:
        media_content.append(load_media_file(file_path))
    return media_content


def add_media_content_to_message_parts(content_parts: List[Dict], media_content: List[MediaContent]) -> None:
    """
    Add media content to message parts in the standard format.

    Args:
        content_parts: List of content parts to append to
        media_content: List of MediaContent objects to add
    """
    for media in media_content:
        if media.type == MediaType.IMAGE:
            content_parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{media.mime_type};base64,{media.data}"},
                }
            )
        # For audio/video, add as text descriptions
        elif media.type == MediaType.AUDIO:
            content_parts.append(
                {
                    "type": "text",
                    "text": f"[Audio file: {media.filename}]",
                }
            )
        elif media.type == MediaType.VIDEO:
            content_parts.append(
                {
                    "type": "text",
                    "text": f"[Video file: {media.filename}]",
                }
            )
