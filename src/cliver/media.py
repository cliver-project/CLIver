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

logger = logging.getLogger(__name__)


class MediaType(Enum):
    """Enumeration of media types."""

    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    TEXT = "text"


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


@dataclass
class MediaContent:
    """Represents a media content item."""

    type: MediaType
    data: str  # base64 encoded data
    mime_type: str  # e.g., "image/jpeg", "audio/wav"
    filename: Optional[str] = None
    source: str = "local"  # local, url, etc.

    def save(self, file_path: Path) -> bool:
        """
        Save the media content to a file.

        Args:
            file_path: Path where to save the file

        Returns:
            True if successful, False otherwise

        Raises:
            Exception: If failed to save the file
        """
        try:
            # Handle data URLs (data:image/jpeg;base64,...)
            if self.data.startswith("data:"):
                base64_data = self.data.split(",", 1)[1] if "," in self.data else self.data
            else:
                base64_data = self.data

            # Decode base64 data
            binary_data = base64.b64decode(base64_data)

            # Write to file
            with open(file_path, "wb") as f:
                f.write(binary_data)

            return True
        except Exception as e:
            logger.error("Error saving media to %s, exception: %s", file_path, e, exc_info=True)
            raise e

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

    import requests

    # Check if source is a URL
    parsed_url = urlparse(source)
    is_url = bool(parsed_url.scheme and parsed_url.netloc)

    file_data = None
    filename = None

    if is_url:
        # Load from URL
        try:
            response = requests.get(source)
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
