"""
Media utilities for LLM engines.

This module provides common functionality for extracting media content
from LLM responses across different providers.
"""

import re
import logging
from typing import List, Optional

from cliver.media import MediaContent, MediaType, get_file_extension

logger = logging.getLogger(__name__)


def extract_data_urls(text: str) -> List[str]:
    """
    Extract data URLs from text content.

    Args:
        text: Text content to search

    Returns:
        List of data URLs found in the text
    """
    # Pattern to match data URLs
    data_url_pattern = r'data:[^,\s]+;base64,[A-Za-z0-9+/=]+'
    return re.findall(data_url_pattern, text)


def _determine_media_type_and_filename(mime_type: str, filename_base: str) -> tuple[MediaType, str]:
    """
    Determine media type from MIME type and generate filename.

    Args:
        mime_type: MIME type string
        filename_base: Base name for the file

    Returns:
        Tuple of (MediaType, filename)
    """
    # Determine media type from MIME type
    if mime_type.startswith('image/'):
        media_type = MediaType.IMAGE
    elif mime_type.startswith('audio/'):
        media_type = MediaType.AUDIO
    elif mime_type.startswith('video/'):
        media_type = MediaType.VIDEO
    else:
        # Unknown media type
        media_type = MediaType.TEXT

    # Generate filename with appropriate extension
    extension = get_file_extension(mime_type)
    filename = f"{filename_base}{extension}"

    return media_type, filename


def data_url_to_media_content(data_url: str, filename_base: str) -> Optional[MediaContent]:
    """
    Convert a data URL to a MediaContent object.

    Args:
        data_url: Data URL string (data:image/png;base64,...)
        filename_base: Base name for the file

    Returns:
        MediaContent object or None if invalid
    """
    if not data_url.startswith('data:'):
        return None

    # Split data URL into parts
    try:
        mime_part, base64_data = data_url.split(',', 1)
        mime_type = mime_part.split(':')[1].split(';')[0]

        # Determine media type and filename
        media_type, filename = _determine_media_type_and_filename(mime_type, filename_base)

        return MediaContent(
            type=media_type,
            data=base64_data,
            mime_type=mime_type,
            filename=filename,
            source="llm_response"
        )
    except Exception as e:
        logger.warning(f"Error converting data URL to MediaContent: {e}")
        return None


def extract_media_from_json(parsed_content: dict, source_prefix: str = "llm") -> List[MediaContent]:
    """
    Extract media content from parsed JSON response.

    Args:
        parsed_content: Parsed JSON content
        source_prefix: Prefix for the source field

    Returns:
        List of MediaContent objects
    """
    media_content = []

    # Check for media content in structured response
    if isinstance(parsed_content, dict):
        # Look for common patterns in LLM responses
        media_items = parsed_content.get('media_content', [])
        if not media_items and 'data' in parsed_content:
            # Some APIs return media in a 'data' field
            data_items = parsed_content.get('data', [])
            if isinstance(data_items, list):
                media_items = data_items

        if isinstance(media_items, list):
            for i, item in enumerate(media_items):
                if isinstance(item, dict):
                    media = dict_to_media_content(item, f"{source_prefix}_json_{i}")
                    if media:
                        media_content.append(media)

    return media_content


def dict_to_media_content(data: dict, filename_base: str) -> Optional[MediaContent]:
    """
    Convert a dictionary to a MediaContent object.

    Args:
        data: Dictionary with media content data
        filename_base: Base name for the file

    Returns:
        MediaContent object or None if invalid
    """
    try:
        # Extract required fields
        mime_type = data.get('mime_type', '')
        base64_data = data.get('data', '')

        if not mime_type or not base64_data:
            return None

        # Determine media type and filename
        media_type, filename = _determine_media_type_and_filename(mime_type, filename_base)

        return MediaContent(
            type=media_type,
            data=base64_data,
            mime_type=mime_type,
            filename=filename,
            source="llm_json_response"
        )
    except Exception as e:
        logger.warning(f"Error converting dict to MediaContent: {e}")
        return None
