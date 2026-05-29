"""Media response handler — text extraction and media saving utilities."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from cliver.media import MediaContent, MediaType, get_file_extension
from cliver.messages import CLIverMessage
from cliver.provider import CLIverResponse

logger = logging.getLogger(__name__)


class MultimediaResponse:
    """Represents a multimedia response from an LLM."""

    def __init__(
        self,
        text_content: str = "",
        media_content: List[MediaContent] = None,
    ):
        self.text_content = text_content
        self.media_content = media_content or []

    def has_text(self) -> bool:
        return bool(self.text_content and self.text_content.strip())

    def has_media(self) -> bool:
        return len(self.media_content) > 0

    def get_media_by_type(self, media_type: MediaType) -> List[MediaContent]:
        return [m for m in self.media_content if m.type == media_type]


def extract_response_text(response, fallback: str = "") -> str:
    """Extract text from any response type — CLIverResponse, CLIverMessage, str, or None."""
    if response is None:
        return fallback

    if isinstance(response, CLIverResponse):
        return response.message.text or fallback

    if isinstance(response, CLIverMessage):
        return response.text or fallback

    if isinstance(response, str):
        return response

    if hasattr(response, "text"):
        return response.text or fallback

    return fallback


class MultimediaResponseHandler:
    def __init__(self, save_directory: Optional[str] = None):
        self.save_directory = Path(save_directory) if save_directory else Path.cwd()
        self.save_directory.mkdir(parents=True, exist_ok=True)

    def process_response(self, response) -> MultimediaResponse:
        text = extract_response_text(response)
        media = []

        if isinstance(response, CLIverResponse) and response.media:
            media = response.media
        elif hasattr(response, "additional_kwargs"):
            kwargs_media = response.additional_kwargs.get("media_content", [])
            if kwargs_media:
                media = kwargs_media

        return MultimediaResponse(text_content=text, media_content=media)

    def save_media_content(self, response: MultimediaResponse, prefix: str = "cliver_media") -> List[str]:
        saved = []
        for i, media in enumerate(response.media_content):
            try:
                if media.saved_path:
                    saved.append(media.saved_path)
                    continue
                if not media.filename:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    ext = get_file_extension(media.mime_type)
                    media.filename = f"{prefix}_{ts}_{i}{ext}"
                path = self.save_directory / media.filename
                media.save(path)
                saved.append(str(path))
                logger.info("Saved media to %s", path)
            except Exception as e:
                logger.error("Error saving media %s: %s", media.filename, e)
        return saved

    @staticmethod
    def display_response(response: MultimediaResponse, show_text: bool = True, show_media_info: bool = True) -> str:
        output = []
        if show_text and response.has_text():
            output.append(response.text_content)
        if show_media_info and response.has_media():
            output.append(f"\n[Media Content: {len(response.media_content)} items]")
            for i, media in enumerate(response.media_content):
                info = f"  {i + 1}. {media.type.value}"
                if media.filename:
                    info += f" ({media.filename})"
                if media.mime_type:
                    info += f" [{media.mime_type}]"
                output.append(info)
        return "\n".join(output)

    @staticmethod
    def get_response_summary(response: MultimediaResponse) -> Dict[str, Any]:
        return {
            "has_text": response.has_text(),
            "text_length": len(response.text_content) if response.text_content else 0,
            "has_media": response.has_media(),
            "media_count": len(response.media_content),
            "media_types": [m.type.value for m in response.media_content],
            "media_filenames": [m.filename for m in response.media_content if m.filename],
        }


def save_response_media(response, save_directory: str = None, prefix: str = "cliver_media") -> List[str]:
    handler = MultimediaResponseHandler(save_directory)
    if not isinstance(response, MultimediaResponse):
        response = handler.process_response(response)
    return handler.save_media_content(response, prefix)
