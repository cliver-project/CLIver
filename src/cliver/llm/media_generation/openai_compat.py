"""OpenAI-compatible media generation helpers (image, audio/TTS, video)."""

import logging
from typing import Any, Dict, Optional

from cliver.llm.media_generation.base import MediaGenerationHelper
from cliver.media import MediaContent, MediaType

logger = logging.getLogger(__name__)

# -- Image ----------------------------------------------------------------

DEFAULT_IMAGE_MODEL = "dall-e-3"


class OpenAIImageHelper(MediaGenerationHelper):
    def build_request(
        self,
        prompt: str,
        model_name: Optional[str] = None,
        **params,
    ) -> Dict[str, Any]:
        body = {
            "model": model_name or DEFAULT_IMAGE_MODEL,
            "prompt": prompt,
            "response_format": params.pop("response_format", "url"),
        }
        for key in ("size", "quality", "n", "style"):
            if key in params:
                body[key] = params[key]
        return body

    def parse_response(self, response_data: Dict[str, Any] | bytes) -> list[MediaContent]:
        if isinstance(response_data, bytes):
            raise RuntimeError("OpenAI image: expected JSON, got bytes")
        images = response_data.get("data", [])
        return [
            MediaContent(
                type=MediaType.IMAGE,
                data=item.get("url", item.get("b64_json", "")),
                mime_type="image/png",
                source="openai_image_generation",
            )
            for item in images
        ]


# -- Audio / TTS ----------------------------------------------------------

DEFAULT_TTS_MODEL = "tts-1"
_TTS_MIME: dict[str, str] = {
    "mp3": "audio/mpeg",
    "opus": "audio/opus",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "wav": "audio/wav",
    "pcm": "audio/pcm",
}


class OpenAIAudioHelper(MediaGenerationHelper):
    """OpenAI-compatible TTS — binary audio response."""

    def build_request(
        self,
        prompt: str,
        model_name: Optional[str] = None,
        **params,
    ) -> Dict[str, Any]:
        return {
            "model": model_name or DEFAULT_TTS_MODEL,
            "input": prompt,
            "voice": params.get("voice", "alloy"),
            "speed": float(params.get("speed", 1.0)),
            "response_format": params.get("format", params.get("response_format", "mp3")),
        }

    def parse_response(self, response_data: Dict[str, Any] | bytes) -> list[MediaContent]:
        if not isinstance(response_data, bytes):
            raise RuntimeError(f"OpenAI TTS: expected binary audio, got {type(response_data)}")
        fmt = "mp3"
        return [
            MediaContent(
                type=MediaType.AUDIO,
                data=response_data,
                mime_type=_TTS_MIME.get(fmt, "audio/mpeg"),
                filename=f"tts_output.{fmt}",
                source="openai_tts",
            )
        ]


# -- Video ----------------------------------------------------------------

DEFAULT_VIDEO_MODEL = "sora-2"


class OpenAIVideoHelper(MediaGenerationHelper):
    """OpenAI-compatible video generation — JSON response with task/URL."""

    def build_request(
        self,
        prompt: str,
        model_name: Optional[str] = None,
        **params,
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "model": model_name or DEFAULT_VIDEO_MODEL,
            "prompt": prompt,
        }
        for key in ("duration", "resolution", "size", "n", "style", "aspect_ratio"):
            if key in params:
                body[key] = params[key]
        if "first_frame_image" in params:
            body["first_frame_image"] = params["first_frame_image"]
        return body

    def parse_response(self, response_data: Dict[str, Any] | bytes) -> list[MediaContent]:
        if isinstance(response_data, bytes):
            raise RuntimeError("OpenAI video: expected JSON, got bytes")
        # Direct URL response
        url = response_data.get("url") or response_data.get("video_url") or response_data.get("output_url")
        if url:
            return [MediaContent(type=MediaType.VIDEO, data=url, mime_type="video/mp4", source="openai_video")]
        # data array (like image gen format)
        items = response_data.get("data", [])
        if items:
            return [
                MediaContent(
                    type=MediaType.VIDEO,
                    data=item.get("url", item.get("video_url", "")),
                    mime_type="video/mp4",
                    source="openai_video",
                )
                for item in items
            ]
        raise RuntimeError("OpenAI video: unrecognised response format")
