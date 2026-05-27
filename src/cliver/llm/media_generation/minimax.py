"""MiniMax media generation helpers (image, audio/TTS, video/Hailuo)."""

import logging
from typing import Any, Dict, Optional

from cliver.llm.media_generation.base import MediaGenerationHelper
from cliver.media import MediaContent, MediaType

logger = logging.getLogger(__name__)

# -- Image ----------------------------------------------------------------

DEFAULT_IMAGE_MODEL = "image-01"


class MiniMaxImageHelper(MediaGenerationHelper):
    def build_request(
        self,
        prompt: str,
        model_name: Optional[str] = None,
        **params,
    ) -> Dict[str, Any]:
        response_format = params.pop("response_format", "base64")
        body = {
            "model": model_name or DEFAULT_IMAGE_MODEL,
            "prompt": prompt,
            "response_format": response_format,
        }
        for key in ("aspect_ratio", "n", "prompt_optimizer", "width", "height"):
            if key in params:
                body[key] = params[key]
        return body

    def parse_response(self, response_data: Dict[str, Any]) -> list[MediaContent]:
        base_resp = response_data.get("base_resp", {})
        if base_resp.get("status_code", 0) != 0:
            raise RuntimeError(f"MiniMax image generation failed: {base_resp.get('status_msg', 'unknown error')}")

        data = response_data.get("data", {})
        items = data.get("image_base64") or data.get("image_urls") or []

        return [
            MediaContent(
                type=MediaType.IMAGE,
                data=item,
                mime_type="image/png",
                source="minimax_image_generation",
            )
            for item in items
        ]


# -- Audio / TTS ----------------------------------------------------------

DEFAULT_TTS_VOICE = "male-qn-qingse"
DEFAULT_TTS_FORMAT = "mp3"
_TTS_MIME: dict[str, str] = {"mp3": "audio/mpeg", "wav": "audio/wav", "pcm": "audio/pcm", "flac": "audio/flac"}


class MiniMaxAudioHelper(MediaGenerationHelper):
    """MiniMax TTS — binary audio response."""

    def build_request(
        self,
        prompt: str,
        model_name: Optional[str] = None,
        **params,
    ) -> Dict[str, Any]:
        return {
            "model": model_name or "speech-02-hd",
            "input": prompt,
            "voice": params.get("voice", DEFAULT_TTS_VOICE),
            "speed": float(params.get("speed", 1.0)),
            "format": params.get("format", DEFAULT_TTS_FORMAT),
        }

    def parse_response(self, response_data: Dict[str, Any] | bytes) -> list[MediaContent]:
        if not isinstance(response_data, bytes):
            raise RuntimeError(f"MiniMax TTS: expected binary audio, got {type(response_data)}")
        fmt = DEFAULT_TTS_FORMAT
        return [
            MediaContent(
                type=MediaType.AUDIO,
                data=response_data,
                mime_type=_TTS_MIME.get(fmt, "audio/mpeg"),
                filename=f"tts_output.{fmt}",
                source="minimax_tts",
            )
        ]


# -- Video / Hailuo -------------------------------------------------------


class MiniMaxVideoHelper(MediaGenerationHelper):
    """MiniMax Hailuo — task-based async video generation.

    The API returns a task_id; polling is handled externally by the caller.
    """

    def build_request(
        self,
        prompt: str,
        model_name: Optional[str] = None,
        **params,
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "model": model_name or "hailuo-2.3",
            "prompt": prompt,
            "duration": int(params.get("duration", 6)),
        }
        if "resolution" in params:
            body["resolution"] = params["resolution"]
        if "first_frame_image" in params:
            body["first_frame_image"] = params["first_frame_image"]
        return body

    def parse_response(self, response_data: Dict[str, Any] | bytes) -> list[MediaContent]:
        if isinstance(response_data, bytes):
            raise RuntimeError("MiniMax video: expected JSON, got bytes")
        task_id = response_data.get("task_id") or response_data.get("id")
        if not task_id:
            url = _extract_video_url(response_data)
            if url:
                return [MediaContent(type=MediaType.VIDEO, data=url, mime_type="video/mp4", source="minimax_video")]
            raise RuntimeError("MiniMax video: no task_id or URL in response")
        raise RuntimeError(f"MiniMax video task created (task_id={task_id}). Polling not yet implemented.")


def _extract_video_url(data: dict) -> str | None:
    for key in ("video_url", "url", "output_url", "result_url"):
        if key in data and data[key]:
            return data[key]
    result = data.get("result") or data.get("data")
    if isinstance(result, dict):
        for key in ("video_url", "url"):
            if key in result and result[key]:
                return result[key]
    results = data.get("results") or data.get("videos")
    if isinstance(results, list) and results:
        r = results[0]
        if isinstance(r, dict):
            return r.get("url") or r.get("video_url")
    return None
