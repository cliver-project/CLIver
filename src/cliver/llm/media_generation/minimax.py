"""MiniMax image generation API formatter.

Request format:  {"model": "image-01", "prompt": "...", "response_format": "url"}
Response format: {"data": {"image_urls": [...]}, "base_resp": {"status_code": 0, ...}}
"""

import logging
from typing import Any, Dict, Optional

from cliver.llm.media_generation.base import ImageGenerationHelper
from cliver.media import MediaContent, MediaType

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "image-01"


class MiniMaxImageHelper(ImageGenerationHelper):
    def build_request(
        self,
        prompt: str,
        model_name: Optional[str] = None,
        **params,
    ) -> Dict[str, Any]:
        response_format = params.pop("response_format", "b64_json")
        body = {
            "model": model_name or DEFAULT_MODEL,
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
