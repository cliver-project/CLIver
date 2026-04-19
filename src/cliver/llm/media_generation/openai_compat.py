"""OpenAI-compatible image generation API formatter.

Request format:  {"model": "dall-e-3", "prompt": "...", "response_format": "url"}
Response format: {"data": [{"url": "..."}, ...]}
"""

import logging
from typing import Any, Dict, Optional

from cliver.llm.media_generation.base import ImageGenerationHelper
from cliver.media import MediaContent, MediaType

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "dall-e-3"


class OpenAIImageHelper(ImageGenerationHelper):
    def build_request(
        self,
        prompt: str,
        model_name: Optional[str] = None,
        **params,
    ) -> Dict[str, Any]:
        body = {
            "model": model_name or DEFAULT_MODEL,
            "prompt": prompt,
            "response_format": params.pop("response_format", "url"),
        }
        for key in ("size", "quality", "n", "style"):
            if key in params:
                body[key] = params[key]
        return body

    def parse_response(self, response_data: Dict[str, Any]) -> list[MediaContent]:
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
