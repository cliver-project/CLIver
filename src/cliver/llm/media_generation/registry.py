"""Helper registry — selects the right image generation helper by URL pattern."""

import logging
import re
from typing import List, Tuple

from cliver.llm.media_generation.base import ImageGenerationHelper

logger = logging.getLogger(__name__)

_REGISTRY: List[Tuple[re.Pattern, type]] = []
_INITIALIZED = False


def register_helper(url_pattern: str, helper_class: type) -> None:
    _REGISTRY.append((re.compile(url_pattern, re.IGNORECASE), helper_class))


def get_image_helper(image_url: str) -> ImageGenerationHelper:
    _ensure_registered()
    for pattern, cls in _REGISTRY:
        if pattern.search(image_url):
            return cls()
    from cliver.llm.media_generation.openai_compat import OpenAIImageHelper

    return OpenAIImageHelper()


def _ensure_registered():
    global _INITIALIZED
    if _INITIALIZED:
        return
    _INITIALIZED = True
    from cliver.llm.media_generation.minimax import MiniMaxImageHelper

    register_helper(r"minimax", MiniMaxImageHelper)
