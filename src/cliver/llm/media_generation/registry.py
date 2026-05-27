"""Helper registry — selects a media generation helper by provider + category."""

import logging
from typing import Dict, Tuple

from cliver.llm.media_generation.base import MediaGenerationHelper

logger = logging.getLogger(__name__)

# (provider, category) → helper class
_REGISTRY: Dict[Tuple[str, str], type] = {}
# provider → default helper class (used when no category-specific match)
_DEFAULTS: Dict[str, type] = {}
_INITIALIZED = False


def register_helper(provider: str, category: str, helper_class: type) -> None:
    """Register a helper for a specific provider + category combination."""
    _REGISTRY[(provider.lower(), category)] = helper_class


def register_default(provider: str, helper_class: type) -> None:
    """Register a fallback helper for a provider (any category not explicitly registered)."""
    _DEFAULTS[provider.lower()] = helper_class


def get_media_helper(provider: str, category: str = "image") -> MediaGenerationHelper:
    """Return a media generation helper for the given provider and category.

    Looks up (provider, category) first, then provider default, then
    falls back to OpenAI-compatible for unknown providers.
    """
    _ensure_registered()
    key = (provider.lower(), category)
    if key in _REGISTRY:
        return _REGISTRY[key]()
    default = _DEFAULTS.get(provider.lower())
    if default:
        return default()
    from cliver.llm.media_generation.openai_compat import OpenAIImageHelper

    return OpenAIImageHelper()


# Backward-compatible alias (tests use this)
def get_image_helper(image_url: str) -> MediaGenerationHelper:
    """Legacy alias — finds a helper by provider name substring in the URL."""
    _ensure_registered()
    url_lower = image_url.lower()
    for (prov, _), cls in _REGISTRY.items():
        if prov in url_lower:
            return cls()
    for prov, cls in _DEFAULTS.items():
        if prov in url_lower:
            return cls()
    from cliver.llm.media_generation.openai_compat import OpenAIImageHelper

    return OpenAIImageHelper()


def _ensure_registered():
    global _INITIALIZED
    if _INITIALIZED:
        return
    _INITIALIZED = True
    from cliver.llm.media_generation.minimax import MiniMaxAudioHelper, MiniMaxImageHelper, MiniMaxVideoHelper
    from cliver.llm.media_generation.openai_compat import OpenAIAudioHelper, OpenAIImageHelper, OpenAIVideoHelper

    register_helper("minimax", "image", MiniMaxImageHelper)
    register_helper("minimax", "audio", MiniMaxAudioHelper)
    register_helper("minimax", "video", MiniMaxVideoHelper)
    register_default("minimax", MiniMaxImageHelper)

    register_helper("openai", "image", OpenAIImageHelper)
    register_helper("openai", "audio", OpenAIAudioHelper)
    register_helper("openai", "video", OpenAIVideoHelper)
    register_default("openai", OpenAIImageHelper)
