"""Generic OpenAI-compatible provider."""

from __future__ import annotations

from cliver.provider.providers import _EngineProvider


class OpenAIProvider(_EngineProvider):
    """Generic OpenAI-compatible provider. No special handling needed."""

    supported_protocols = ["openai"]
    default_base_url = "https://api.openai.com/v1"
