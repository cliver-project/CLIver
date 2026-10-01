"""Generic Anthropic-compatible provider."""

from __future__ import annotations

from cliver_core.provider.providers import _EngineProvider


class AnthropicProvider(_EngineProvider):
    """Generic Anthropic-compatible provider. No special handling needed."""

    supported_protocols = ["anthropic"]
    default_base_url = "https://api.anthropic.com"
