"""Zhipu GLM: OpenAI-compatible with parameter constraints."""

from __future__ import annotations

from cliver.provider.providers import _EngineProvider


class GLMProvider(_EngineProvider):
    """Zhipu / GLM provider."""

    supported_protocols = ["openai"]

    UNSUPPORTED_PARAMS = {
        "frequency_penalty",
        "presence_penalty",
        "logit_bias",
    }

    def filter_options(self, options: dict) -> dict:
        return {k: v for k, v in options.items() if k not in self.UNSUPPORTED_PARAMS}
