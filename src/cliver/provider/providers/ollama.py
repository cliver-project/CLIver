"""Ollama: local OpenAI-compatible provider."""

from __future__ import annotations

from cliver.provider.providers import _EngineProvider


class OllamaProvider(_EngineProvider):
    """Ollama provider (local, no API key needed)."""

    supported_protocols = ["openai"]
    default_base_url = "http://localhost:11434/v1"

    UNSUPPORTED_PARAMS = {
        "frequency_penalty",
        "presence_penalty",
        "logit_bias",
        "reasoning_effort",
        "parallel_tool_calls",
    }

    def filter_options(self, options: dict) -> dict:
        return {k: v for k, v in options.items() if k not in self.UNSUPPORTED_PARAMS}
