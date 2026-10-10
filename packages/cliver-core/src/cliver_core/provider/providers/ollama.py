"""Ollama: local OpenAI-compatible provider."""

from __future__ import annotations

from cliver_core.provider.providers.openai import OpenAIProvider


class OllamaProvider(OpenAIProvider):
    """Ollama provider (local, no API key needed)."""

    brand = "ollama"
    default_base_url = "http://localhost:11434/v1"
    # Local server: no generation endpoints — media purposes report
    # unsupported up front.  Embeddings keep the protocol default
    # (``/v1/embeddings`` works); the opt-out is media-scoped.
    _standard_media_fallback = False

    UNSUPPORTED_PARAMS = {
        "frequency_penalty",
        "presence_penalty",
        "logit_bias",
        "reasoning_effort",
        "parallel_tool_calls",
    }

    def filter_options(self, options: dict) -> dict:
        return {k: v for k, v in options.items() if k not in self.UNSUPPORTED_PARAMS}
