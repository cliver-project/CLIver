"""Zhipu GLM: OpenAI- and Anthropic-compatible endpoints."""

from __future__ import annotations

from cliver_core.provider.providers.anthropic import AnthropicProvider
from cliver_core.provider.providers.openai import OpenAIProvider


class GLMProvider(OpenAIProvider):
    """Zhipu / GLM provider (OpenAI-compatible endpoint)."""

    brand = "glm"
    default_base_url = "https://open.bigmodel.cn/api/paas/v4"


class GLMAnthropicProvider(AnthropicProvider):
    """Zhipu GLM Anthropic-compatible endpoint (Claude Code-style base URL)."""

    brand = "glm"
    default_base_url = "https://open.bigmodel.cn/api/anthropic"
    # This endpoint accepts both Bearer and x-api-key; Bearer matches the
    # other third-party Anthropic-compatible endpoints (DeepSeek, MiniMax).
    _anthropic_use_bearer_auth = True
