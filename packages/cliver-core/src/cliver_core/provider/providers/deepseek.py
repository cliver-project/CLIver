"""DeepSeek provider classes (OpenAI and Anthropic endpoints).

DeepSeek ships no SDK: both leaves are plain protocol providers carrying just
the brand identity and the endpoint default URL — no behavioral overrides.
``reasoning_content`` round-trips through ``CLIverMessage.native``: the field
sits in the native payload on responses and passes through unchanged when the
message is echoed back on the next request.
"""

from __future__ import annotations

from cliver_core.provider.providers.anthropic import AnthropicProvider
from cliver_core.provider.providers.openai import OpenAIProvider


class DeepSeekOpenAIProvider(OpenAIProvider):
    """DeepSeek OpenAI-compatible endpoint."""

    brand = "deepseek"
    default_base_url = "https://api.deepseek.com/v1"
    # DeepSeek serves chat only — no media endpoints.  Media purposes report
    # unsupported up front instead of posting to the wire-standard paths; an
    # explicit binding ``path=`` still opts a purpose in.
    _standard_media_fallback = False


class DeepSeekAnthropicProvider(AnthropicProvider):
    """DeepSeek Anthropic-compatible endpoint."""

    brand = "deepseek"
    default_base_url = "https://api.deepseek.com/anthropic"
    # This endpoint accepts both Bearer and x-api-key (verified live); Bearer
    # matches the other third-party Anthropic-compatible endpoints.
    _anthropic_use_bearer_auth = True
