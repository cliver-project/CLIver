"""Qwen chat (Alibaba Model Studio / Bailian) over its OpenAI-compatible endpoint.

Qwen models serve chat through ``/compatible-mode/v1`` on the DashScope host,
so the OpenAI protocol provider covers them unchanged — this brand entry
exists to pin that endpoint and the conventional ``DASHSCOPE_API_KEY``.
Native media generation is the separate ``dashscope`` brand.
"""

from __future__ import annotations

from cliver_core.provider.providers.openai import OpenAIProvider


class QwenProvider(OpenAIProvider):
    """Qwen chat on DashScope's OpenAI-compatible endpoint (bearer auth).

    Keys are region-bound: an API key from another region (e.g. the
    Singapore endpoint) needs an explicit ``base_url`` override.
    """

    brand = "qwen"
    default_base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
