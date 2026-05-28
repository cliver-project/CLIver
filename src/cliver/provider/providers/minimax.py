"""MiniMax: strips unsupported params, normalises thinking key."""

from __future__ import annotations

from cliver.messages import CLIverMessageChunk
from cliver.provider.providers import _EngineProvider


class MiniMaxProvider(_EngineProvider):
    """MiniMax provider.

    Filters out params MiniMax doesn't support, and remaps the
    Anthropic-native ``thinking`` key to the canonical ``reasoning_content``
    key in streaming chunks.
    """

    supported_protocols = ["openai", "anthropic"]

    UNSUPPORTED_PARAMS = {
        "frequency_penalty",
        "presence_penalty",
        "logprobs",
        "logit_bias",
        "parallel_tool_calls",
    }

    def filter_options(self, options: dict) -> dict:
        return {k: v for k, v in options.items() if k not in self.UNSUPPORTED_PARAMS}

    def on_chunk(self, chunk: CLIverMessageChunk) -> CLIverMessageChunk:
        if "thinking" in chunk.vendor_ext:
            chunk.vendor_ext["reasoning_content"] = chunk.vendor_ext.pop("thinking")
        return chunk
