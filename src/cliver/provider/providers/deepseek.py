"""DeepSeek: reasoning_content round-tripping on every assistant message."""

from __future__ import annotations

from cliver.messages import CLIverMessage, CLIverMessageChunk
from cliver.provider import CLIverResponse
from cliver.provider.providers import _EngineProvider


class DeepSeekProvider(_EngineProvider):
    """DeepSeek provider.

    Injects ``reasoning_content`` into every assistant message sent to
    the API, and extracts it from responses into ``vendor_ext`` so it
    round-trips correctly across multi-turn conversations.
    """

    supported_protocols = ["openai", "anthropic"]
    default_base_url = "https://api.deepseek.com/v1"

    def msg_to_native(self, msg: CLIverMessage) -> dict:
        native = self.engine.msg_to_native(msg)
        if msg.role == "assistant" and "reasoning_content" in msg.vendor_ext:
            native["reasoning_content"] = msg.vendor_ext["reasoning_content"]
        return native

    def on_response(self, response: CLIverResponse) -> CLIverResponse:
        rc = response.message.vendor_ext.get("reasoning_content")
        if rc:
            response.message.vendor_ext["reasoning_content"] = rc
        return response

    def on_chunk(self, chunk: CLIverMessageChunk) -> CLIverMessageChunk:
        return chunk
