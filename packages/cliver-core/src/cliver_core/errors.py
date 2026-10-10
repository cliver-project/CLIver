"""Shared exception types.

Deliberately dependency-free — both the provider layer and the bindings layer
raise :class:`CapabilityNotSupported`, and ``bindings`` imports ``provider``
(not the other way round), so the exception lives here.
"""

from __future__ import annotations

__all__ = ["CapabilityNotSupported"]


class CapabilityNotSupported(Exception):
    """A capability (image / video / tts / asr / embed / rerank) is not available.

    Raised when either

    * AgentCore has no model bound for the purpose (``bindings.<purpose>``), or
    * the bound provider/brand does not implement the purpose (e.g. image
      generation on Anthropic, TTS on MiniMax).
    """
