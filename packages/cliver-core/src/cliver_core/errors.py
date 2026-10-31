"""Shared exception types.

Deliberately dependency-free — both the provider layer and the bindings layer
raise :class:`CapabilityNotSupported`, and ``bindings`` imports ``provider``
(not the other way round), so the exception lives here.
"""

from __future__ import annotations

__all__ = ["CapabilityNotSupported", "ProviderError"]


class CapabilityNotSupported(Exception):
    """A capability (image / video / tts / asr / embed / rerank) is not available.

    Raised when either

    * AgentCore has no model bound for the purpose (``bindings.<purpose>``), or
    * the bound provider/brand does not implement the purpose (e.g. image
      generation on Anthropic, TTS on MiniMax).
    """


class ProviderError(RuntimeError):
    """A model-provider failure — HTTP, SDK, connection, or timeout.

    Protocol providers raise this at their transport boundary with the
    vendor's exception preserved as ``__cause__``.  ``retryable`` is a
    best-effort classification (rate limits, timeouts, connection failures,
    and 5xx responses are retryable); consumers may map it into their own
    error taxonomy.  AgentCore re-raises it unchanged unless another binding
    of the same purpose can take over a retryable failure.

    It subclasses :class:`RuntimeError` for backward compatibility: the
    generic embed/rerank endpoints raised RuntimeError before this contract
    existed, and their callers keep catching those failures.
    """

    retryable: bool = False

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable
