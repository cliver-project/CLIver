"""Shared exception types.

Deliberately dependency-free — both the provider layer and the bindings layer
raise :class:`CapabilityNotSupported`, and ``bindings`` imports ``provider``
(not the other way round), so the exception lives here.
"""

from __future__ import annotations

__all__ = ["CapabilityNotSupported", "ProviderError", "RunTimeoutError"]


class CapabilityNotSupported(Exception):
    """A capability (image / video / tts / asr / embed / rerank) is not available.

    Raised when either

    * AgentCore has no model bound for the purpose (``bindings.<purpose>``), or
    * the bound provider/brand does not implement the purpose (e.g. image
      generation on Anthropic, TTS on MiniMax).
    """


class RunTimeoutError(TimeoutError):
    """The run exceeded its run-level deadline (``run_timeout``).

    Raised by ``AgentCore.chat()`` / ``stream()`` / ``run_skill()`` and
    ``chat_structured()`` when the wall-clock budget expires.  It is a plain
    ``TimeoutError``, deliberately not a :class:`ProviderError`: no failover
    is attempted, and the application decides whether to retry.
    ``CancelledError`` from the caller is never converted into this error.
    """

    def __init__(self, timeout: float, run_id: str | None = None) -> None:
        self.timeout = timeout
        self.run_id = run_id
        super().__init__(
            f"Run '{run_id}' timed out after {timeout:g}s" if run_id else f"Run timed out after {timeout:g}s"
        )


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
