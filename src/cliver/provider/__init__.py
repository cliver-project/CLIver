"""Provider interface and request/response models."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator

from pydantic import BaseModel, Field

from cliver.messages import CLIverMessage, CLIverMessageChunk, UsageInfo
from cliver.tool import CLIverTool

logger = logging.getLogger(__name__)


class CLIverRequest(BaseModel):
    """A request to an LLM provider."""

    messages: list[CLIverMessage]
    tools: list[CLIverTool] | None = None
    model: str
    options: dict[str, Any] = Field(default_factory=dict)
    # temperature, top_p, max_tokens, thinking, etc.
    # Passed through — each engine filters what it supports.


class CLIverResponse(BaseModel):
    """A response from an LLM provider."""

    message: CLIverMessage
    usage: UsageInfo | None = None


class MessageConverter(ABC):
    """Converts CLIverMessage to native provider format.

    Shared by both ProtocolEngine (protocol-level conversion) and
    Provider (brand-specific injection on top of the engine).
    """

    @abstractmethod
    def msg_to_native(self, msg: CLIverMessage) -> Any:
        """Convert CLIverMessage → native format for the target protocol."""
        ...


class Provider(MessageConverter):
    """Interface for LLM inference.

    A Provider wraps a specific brand (DeepSeek, MiniMax, OpenAI, ...)
    and delegates to a ProtocolEngine (OpenAI or Anthropic) for the
    actual API calls and message conversion.

    Subclasses MUST define ``supported_protocols`` — a list of protocol
    names this provider supports (e.g. ``["openai", "anthropic"]``).
    The check runs at class-definition time, so a missing declaration
    is caught at import, not at the first API call.
    """

    supported_protocols: list[str]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Only validate classes that declare supported_protocols in their
        # own __dict__ (concrete providers).  Intermediate abstract bases
        # (e.g. _EngineProvider) skip the check and let subclasses define it.
        if "supported_protocols" in cls.__dict__:
            if not cls.supported_protocols:
                raise TypeError(
                    f"{cls.__name__} must define supported_protocols "
                    f"(e.g. supported_protocols = ['openai', 'anthropic']), "
                    f"not an empty list"
                )

    def __init__(self, protocol: str, api_key: str, base_url: str):
        if protocol not in self.supported_protocols:
            raise ValueError(
                f"Provider '{self.provider_name()}' does not support protocol '{protocol}'. "
                f"Supported: {', '.join(self.supported_protocols)}"
            )
        self.protocol = protocol
        self.api_key = api_key
        self.base_url = base_url

    @classmethod
    def provider_name(cls) -> str:
        """Short name for logging/registration."""
        return cls.__name__.removesuffix("Provider").lower()

    # ── Hooks (subclasses override for brand-specific behavior) ──

    def on_response(self, response: CLIverResponse) -> CLIverResponse:
        """Post-process a response. Override to extract vendor_ext fields."""
        return response

    def on_chunk(self, chunk: CLIverMessageChunk) -> CLIverMessageChunk:
        """Post-process a streaming chunk. Override to remap vendor_ext keys."""
        return chunk

    def filter_options(self, options: dict[str, Any]) -> dict[str, Any]:
        """Filter/transform options before passing to the engine."""
        return options

    # ── Public API ─────────────────────────────────────────

    @abstractmethod
    async def chat(self, request: CLIverRequest) -> CLIverResponse: ...

    @abstractmethod
    async def stream(
        self, request: CLIverRequest
    ) -> AsyncIterator[CLIverMessageChunk]: ...
