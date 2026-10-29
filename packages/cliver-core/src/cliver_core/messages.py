"""Canonical message models for CLIver.

The message owns its per-message wire shape for the two supported
protocols (OpenAI / Anthropic): ``to_native`` / ``from_native`` convert
between the agnostic fields and the native message dict, with the native
structure kept readable on ``CLIverMessage.native``.  Conversation-level
shape (system split, tool_result merge, tool-name sanitize) stays in the
providers.
"""

from __future__ import annotations

import copy
import json
import logging
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_SUPPORTED_PROTOCOLS = ("openai", "anthropic")


class ToolCall(BaseModel):
    """A tool call within an assistant message."""

    id: str
    name: str  # "Read" or "mcp__my_server__my_tool"
    args: dict[str, Any] = Field(default_factory=dict)


class ToolCallChunk(BaseModel):
    """Incremental tool call info during streaming.

    Providers stream tool calls in pieces:
    - First chunk: has `name` (and possibly `id`)
    - Middle chunks: `args_delta` (partial JSON fragment)
    - Last chunk: the final args_delta completes the JSON
    - Some providers also set `id` in the first chunk
    """

    index: int = 0
    id: str | None = None
    name: str | None = None
    args_delta: str | None = None  # partial JSON, accumulated by caller


def _tool_call_from_openai(raw: dict[str, Any]) -> ToolCall:
    """Parse one OpenAI ``tool_calls`` entry (function wrapper, JSON-string args)."""
    function = raw.get("function") or {}
    try:
        args = json.loads(function.get("arguments") or "{}")
    except json.JSONDecodeError:
        logger.warning("Failed to parse tool call arguments: %s", function.get("arguments"))
        args = {}
    return ToolCall(id=raw.get("id") or "", name=function.get("name") or "", args=args)


class CLIverMessage(BaseModel):
    """Canonical message — the message type AgentCore sees.

    Covers all roles (system, user, assistant, tool) and both text-only
    and multimodal content.  A thin wrapper around the native message:
    when a message originates from a provider (or is finalized from a
    stream), the native wire dict stays readable on ``native`` and
    ``to_native`` passes it through unchanged — vendor extras
    (``reasoning_content``, thinking ``signature``, ...) round-trip
    without any per-brand code.

    Construction is unique per source:
    - agnostic kwargs (``CLIverMessage(role=..., content=...)``) create
      request messages — protocol-free until ``to_native`` is called;
    - ``from_native(native, protocol)`` wraps any native message dict —
      including streamed turns: the SDK-aggregated native message arrives
      on the stream's terminal chunk and wraps with this same call.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[dict] | None = None
    # - str:  plain text (most common)
    # - list[dict]: multimodal content blocks (OpenAI-style or Anthropic-style)
    # - None: assistant message that only contains tool_calls

    tool_calls: list[ToolCall] | None = None
    # Set on assistant messages that request tool invocation.
    # Each ToolCall has id, name, and args.

    tool_call_id: str | None = None
    # Set on tool-result messages to link back to the ToolCall.

    native: dict[str, Any] | None = None
    # The native wire dict this message was built from (when there is one).
    # Vendor extras live here — read them from ``native``; they are NOT
    # mirrored onto the agnostic fields.

    native_protocol: str | None = None
    # Which protocol ``native`` speaks ("openai" / "anthropic").

    trace: list[dict[str, Any]] | None = None
    # Loop diagnostics (AgentCore's Re-Act trace — one entry per iteration),
    # set on final messages.

    # ── Native construction (response side) ───────────────────

    @classmethod
    def from_native(cls, native: dict[str, Any], protocol: str) -> "CLIverMessage":
        """Wrap a native message dict — THE native constructor, any protocol.

        Fields (content, tool_calls, tool_call_id) derive from the native
        structure's JSON form; ``native`` itself is kept verbatim on the
        message for passthrough round-trips and vendor-extra reads.
        """
        if protocol == "openai":
            return cls._from_native_openai(native)
        if protocol == "anthropic":
            return cls._from_native_anthropic(native)
        raise ValueError(f"Unknown protocol '{protocol}'. Supported: {list(_SUPPORTED_PROTOCOLS)}")

    @classmethod
    def _from_native_openai(cls, native: dict[str, Any]) -> "CLIverMessage":
        tool_calls: list[ToolCall] | None = None
        raw_calls = native.get("tool_calls")
        if raw_calls:
            tool_calls = [_tool_call_from_openai(raw) for raw in raw_calls]

        content = native.get("content")
        if tool_calls and not content:
            content = None  # assistant message that only carries tool_calls

        return cls(
            role=native.get("role") or "assistant",
            content=content,
            tool_calls=tool_calls,
            tool_call_id=native.get("tool_call_id"),
            native=native,
            native_protocol="openai",
        )

    @classmethod
    def _from_native_anthropic(cls, native: dict[str, Any]) -> "CLIverMessage":
        blocks = native.get("content")
        if isinstance(blocks, str):  # plain-string user content
            return cls(
                role=native.get("role") or "user",
                content=blocks,
                native=native,
                native_protocol="anthropic",
            )

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        tool_results: list[dict[str, Any]] = []
        for block in blocks or []:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "text":
                text_parts.append(block.get("text") or "")
            elif block_type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.get("id") or "",
                        name=block.get("name") or "",
                        args=dict(block.get("input") or {}),
                    )
                )
            elif block_type == "tool_result":
                tool_results.append(block)

        if tool_results:  # a user message carrying tool_result blocks
            first = tool_results[0]
            result_content = first.get("content")
            if isinstance(result_content, list):
                result_content = "".join(b.get("text") or "" for b in result_content if isinstance(b, dict))
            return cls(
                role="tool",
                content=result_content if result_content else None,
                tool_call_id=first.get("tool_use_id"),
                native=native,
                native_protocol="anthropic",
            )

        return cls(
            role=native.get("role") or "assistant",
            content="\n".join(text_parts) if text_parts else None,
            tool_calls=tool_calls if tool_calls else None,
            native=native,
            native_protocol="anthropic",
        )

    # ── Native conversion (request side) ──────────────────────

    def to_native(self, protocol: str) -> dict[str, Any]:
        """The message's native wire dict for *protocol*.

        Passthrough — a deep copy of ``native`` — when the message
        already carries a native form of that protocol (vendor extras
        included, byte-identical wire shape); otherwise built from the
        agnostic fields.
        """
        if protocol not in _SUPPORTED_PROTOCOLS:
            raise ValueError(f"Unknown protocol '{protocol}'. Supported: {list(_SUPPORTED_PROTOCOLS)}")
        if self.native is not None and self.native_protocol == protocol:
            return copy.deepcopy(self.native)
        if protocol == "openai":
            return self._build_native_openai()
        return self._build_native_anthropic()

    def _build_native_openai(self) -> dict[str, Any]:
        native: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            native["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.args, ensure_ascii=False)},
                }
                for tc in self.tool_calls
            ]
        if self.tool_call_id:
            native["tool_call_id"] = self.tool_call_id
        return native

    def _build_native_anthropic(self) -> dict[str, Any]:
        if self.role == "tool":
            return {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": self.tool_call_id,
                        "content": self.content or "",
                    }
                ],
            }

        content: list[dict[str, Any]] = []
        if isinstance(self.content, str):
            if self.content:
                content.append({"type": "text", "text": self.content})
        elif isinstance(self.content, list):
            content = list(self.content)

        if self.tool_calls:
            content.extend(
                {"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.args} for tc in self.tool_calls
            )

        return {"role": self.role, "content": content}

    # ── Convenience properties ────────────────────────────────

    @property
    def text(self) -> str | None:
        """Return content as a plain string, or None if multimodal/empty."""
        if isinstance(self.content, str):
            return self.content
        return None

    @property
    def is_multimodal(self) -> bool:
        """True if content is a list of content blocks."""
        return isinstance(self.content, list)

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    @property
    def is_tool_result(self) -> bool:
        return self.role == "tool"

    def model_dump(self, **kwargs) -> dict:
        """Exclude None fields (content, tool_calls, tool_call_id) from output."""
        kwargs.setdefault("exclude_none", True)
        return super().model_dump(**kwargs)


class UsageInfo(BaseModel):
    """Token usage information from an LLM response."""

    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int | None = None

    @property
    def total_tokens(self) -> int:
        """Billable input + output tokens.

        Cache reads are a subset of ``input_tokens``, not additional tokens.
        """
        return self.input_tokens + self.output_tokens


class CLIverMessageChunk(BaseModel):
    """Streaming chunk — yielded during Provider.stream().

    content:        text delta (accumulate across chunks for live display).
    tool_call_chunks: incremental tool call info (observability).
    vendor_ext:     provider-specific deltas keyed by name
                    (e.g. {"reasoning_content": "..."}).
                    These are string deltas — consumers concatenate them.
    native:         the stream's aggregated native assistant message
                    (SDK-finalized) — set on the terminal chunk only;
                    ``None`` on ordinary delta chunks.  This is the
                    streamed turn's message source.
    usage:          token usage — set on the terminal chunk when the
                    stream carries it.
    """

    content: str | None = None
    tool_call_chunks: list[ToolCallChunk] | None = None
    vendor_ext: dict[str, str] = Field(default_factory=dict)
    # String deltas — vendor-specific keys the SDK surfaced
    # {"reasoning_content": "I should...", "thinking": "Let me analyze..."}
    native: dict[str, Any] | None = None
    usage: UsageInfo | None = None

    def model_dump(self, **kwargs) -> dict:
        kwargs.setdefault("exclude_none", True)
        return super().model_dump(**kwargs)
