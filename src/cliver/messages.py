"""Canonical message models for CLIver"""

from __future__ import annotations

import json
import logging
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ToolCall(BaseModel):
    """A tool call within an assistant message."""

    id: str
    name: str  # "Read" or "my_server#my_tool"
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


class CLIverMessage(BaseModel):
    """Canonical message — the message type AgentCore sees.

    Covers all roles (system, user, assistant, tool) and both
    text-only and multimodal content.  Each provider converts between
    this type and its native SDK types.
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

    vendor_ext: dict[str, Any] = Field(default_factory=dict)
    # Provider-specific metadata, preserved opaquely across round-trips.
    # Examples:
    #   {"reasoning_content": "..."}            — DeepSeek thinking
    #   {"thinking": "..."}                     — Anthropic extended thinking
    #   {"anthropic_content_blocks": [...]}     — raw content blocks for round-trip
    # AgentCore never inspects this; each Provider reads what it wrote.

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


class CLIverMessageChunk(BaseModel):
    """Streaming chunk — yielded during Provider.stream().

    content:        text delta (accumulate across chunks for full text).
    tool_call_chunks: incremental tool call info (accumulate by index).
    vendor_ext:     provider-specific deltas keyed by name
                    (e.g. {"reasoning_content": "..."}).
                    These are string deltas — the accumulator concatenates them.
    """

    content: str | None = None
    tool_call_chunks: list[ToolCallChunk] | None = None
    vendor_ext: dict[str, str] = Field(default_factory=dict)
    # String deltas — accumulated key-by-key in AgentCore.
    # {"reasoning_content": "I should...", "thinking": "Let me analyze..."}

    def model_dump(self, **kwargs) -> dict:
        kwargs.setdefault("exclude_none", True)
        return super().model_dump(**kwargs)


class UsageInfo(BaseModel):
    """Token usage information from an LLM response."""

    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int | None = None


# ── Tool call accumulation helper (used by AgentCore during streaming) ──


class ToolCallAccumulator:
    """Merges ToolCallChunk deltas into complete ToolCall objects.

    Handles the streaming pattern where tool calls arrive incrementally:
    chunks at the same index accumulate args_delta fragments.
    """

    def __init__(self):
        self._by_index: dict[int, dict] = {}

    def feed(self, chunk: ToolCallChunk) -> None:
        entry = self._by_index.setdefault(
            chunk.index, {"id": None, "name": None, "args_parts": []}
        )
        if chunk.id is not None:
            entry["id"] = chunk.id
        if chunk.name is not None:
            entry["name"] = chunk.name
        if chunk.args_delta is not None:
            entry["args_parts"].append(chunk.args_delta)

    def finalize(self) -> list[ToolCall]:
        result = []
        for _idx, entry in sorted(self._by_index.items()):
            args_str = "".join(entry["args_parts"])
            try:
                args = json.loads(args_str) if args_str else {}
            except json.JSONDecodeError:
                logger.warning(
                    "Failed to parse tool call args as JSON: %s", args_str[:200]
                )
                args = {}
            result.append(
                ToolCall(id=entry["id"] or "", name=entry["name"] or "", args=args)
            )
        return result
