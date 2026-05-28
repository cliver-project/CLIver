"""Event models for monitoring LLM inference and tool execution."""

from __future__ import annotations

import time
from enum import Enum
from typing import Any, Awaitable, Callable

from pydantic import BaseModel, Field


class InferenceEventType(str, Enum):
    """Events fired during LLM inference (by ProtocolEngine)."""

    STARTED = "inference.started"
    FIRST_TOKEN = "inference.first_token"  # TTFT (time-to-first-token)
    CHUNK = "inference.chunk"  # content delta
    TOOL_CALL = "inference.tool_call"  # tool call block detected
    COMPLETED = "inference.completed"  # full response done
    ERROR = "inference.error"


class ToolEventType(str, Enum):
    """Events fired during tool execution (by AgentCore)."""

    START = "tool.start"
    END = "tool.end"
    ERROR = "tool.error"
    PERMISSION_DENIED = "tool.permission_denied"


class InferenceEvent(BaseModel):
    """An event from the LLM inference lifecycle."""

    event: InferenceEventType
    model: str
    provider: str
    request_id: str = Field(default_factory=lambda: "")
    data: dict[str, Any] | None = None
    # Event-specific payloads:
    #   STARTED:    {"message_count": 5, "tool_count": 12}
    #   FIRST_TOKEN: {"latency_ms": 342}
    #   CHUNK:      {"content_delta": "Hello"} (optional, for monitors)
    #   TOOL_CALL:  {"tool_name": "Read", "tool_call_id": "call_1"}
    #   COMPLETED:  {"total_latency_ms": 1200, "usage": {...}}
    #   ERROR:      {"error": "rate limited", "retryable": true}
    timestamp: float = Field(default_factory=time.monotonic)


class ToolEvent(BaseModel):
    """An event from tool execution."""

    event: ToolEventType
    tool_name: str  # "Read" or "my_server#my_tool"
    tool_call_id: str
    args: dict[str, Any] | None = None
    result: str | None = None  # TOOL_END: truncated result text
    error: str | None = None  # TOOL_ERROR: error message
    duration_ms: float | None = None
    timestamp: float = Field(default_factory=time.monotonic)


EventHandler = Callable[[InferenceEvent | ToolEvent], Awaitable[None]]
