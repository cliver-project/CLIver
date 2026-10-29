"""Event models for monitoring LLM inference and tool execution."""

from __future__ import annotations

import inspect
import time
from enum import Enum
from typing import Any, Awaitable, Callable

from pydantic import BaseModel, Field


class InferenceEventType(str, Enum):
    """Events fired during LLM inference (by the protocol providers)."""

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
    tool_name: str  # "Read" or "mcp__my_server__my_tool"
    tool_call_id: str
    args: dict[str, Any] | None = None
    result: str | None = None  # TOOL_END: truncated result text
    error: str | None = None  # TOOL_ERROR: error message
    duration_ms: float | None = None
    timestamp: float = Field(default_factory=time.monotonic)


class PipelineEventType(str, Enum):
    """Events fired during a SkillPipeline run (by ``cliver_core.pipeline``)."""

    START = "pipeline.start"
    STEP_START = "pipeline.step_start"
    STEP_END = "pipeline.step_end"
    STEP_ERROR = "pipeline.step_error"
    END = "pipeline.end"


class PipelineEvent(BaseModel):
    """An event from the pipeline / step lifecycle.

    Tool events raised inside a step are forwarded unchanged as
    :class:`ToolEvent`; this model covers the pipeline's own lifecycle.
    """

    event: PipelineEventType
    step_name: str | None = None
    step_index: int | None = None
    data: dict[str, Any] | None = None
    #   START:      {"step_count": 3}
    #   STEP_END:   {"content_chars": 1200, "usage": {...}}
    #   END:        {"ok": true, "stopped": false, "steps_run": 3}
    #   END (cancel): {"cancelled": true, "steps_run": 1}
    error: str | None = None  # STEP_ERROR: error message
    duration_ms: float | None = None
    timestamp: float = Field(default_factory=time.monotonic)


#: An event subscriber.  Sync and async callables are both accepted; a sync
#: handler runs inline.  Handler exceptions are a **contract**: they are not
#: swallowed.  A raise from ``provider.chat()`` / ``provider.stream()``
#: propagates unchanged (and a raise on ``inference.started`` happens before
#: the transport call); AgentCore does not catch or wrap it either.
EventHandler = Callable[
    [InferenceEvent | ToolEvent | PipelineEvent],
    Awaitable[None] | None,
]


async def dispatch_event(
    handler: EventHandler | None,
    event: InferenceEvent | ToolEvent | PipelineEvent,
) -> None:
    """Invoke *handler* with *event* — sync or async, exceptions unchanged."""
    if handler is None:
        return
    result = handler(event)
    if inspect.isawaitable(result):
        await result
