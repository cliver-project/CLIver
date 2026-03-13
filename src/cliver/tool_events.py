"""
Tool execution event types for CLIver.

These events are emitted by TaskExecutor during tool execution.
The CLI layer can subscribe to these events to render progress
displays. API consumers can provide their own handler or ignore them.

This module has NO dependencies on CLI frameworks (Click, Rich, etc.).
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Optional


class ToolEventType(str, Enum):
    """Types of tool execution events."""

    TOOL_START = "tool_start"
    TOOL_END = "tool_end"
    TOOL_ERROR = "tool_error"


@dataclass
class ToolEvent:
    """A single tool execution event.

    Attributes:
        event_type: The type of event (start, end, error)
        tool_name: The full tool name (e.g., "read_file" or "filesystem#list")
        tool_call_id: Unique identifier for this tool call
        args: The arguments passed to the tool (for start events)
        result: The result string (for end events)
        error: The error message (for error events)
        duration_ms: Execution duration in milliseconds (for end/error events)
    """

    event_type: ToolEventType
    tool_name: str
    tool_call_id: str = ""
    args: Optional[Dict[str, Any]] = None
    result: Optional[str] = None
    error: Optional[str] = None
    duration_ms: Optional[float] = None


# Type alias for the event handler callback
ToolEventHandler = Callable[[ToolEvent], None]
