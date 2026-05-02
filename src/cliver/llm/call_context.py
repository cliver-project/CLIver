from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Optional

_current_call_context: ContextVar[Optional["CallContext"]] = ContextVar("_current_call_context", default=None)


@dataclass
class CallContext:
    """Per-call mutable state for AgentCore — isolated per process_user_input invocation.

    Each call to process_user_input() or stream_user_input() creates a fresh
    instance so concurrent conversations do not share mutable state.

    Use ``get_current()`` from builtin tools to access the active context
    (e.g. so ImageGenerateTool can accumulate media on the outer loop's context).
    """

    tool_call_count: int = 0
    """Number of tool calls executed — used by skill auto-learning to gauge complexity."""

    generated_media: list = field(default_factory=list)
    """Accumulated media from image generation, attached to the final AI response."""

    tool_result_cache: dict[str, list] = field(default_factory=dict)
    """Dedup cache: maps (tool_name, args) hash to cached result list."""

    allowed_tools: Optional[set[str]] = None
    """When set, only these tool names may execute. Others return an error."""

    def activate(self) -> None:
        """Set this context as the current one (via ContextVar)."""
        _current_call_context.set(self)

    @staticmethod
    def get_current() -> Optional["CallContext"]:
        """Return the active CallContext, or None if outside a Re-Act loop."""
        return _current_call_context.get()
