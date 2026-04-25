from dataclasses import dataclass, field


@dataclass
class CallContext:
    """Per-call mutable state for AgentCore — isolated per process_user_input invocation.

    Each call to process_user_input() or stream_user_input() creates a fresh
    instance so concurrent conversations do not share mutable state.
    """

    tool_call_count: int = 0
    """Number of tool calls executed — used by skill auto-learning to gauge complexity."""

    generated_media: list = field(default_factory=list)
    """Accumulated media from image generation, attached to the final AI response."""

    tool_result_cache: dict[str, list] = field(default_factory=dict)
    """Dedup cache: maps (tool_name, args) hash to cached result list."""
