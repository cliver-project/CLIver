from dataclasses import dataclass, field


@dataclass
class CallContext:
    """Per-call mutable state for AgentCore — isolated per process_user_input invocation."""

    tool_call_count: int = 0
    generated_media: list = field(default_factory=list)
    tool_result_cache: dict[str, list] = field(default_factory=dict)
