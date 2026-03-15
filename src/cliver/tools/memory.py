"""Built-in memory tools for persistent knowledge across sessions.

The LLM uses these tools to save and recall knowledge that persists
beyond the current conversation. Memory is managed by AgentProfile
and stored as markdown files on disk.
"""

import logging
from typing import Literal, Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from cliver.agent_profile import get_current_profile

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# memory_read
# ---------------------------------------------------------------------------


class MemoryReadInput(BaseModel):
    """Input schema for memory_read."""

    pass


class MemoryReadTool(BaseTool):
    """Read persistent memory to recall knowledge from previous sessions."""

    name: str = "memory_read"
    description: str = (
        "Read your persistent memory to recall knowledge from previous conversations. "
        "Returns both global memory (shared across all agents) and your agent-specific memory. "
        "Use this when you need to check what you already know about the user's preferences, "
        "past decisions, or previously learned information."
    )
    args_schema: Type[BaseModel] = MemoryReadInput
    tags: list = ["memory", "context"]

    def _run(self) -> str:
        profile = get_current_profile()
        if profile is None:
            return "Memory is not available in this session."

        content = profile.load_memory()
        if not content:
            return "No memories stored yet."
        return content


# ---------------------------------------------------------------------------
# memory_write
# ---------------------------------------------------------------------------


class MemoryWriteInput(BaseModel):
    """Input schema for memory_write."""

    entry: str = Field(
        description="The knowledge to remember. Be concise and factual."
    )
    scope: Optional[Literal["agent", "global"]] = Field(
        default="agent",
        description=(
            "Where to store: 'agent' for your personal memory "
            "(default), 'global' for knowledge shared across all agents."
        ),
    )


class MemoryWriteTool(BaseTool):
    """Save knowledge to persistent memory for future sessions."""

    name: str = "memory_write"
    description: str = (
        "Save important knowledge to your persistent memory so you can recall it "
        "in future conversations. Memories are timestamped and append-only.\n\n"
        "When to use:\n"
        "- User explicitly asks you to remember something\n"
        "- User corrects your behavior or states a preference\n"
        "- You learn an important fact about the user's setup or workflow\n"
        "- A decision is made that should persist across sessions\n\n"
        "When NOT to use:\n"
        "- Temporary or session-specific information\n"
        "- Information already in memory (check with memory_read first)\n"
        "- Trivial or obvious facts\n\n"
        "Be concise — write facts, not narratives."
    )
    args_schema: Type[BaseModel] = MemoryWriteInput
    tags: list = ["memory", "context"]

    def _run(self, entry: str, scope: str = "agent") -> str:
        profile = get_current_profile()
        if profile is None:
            return "Memory is not available in this session."

        profile.append_memory(entry, scope=scope)
        return f"Saved to {scope} memory: {entry}"


memory_read = MemoryReadTool()
memory_write = MemoryWriteTool()
