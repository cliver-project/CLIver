"""Built-in memory tools for persistent knowledge across sessions.

The LLM uses these tools to save and recall knowledge that persists
beyond the current conversation. Memory is managed by CliverProfile
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

    name: str = "MemoryRead"
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

    content: str = Field(
        description=(
            "In 'append' mode: the knowledge to remember (concise and factual). "
            "In 'rewrite' mode: the complete memory document in markdown."
        )
    )
    comment: Optional[str] = Field(
        default=None,
        description="Brief context about why this is being saved (only for append mode).",
    )
    mode: Optional[Literal["append", "rewrite"]] = Field(
        default="append",
        description=(
            "'append' (default): add a new timestamped entry. "
            "'rewrite': replace the entire memory document — use when correcting, "
            "consolidating, or reorganizing existing memories."
        ),
    )


class MemoryWriteTool(BaseTool):
    """Save knowledge to persistent memory for future sessions."""

    name: str = "MemoryWrite"
    description: str = (
        "Save important knowledge to your persistent memory so you can recall it "
        "in future conversations.\n\n"
        "Two modes:\n"
        "- **append** (default): add an entry with an optional comment\n"
        "- **rewrite**: replace the entire memory document — use to consolidate, "
        "correct, or reorganize memory by topic\n\n"
        "Memory hygiene:\n"
        "- ALWAYS read memory first to avoid duplicates\n"
        "- Organize by topic (`## Project Setup`, `## Preferences`), not by date\n"
        "- When memory has many scattered entries, consolidate with `rewrite` mode: "
        "merge related entries under topic headings and remove outdated ones\n"
        "- Keep entries concise and factual — no narratives\n\n"
        "When to use:\n"
        "- User explicitly asks you to remember something\n"
        "- User corrects your behavior or states a preference\n"
        "- You learn an important fact about the user's setup or workflow\n"
        "- A decision is made that should persist across sessions\n\n"
        "When NOT to use:\n"
        "- Temporary or session-specific information\n"
        "- Information already in memory (check with MemoryRead first)\n"
        "- Trivial or obvious facts"
    )
    args_schema: Type[BaseModel] = MemoryWriteInput
    tags: list = ["memory", "context"]

    def _run(self, content: str, comment: str = None, mode: str = "append") -> str:
        profile = get_current_profile()
        if profile is None:
            return "Memory is not available in this session."

        if mode == "rewrite":
            profile.save_memory(content)
            return "Rewrote memory."
        else:
            profile.append_memory(content, comment=comment or "")
            return f"Saved to memory: {content}"


memory_read = MemoryReadTool()
memory_write = MemoryWriteTool()
