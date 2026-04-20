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
    scope: Optional[Literal["agent", "global"]] = Field(
        default="agent",
        description=(
            "Where to store: 'agent' for your personal memory "
            "(default), 'global' for knowledge shared across all agents."
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

    def _run(self, content: str, comment: str = None, mode: str = "append", scope: str = "agent") -> str:
        profile = get_current_profile()
        if profile is None:
            return "Memory is not available in this session."

        if mode == "rewrite":
            profile.save_memory(content, scope=scope)
            return f"Rewrote {scope} memory."
        else:
            profile.append_memory(content, scope=scope, comment=comment or "")
            return f"Saved to {scope} memory: {content}"


# ---------------------------------------------------------------------------
# identity_update
# ---------------------------------------------------------------------------


class IdentityUpdateInput(BaseModel):
    """Input schema for identity_update."""

    content: str = Field(
        description=(
            "The complete identity document in markdown format. "
            "This replaces the entire identity — include ALL information, "
            "not just the new parts."
        )
    )


class IdentityUpdateTool(BaseTool):
    """Update the identity profile — a living document about the agent and user."""

    name: str = "Identity"
    description: str = (
        "Update the identity profile — a living markdown document that describes "
        "who you are (agent persona) and who the user is (their profile).\n\n"
        "Unlike memory (append-only), identity is **rewritten as a whole** each time. "
        "Always include all existing information plus any updates.\n\n"
        "The identity document typically contains:\n"
        "- User profile: name, location, role, preferences, environment\n"
        "- Agent persona: communication style, behavior preferences\n"
        "- Key context: timezone, language, tools the user prefers\n\n"
        "When to update:\n"
        "- User shares personal info (name, location, role)\n"
        "- User states a preference about how you should behave\n"
        "- You learn about the user's environment or workflow\n\n"
        "Read the current identity first to avoid losing existing information."
    )
    args_schema: Type[BaseModel] = IdentityUpdateInput
    tags: list = ["memory", "identity", "context"]

    def _run(self, content: str) -> str:
        profile = get_current_profile()
        if profile is None:
            return "Identity is not available in this session."

        profile.save_identity(content)
        return "Identity profile updated."


memory_read = MemoryReadTool()
memory_write = MemoryWriteTool()
identity_update = IdentityUpdateTool()
