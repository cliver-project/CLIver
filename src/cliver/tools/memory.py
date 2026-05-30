"""Built-in memory tools for persistent knowledge across sessions.

The LLM uses these tools to save and recall knowledge that persists
beyond the current conversation. Memory is managed by CliverProfile
and stored as markdown files on disk.
"""

import logging
from typing import Optional

from cliver.agent_profile import get_current_profile
from cliver.tool import tool

logger = logging.getLogger(__name__)


@tool(
    name="MemoryRead",
    description=(
        "Recall knowledge from previous conversations that persists across sessions. "
        "ONLY use this when the user's request references past context, preferences, "
        "or decisions that you need to look up. Do NOT read memory for greetings, "
        "casual chat, or general questions — it adds latency and cost for no benefit."
    ),
)
def memory_read() -> list[dict]:
    """Read persistent memory to recall knowledge from previous sessions.

    Returns both global memory (shared across all agents) and agent-specific memory.
    Use when you need to check what you already know about the user's preferences,
    past decisions, or previously learned information.
    """
    profile = get_current_profile()
    if profile is None:
        return [{"text": "Memory is not available in this session."}]

    content = profile.load_memory()
    if not content:
        return [{"text": "No memories stored yet."}]
    return [{"text": content}]


@tool(
    name="MemoryWrite",
    description=(
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
    ),
)
def memory_write(
    content: str,
    comment: Optional[str] = None,
    mode: Optional[str] = "append",
) -> list[dict]:
    """Save knowledge to persistent memory for future sessions.

    Two modes:
    - append (default): add a new timestamped entry.
    - rewrite: replace the entire memory document — use when correcting,
      consolidating, or reorganizing existing memories.

    Args:
        content: In 'append' mode: the knowledge to remember (concise and factual).
            In 'rewrite' mode: the complete memory document in markdown.
        comment: Brief context about why this is being saved (only for append mode).
        mode: 'append' (default) or 'rewrite'.
    """
    profile = get_current_profile()
    if profile is None:
        return [{"text": "Memory is not available in this session."}]

    if mode == "rewrite":
        profile.save_memory(content)
        return [{"text": "Rewrote memory."}]
    else:
        profile.append_memory(content, comment=comment or "")
        return [{"text": f"Saved to memory: {content}"}]


@tool(
    name="Identity",
    description=(
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
    ),
)
def identity_update(content: str) -> list[dict]:
    """Update the identity profile.

    A living markdown document that describes who you are (agent persona)
    and who the user is (their profile). Replaces the entire identity.

    Args:
        content: The complete identity document in markdown format.
            This replaces the entire identity. Include ALL information,
            not just the new parts.
    """
    profile = get_current_profile()
    if profile is None:
        return [{"text": "Identity is not available in this session."}]

    profile.save_identity(content)
    return [{"text": "Identity profile updated."}]
