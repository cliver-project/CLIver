"""Built-in tool for searching past conversation sessions.

The LLM uses this tool to search its own conversation history,
enabling recall of prior discussions and context.
"""

import logging
from typing import Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from cliver.agent_profile import get_current_profile

logger = logging.getLogger(__name__)


class SearchSessionsInput(BaseModel):
    """Input schema for search_sessions."""

    query: str = Field(description="Search query for past sessions")
    limit: int = Field(default=10, description="Max number of sessions to return")


class SearchSessionsTool(BaseTool):
    """Search past conversation sessions for relevant context."""

    name: str = "SearchSessions"
    description: str = (
        "Search past conversation sessions using full-text search. "
        "Use when the user references prior work, asks about previous discussions, "
        "or you need to recall context from earlier conversations. "
        "Returns matching session metadata and text snippets."
    )
    args_schema: Type[BaseModel] = SearchSessionsInput

    def _run(self, query: str, limit: int = 10) -> str:
        profile = get_current_profile()
        if profile is None:
            return "Session search is not available in this session."

        from cliver.session_manager import SessionManager

        sm = SessionManager(profile.sessions_dir)
        try:
            results = sm.search(query, limit=limit)
        except Exception as e:
            logger.warning(f"Session search failed: {e}")
            return f"Search failed: {e}"

        if not results:
            return f"No past sessions found matching '{query}'."

        lines = [f"Found {len(results)} session(s) matching '{query}':\n"]
        for r in results:
            title = r.get("title") or "(untitled)"
            lines.append(f"--- Session {r['session_id']}: {title} ({r['created_at']}) ---")
            for snippet in r.get("snippets", []):
                lines.append(f"  [{snippet['role']}]: {snippet['content']}")
            lines.append("")

        return "\n".join(lines)


# Module-level instance for tool registry discovery
search_sessions = SearchSessionsTool()
