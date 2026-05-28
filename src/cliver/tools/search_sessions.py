"""Built-in tool for searching past conversation sessions.

The LLM uses this tool to search its own conversation history,
enabling recall of prior discussions and context.
"""

import logging

from cliver.agent_profile import get_current_profile
from cliver.tool import tool

logger = logging.getLogger(__name__)


@tool(
    name="SearchSessions",
    description=(
        "Search past conversation sessions using full-text search. "
        "Use when the user references prior work, asks about previous discussions, "
        "or you need to recall context from earlier conversations. "
        "Returns matching session metadata and text snippets."
    ),
)
def search_sessions(query: str, limit: int = 10) -> list[dict]:
    """Search past conversation sessions for relevant context.

    Args:
        query: Search query for past sessions.
        limit: Max number of sessions to return (default 10).
    """
    profile = get_current_profile()
    if profile is None:
        return [{"text": "Session search is not available in this session."}]

    from cliver.session_manager import SessionManager

    sm = SessionManager(profile.db_path)
    try:
        results = sm.search(query, limit=limit)
    except Exception as e:
        logger.warning(f"Session search failed: {e}")
        return [{"error": f"Search failed: {e}"}]

    if not results:
        return [{"text": f"No past sessions found matching '{query}'."}]

    lines = [f"Found {len(results)} session(s) matching '{query}':\n"]
    for r in results:
        title = r.get("title") or "(untitled)"
        lines.append(f"--- Session {r['session_id']}: {title} ({r['created_at']}) ---")
        for snippet in r.get("snippets", []):
            lines.append(f"  [{snippet['role']}]: {snippet['content']}")
        lines.append("")

    return [{"text": "\n".join(lines)}]
