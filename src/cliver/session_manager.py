"""
Session Manager — hybrid session persistence.

SQLite for session metadata (fast listing, search, cleanup).
JSONL files for full conversation turns (preserves tool_calls,
reasoning_content, additional_kwargs, media references).

Directory layout:
    sessions_dir/
    ├── sessions.db           # Metadata index
    └── data/
        └── {session_id}/
            ├── turns.jsonl   # One JSON object per turn
            └── media/        # Media files referenced by turns
"""

import json
import logging
import shutil
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from cliver.db import SQLiteStore

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    title TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    turn_count INTEGER DEFAULT 0,
    options TEXT
);
"""


class SessionManager:
    """Manages conversation sessions with SQLite metadata + JSONL turn files."""

    def __init__(self, sessions_dir: Path):
        self.sessions_dir = sessions_dir
        self._data_dir = sessions_dir / "data"
        self._store: Optional[SQLiteStore] = None

    def _get_store(self) -> SQLiteStore:
        if self._store is None:
            self.sessions_dir.mkdir(parents=True, exist_ok=True)
            db_path = self.sessions_dir / "sessions.db"
            self._store = SQLiteStore(db_path)
            self._store.execute_schema(_SCHEMA)
        return self._store

    def _session_dir(self, session_id: str) -> Path:
        return self._data_dir / session_id

    def _turns_path(self, session_id: str) -> Path:
        return self._session_dir(session_id) / "turns.jsonl"

    # -- Session lifecycle -----------------------------------------------------

    def create_session(self, title: str = "") -> str:
        """Create a new session. Returns session_id."""
        session_id = str(uuid.uuid4())[:8]
        now = _timestamp()
        with self._get_store().write() as db:
            db.execute(
                "INSERT INTO sessions (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (session_id, title or None, now, now),
            )
        self._session_dir(session_id).mkdir(parents=True, exist_ok=True)
        return session_id

    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all sessions with metadata, most recent first."""
        with self._get_store().read() as db:
            rows = db.execute(
                "SELECT id, title, created_at, updated_at, turn_count, options FROM sessions ORDER BY updated_at DESC"
            ).fetchall()
        return [_row_to_dict(r) for r in rows]

    def delete_session(self, session_id: str) -> bool:
        """Delete a session, its turns file, and media directory."""
        with self._get_store().write() as db:
            cursor = db.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        session_dir = self._session_dir(session_id)
        if session_dir.exists():
            shutil.rmtree(session_dir, ignore_errors=True)
        return cursor.rowcount > 0

    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a session."""
        with self._get_store().read() as db:
            row = db.execute(
                "SELECT id, title, created_at, updated_at, turn_count, options FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        return _row_to_dict(row)

    # -- Conversation recording ------------------------------------------------

    def append_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        *,
        msg_type: Optional[str] = None,
        additional_kwargs: Optional[Dict[str, Any]] = None,
        tool_calls: Optional[list] = None,
        tool_call_id: Optional[str] = None,
        tool_name: Optional[str] = None,
        media: Optional[List[Dict[str, str]]] = None,
    ) -> None:
        """Append a conversation turn to a session's JSONL file."""
        now = _timestamp()

        # Build turn record
        turn: Dict[str, Any] = {
            "role": role,
            "content": content,
            "timestamp": now,
        }
        if msg_type:
            turn["type"] = msg_type
        if additional_kwargs:
            turn["additional_kwargs"] = additional_kwargs
        if tool_calls:
            turn["tool_calls"] = tool_calls
        if tool_call_id:
            turn["tool_call_id"] = tool_call_id
        if tool_name:
            turn["tool_name"] = tool_name
        if media:
            turn["media"] = media

        # Write to JSONL
        turns_path = self._turns_path(session_id)
        turns_path.parent.mkdir(parents=True, exist_ok=True)
        with open(turns_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(turn, ensure_ascii=False) + "\n")

        # Update metadata
        with self._get_store().write() as db:
            db.execute(
                "UPDATE sessions SET updated_at = ?, turn_count = turn_count + 1 WHERE id = ?",
                (now, session_id),
            )
            if role == "user":
                db.execute(
                    "UPDATE sessions SET title = ? WHERE id = ? AND (title IS NULL OR title = '')",
                    (content[:80], session_id),
                )

    def append_turn_from_message(self, session_id: str, message) -> None:
        """Append a turn from a LangChain BaseMessage, preserving all metadata."""
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

        if isinstance(message, HumanMessage):
            role, msg_type = "user", "human"
        elif isinstance(message, AIMessage):
            role, msg_type = "assistant", "ai"
        elif isinstance(message, ToolMessage):
            role, msg_type = "tool", "tool"
        elif isinstance(message, SystemMessage):
            role, msg_type = "system", "system"
        else:
            role, msg_type = "unknown", type(message).__name__.lower()

        content = message.content if isinstance(message.content, str) else str(message.content)
        kwargs = getattr(message, "additional_kwargs", None) or None
        tool_calls = getattr(message, "tool_calls", None) or None
        tool_call_id = getattr(message, "tool_call_id", None) if isinstance(message, ToolMessage) else None
        tool_name = getattr(message, "name", None) if isinstance(message, ToolMessage) else None

        # Filter empty kwargs
        if kwargs:
            kwargs = {k: v for k, v in kwargs.items() if v is not None and v != ""}
            if not kwargs:
                kwargs = None

        self.append_turn(
            session_id,
            role,
            content,
            msg_type=msg_type,
            additional_kwargs=kwargs,
            tool_calls=tool_calls,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
        )

    def load_turns(self, session_id: str) -> List[Dict[str, Any]]:
        """Load all conversation turns from a session's JSONL file."""
        turns_path = self._turns_path(session_id)
        if not turns_path.exists():
            return []
        turns = []
        with open(turns_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        turns.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return turns

    def trim_turns(self, session_id: str, keep_last: int = 50) -> int:
        """Keep only the most recent turns, rewriting the JSONL file.

        Returns the number of turns deleted.
        """
        turns = self.load_turns(session_id)
        if len(turns) <= keep_last:
            return 0
        deleted = len(turns) - keep_last
        kept = turns[-keep_last:]
        turns_path = self._turns_path(session_id)
        with open(turns_path, "w", encoding="utf-8") as f:
            for turn in kept:
                f.write(json.dumps(turn, ensure_ascii=False) + "\n")
        with self._get_store().write() as db:
            db.execute(
                "UPDATE sessions SET turn_count = ? WHERE id = ?",
                (len(kept), session_id),
            )
        return deleted

    # -- Media -----------------------------------------------------------------

    def get_media_dir(self, session_id: str) -> Path:
        """Get the media directory for a session, creating it if needed."""
        media_dir = self._session_dir(session_id) / "media"
        media_dir.mkdir(parents=True, exist_ok=True)
        return media_dir

    # -- Cleanup ---------------------------------------------------------------

    def delete_oldest_sessions(self, keep: int = 300) -> int:
        """Delete oldest sessions when total count exceeds keep limit."""
        with self._get_store().write() as db:
            total = db.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
            if total <= keep:
                return 0
            rows = db.execute(
                "SELECT id FROM sessions ORDER BY updated_at ASC LIMIT ?",
                (total - keep,),
            ).fetchall()
            ids = [r["id"] for r in rows]
            if not ids:
                return 0
            placeholders = ",".join("?" * len(ids))
            db.execute(f"DELETE FROM sessions WHERE id IN ({placeholders})", ids)

        for sid in ids:
            session_dir = self._session_dir(sid)
            if session_dir.exists():
                shutil.rmtree(session_dir, ignore_errors=True)
        return len(ids)

    def delete_stale_sessions(self, max_age_days: int = 90) -> int:
        """Delete sessions not updated within max_age_days."""
        from datetime import timedelta

        cutoff = (datetime.now(timezone.utc) - timedelta(days=max_age_days)).isoformat()
        with self._get_store().write() as db:
            rows = db.execute("SELECT id FROM sessions WHERE updated_at < ?", (cutoff,)).fetchall()
            ids = [r["id"] for r in rows]
            if not ids:
                return 0
            placeholders = ",".join("?" * len(ids))
            db.execute(f"DELETE FROM sessions WHERE id IN ({placeholders})", ids)

        for sid in ids:
            session_dir = self._session_dir(sid)
            if session_dir.exists():
                shutil.rmtree(session_dir, ignore_errors=True)
        return len(ids)

    # -- Options ---------------------------------------------------------------

    def save_options(self, session_id: str, options: Dict[str, Any]) -> None:
        """Persist session options as JSON in SQLite."""
        clean = {k: v for k, v in options.items() if v is not None and k != "statusbar"}
        with self._get_store().write() as db:
            db.execute(
                "UPDATE sessions SET options = ? WHERE id = ?",
                (json.dumps(clean, ensure_ascii=False), session_id),
            )

    def load_options(self, session_id: str) -> Dict[str, Any]:
        """Load persisted session options."""
        with self._get_store().read() as db:
            row = db.execute("SELECT options FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if row is None or row["options"] is None:
            return {}
        try:
            return json.loads(row["options"])
        except (json.JSONDecodeError, TypeError):
            return {}

    # -- Title -----------------------------------------------------------------

    def update_title(self, session_id: str, title: str) -> None:
        """Update the title of a session."""
        with self._get_store().write() as db:
            db.execute("UPDATE sessions SET title = ? WHERE id = ?", (title, session_id))

    # -- Search ----------------------------------------------------------------

    def search(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Search across session turns by scanning JSONL files."""
        query_lower = query.lower()
        results: Dict[str, Dict[str, Any]] = {}

        sessions = self.list_sessions()
        for sess in sessions:
            sid = sess["id"]
            turns = self.load_turns(sid)
            snippets = []
            for turn in turns:
                content = turn.get("content", "")
                if query_lower in content.lower():
                    idx = content.lower().index(query_lower)
                    start = max(0, idx - 40)
                    end = min(len(content), idx + len(query) + 40)
                    snippet = ("..." if start > 0 else "") + content[start:end] + ("..." if end < len(content) else "")
                    snippets.append({"role": turn.get("role", "?"), "content": snippet})
            if snippets:
                results[sid] = {
                    "session_id": sid,
                    "title": sess.get("title"),
                    "created_at": sess.get("created_at"),
                    "turn_count": sess.get("turn_count", 0),
                    "snippets": snippets[:5],
                }
            if len(results) >= limit:
                break

        return list(results.values())

    def close(self) -> None:
        """Close the database connection."""
        if self._store:
            self._store.close()
            self._store = None


def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    """Convert a sqlite3.Row to a plain dict, parsing options JSON."""
    d = dict(row)
    if "options" in d and d["options"]:
        try:
            d["options"] = json.loads(d["options"])
        except (json.JSONDecodeError, TypeError):
            d["options"] = {}
    return d


def _timestamp() -> str:
    from cliver.util import format_datetime

    return format_datetime()
