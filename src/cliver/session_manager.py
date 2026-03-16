"""
Session Manager — manages conversation sessions for interactive mode.

Each session is a JSONL file storing conversation turns (user query + LLM response).
Sessions are agent-instance scoped, stored in {config_dir}/agents/{agent_name}/sessions/.

Storage format (JSONL — one JSON object per line):
    {"role": "user", "content": "hello", "timestamp": "2026-03-16 10:00 UTC"}
    {"role": "assistant", "content": "Hi! How can I help?", "timestamp": "2026-03-16 10:00 UTC"}
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SessionManager:
    """Manages conversation sessions stored as JSONL files.

    Directory layout:
        {sessions_dir}/
        ├── index.json                    # session index (id → metadata)
        ├── {session_id}.jsonl            # conversation turns
        └── ...
    """

    def __init__(self, sessions_dir: Path):
        self.sessions_dir = sessions_dir

    def _ensure_dir(self) -> None:
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    # -- Session lifecycle -----------------------------------------------------

    def create_session(self, title: str = "") -> str:
        """Create a new session. Returns session_id."""
        session_id = str(uuid.uuid4())[:8]
        self._ensure_dir()

        # Update index
        index = self._load_index()
        index[session_id] = {
            "title": title,
            "created_at": _timestamp(),
            "updated_at": _timestamp(),
            "turn_count": 0,
        }
        self._save_index(index)

        # Create empty JSONL file
        (self.sessions_dir / f"{session_id}.jsonl").touch()

        logger.info(f"Created session '{session_id}': {title}")
        return session_id

    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all sessions with metadata, most recent first."""
        index = self._load_index()
        sessions = []
        for sid, meta in index.items():
            sessions.append({"id": sid, **meta})
        sessions.sort(key=lambda s: s.get("updated_at", ""), reverse=True)
        return sessions

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and its conversation file."""
        index = self._load_index()
        if session_id not in index:
            return False
        del index[session_id]
        self._save_index(index)

        jsonl_path = self.sessions_dir / f"{session_id}.jsonl"
        if jsonl_path.exists():
            jsonl_path.unlink()
        return True

    # -- Conversation recording ------------------------------------------------

    def append_turn(self, session_id: str, role: str, content: str) -> None:
        """Append a conversation turn to a session.

        Args:
            session_id: The session to append to.
            role: "user" or "assistant".
            content: The message content.
        """
        self._ensure_dir()
        jsonl_path = self.sessions_dir / f"{session_id}.jsonl"

        turn = {"role": role, "content": content, "timestamp": _timestamp()}
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(turn, ensure_ascii=False) + "\n")

        # Update index metadata
        index = self._load_index()
        if session_id in index:
            index[session_id]["updated_at"] = _timestamp()
            index[session_id]["turn_count"] = index[session_id].get("turn_count", 0) + 1
            # Set title from first user message if empty
            if not index[session_id]["title"] and role == "user":
                index[session_id]["title"] = content[:80]
            self._save_index(index)

    def load_turns(self, session_id: str) -> List[Dict[str, str]]:
        """Load all conversation turns from a session.

        Returns list of {"role": "user"|"assistant", "content": "...", "timestamp": "..."}
        """
        jsonl_path = self.sessions_dir / f"{session_id}.jsonl"
        if not jsonl_path.exists():
            return []

        turns = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    turns.append(json.loads(line))
        return turns

    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a session."""
        index = self._load_index()
        if session_id not in index:
            return None
        return {"id": session_id, **index[session_id]}

    # -- Index management ------------------------------------------------------

    def _load_index(self) -> Dict[str, Dict[str, Any]]:
        index_path = self.sessions_dir / "index.json"
        if not index_path.exists():
            return {}
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_index(self, index: Dict[str, Dict[str, Any]]) -> None:
        self._ensure_dir()
        index_path = self.sessions_dir / "index.json"
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2, ensure_ascii=False)


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
