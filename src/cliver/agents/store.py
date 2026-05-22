"""AgentStore -- SQLite persistence for agent configurations."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from cliver.agents.models import Agent
from cliver.db import SQLiteStore

_SCHEMA = """
CREATE TABLE IF NOT EXISTS agents (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    type TEXT NOT NULL DEFAULT 'cliver',
    description TEXT,
    role TEXT,
    model TEXT,
    is_default INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


class AgentStore:
    """CRUD store for agent configurations.

    Instantiate with a db_path, or use ``AgentStore.from_config_dir(config_dir)``
    to resolve the default ``cliver.db`` path automatically.
    """

    def __init__(self, db_path: Path):
        self._db_path = Path(db_path)
        self._store: Optional[SQLiteStore] = None

    @classmethod
    def from_config_dir(cls, config_dir: Path) -> "AgentStore":
        """Create a store using the default database path for a config directory."""
        return cls(config_dir / "cliver.db")

    def _get_store(self) -> SQLiteStore:
        if self._store is None:
            self._store = SQLiteStore(self._db_path)
            self._store.execute_schema(_SCHEMA)
        return self._store

    # -- CRUD ----------------------------------------------------------------

    def list_agents(self) -> List[Agent]:
        with self._get_store().read() as db:
            rows = db.execute(
                "SELECT id, name, type, description, role, model, is_default, created_at, updated_at "
                "FROM agents ORDER BY updated_at DESC"
            ).fetchall()
        return [Agent(**dict(r)) for r in rows]

    def get_agent(self, agent_id: str) -> Optional[Agent]:
        with self._get_store().read() as db:
            row = db.execute(
                "SELECT id, name, type, description, role, model, is_default, created_at, updated_at "
                "FROM agents WHERE id = ?",
                (agent_id,),
            ).fetchone()
        if row is None:
            return None
        return Agent(**dict(row))

    def get_agent_by_name(self, name: str) -> Optional[Agent]:
        with self._get_store().read() as db:
            row = db.execute(
                "SELECT id, name, type, description, role, model, is_default, created_at, updated_at "
                "FROM agents WHERE name = ?",
                (name,),
            ).fetchone()
        if row is None:
            return None
        return Agent(**dict(row))

    def create_agent(
        self,
        name: str,
        type: str = "cliver",
        description: Optional[str] = None,
        role: Optional[str] = None,
        model: Optional[str] = None,
        is_default: int = 0,
    ) -> Agent:
        agent = Agent(
            name=name,
            type=type,
            description=description,
            role=role,
            model=model,
            is_default=is_default,
        )
        with self._get_store().write() as db:
            if agent.is_default:
                db.execute("UPDATE agents SET is_default = 0 WHERE is_default = 1")
            db.execute(
                "INSERT INTO agents (id, name, type, description, role, model, is_default, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    agent.id,
                    agent.name,
                    agent.type,
                    agent.description,
                    agent.role,
                    agent.model,
                    agent.is_default,
                    agent.created_at,
                    agent.updated_at,
                ),
            )
        return agent

    def update_agent(
        self,
        agent_id: str,
        **kwargs,
    ) -> Optional[Agent]:
        existing = self.get_agent(agent_id)
        if existing is None:
            return None

        for key, value in kwargs.items():
            if value is not None and hasattr(existing, key):
                setattr(existing, key, value)

        if kwargs.get("is_default"):
            existing.is_default = 1

        from cliver.agents.models import _now

        existing.updated_at = _now()
        with self._get_store().write() as db:
            if existing.is_default:
                db.execute("UPDATE agents SET is_default = 0 WHERE is_default = 1 AND id != ?", (agent_id,))
            db.execute(
                "UPDATE agents SET name=?, type=?, description=?, role=?, model=?, is_default=?, updated_at=? "
                "WHERE id=?",
                (
                    existing.name,
                    existing.type,
                    existing.description,
                    existing.role,
                    existing.model,
                    existing.is_default,
                    existing.updated_at,
                    agent_id,
                ),
            )
        return existing

    def delete_agent(self, agent_id: str) -> bool:
        with self._get_store().write() as db:
            cursor = db.execute("DELETE FROM agents WHERE id = ?", (agent_id,))
        return cursor.rowcount > 0

    def get_default_agent(self) -> Optional[Agent]:
        with self._get_store().read() as db:
            row = db.execute(
                "SELECT id, name, type, description, role, model, is_default, created_at, updated_at "
                "FROM agents WHERE is_default = 1 LIMIT 1"
            ).fetchone()
        if row is None:
            return None
        return Agent(**dict(row))

    def set_default(self, agent_id: str) -> bool:
        existing = self.get_agent(agent_id)
        if existing is None:
            return False
        from cliver.agents.models import _now

        with self._get_store().write() as db:
            db.execute("UPDATE agents SET is_default = 0 WHERE is_default = 1")
            db.execute(
                "UPDATE agents SET is_default = 1, updated_at = ? WHERE id = ?",
                (
                    _now(),
                    agent_id,
                ),
            )
        return True

    def close(self) -> None:
        if self._store:
            self._store.close()
            self._store = None
