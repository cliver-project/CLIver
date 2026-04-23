"""SQLite-backed task run history and live state for the gateway.

Schema:
    task_runs(task_name, execution_id, status, started_at, finished_at, error)
    task_state(task_name, status, updated_at, suspend_reason)
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from cliver.db import SQLiteStore
from cliver.task_manager import TaskRun

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS task_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_name TEXT NOT NULL,
    execution_id TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    error TEXT
);

CREATE INDEX IF NOT EXISTS idx_task_runs_task_name
    ON task_runs(task_name);

CREATE INDEX IF NOT EXISTS idx_task_runs_started_at
    ON task_runs(task_name, started_at DESC);

CREATE TABLE IF NOT EXISTS task_state (
    task_name TEXT PRIMARY KEY,
    status TEXT NOT NULL DEFAULT 'pending',
    updated_at TEXT NOT NULL,
    suspend_reason TEXT
);

CREATE INDEX IF NOT EXISTS idx_task_state_status
    ON task_state(status);
"""


class TaskRunStore:
    """SQLite store for task execution history and live state."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self._store = SQLiteStore(db_path)
        self._store.execute_schema(_SCHEMA)

    def record_run(self, run: TaskRun) -> None:
        with self._store.write() as db:
            db.execute(
                """INSERT INTO task_runs
                   (task_name, execution_id, status, started_at, finished_at, error)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (run.task_name, run.execution_id, run.status, run.started_at, run.finished_at, run.error),
            )

    def get_runs(self, task_name: str, limit: int = 10) -> List[TaskRun]:
        with self._store.read() as db:
            rows = db.execute(
                """SELECT task_name, execution_id, status,
                          started_at, finished_at, error
                   FROM task_runs
                   WHERE task_name = ?
                   ORDER BY id DESC
                   LIMIT ?""",
                (task_name, limit),
            ).fetchall()
            return [TaskRun(**dict(r)) for r in rows]

    def delete_runs(self, task_name: str) -> int:
        with self._store.write() as db:
            cursor = db.execute(
                "DELETE FROM task_runs WHERE task_name = ?",
                (task_name,),
            )
            return cursor.rowcount

    def get_last_run_time(self, task_name: str) -> Optional[float]:
        with self._store.read() as db:
            row = db.execute(
                """SELECT started_at FROM task_runs
                   WHERE task_name = ?
                   ORDER BY id DESC LIMIT 1""",
                (task_name,),
            ).fetchone()
            if not row:
                return None
            return _parse_timestamp(row["started_at"])

    def get_all_task_names(self) -> List[str]:
        with self._store.read() as db:
            rows = db.execute("SELECT DISTINCT task_name FROM task_runs").fetchall()
            return [r["task_name"] for r in rows]

    # -- Task state -----------------------------------------------------------

    def set_task_state(self, task_name: str, status: str, reason: Optional[str] = None) -> None:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        with self._store.write() as db:
            db.execute(
                """INSERT INTO task_state (task_name, status, updated_at, suspend_reason)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(task_name)
                   DO UPDATE SET status=excluded.status,
                                 updated_at=excluded.updated_at,
                                 suspend_reason=excluded.suspend_reason""",
                (task_name, status, now, reason),
            )

    def get_task_state(self, task_name: str) -> Optional[dict]:
        with self._store.read() as db:
            row = db.execute(
                "SELECT task_name, status, updated_at, suspend_reason FROM task_state WHERE task_name = ?",
                (task_name,),
            ).fetchone()
            return dict(row) if row else None

    def get_tasks_by_status(self, status: str) -> List[dict]:
        with self._store.read() as db:
            rows = db.execute(
                "SELECT task_name, status, updated_at, suspend_reason FROM task_state WHERE status = ?",
                (status,),
            ).fetchall()
            return [dict(r) for r in rows]

    def delete_task_state(self, task_name: str) -> None:
        with self._store.write() as db:
            db.execute("DELETE FROM task_state WHERE task_name = ?", (task_name,))

    def close(self) -> None:
        self._store.close()


def _parse_timestamp(ts: str) -> float:
    from datetime import datetime, timezone

    ts = ts.strip()
    if ts.endswith(" UTC"):
        ts = ts[:-4]
    try:
        dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
        return dt.replace(tzinfo=timezone.utc).timestamp()
    except ValueError:
        try:
            dt = datetime.strptime(ts, "%Y-%m-%d")
            return dt.replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            return 0.0
