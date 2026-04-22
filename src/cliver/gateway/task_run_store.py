"""SQLite-backed task run history for the gateway.

Replaces the file-based *.runs.yaml approach and the separate
cron-state.json — cron "last run" is derived from this table.

Schema:
    task_runs(task_name, execution_id, status, started_at, finished_at, error)
"""

import logging
import sqlite3
from pathlib import Path
from typing import List, Optional

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
"""


class TaskRunStore:
    """SQLite store for task execution history."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self._db: Optional[sqlite3.Connection] = None

    def _get_db(self) -> sqlite3.Connection:
        """Lazy-init the SQLite connection."""
        if self._db is not None:
            return self._db
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(self.db_path))
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.row_factory = sqlite3.Row
        self._db.executescript(_SCHEMA)
        return self._db

    def record_run(self, run: TaskRun) -> None:
        """Insert a task run record."""
        db = self._get_db()
        db.execute(
            """INSERT INTO task_runs
               (task_name, execution_id, status, started_at, finished_at, error)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                run.task_name,
                run.execution_id,
                run.status,
                run.started_at,
                run.finished_at,
                run.error,
            ),
        )
        db.commit()

    def get_runs(self, task_name: str, limit: int = 10) -> List[TaskRun]:
        """Get recent runs for a task, most recent first."""
        db = self._get_db()
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
        """Delete all runs for a task. Returns number of rows deleted."""
        db = self._get_db()
        cursor = db.execute(
            "DELETE FROM task_runs WHERE task_name = ?",
            (task_name,),
        )
        db.commit()
        return cursor.rowcount

    def get_last_run_time(self, task_name: str) -> Optional[float]:
        """Get the epoch timestamp of the most recent run for cron scheduling.

        Returns None if the task has never run.
        """
        db = self._get_db()
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
        """Return all distinct task names that have run records."""
        db = self._get_db()
        rows = db.execute("SELECT DISTINCT task_name FROM task_runs").fetchall()
        return [r["task_name"] for r in rows]

    def close(self) -> None:
        """Close the database connection."""
        if self._db:
            self._db.close()
            self._db = None


def _parse_timestamp(ts: str) -> float:
    """Parse a TaskManager timestamp string to epoch seconds.

    Handles the format "2026-01-01 09:00:00 UTC".
    """
    from datetime import datetime, timezone

    ts = ts.strip()
    if ts.endswith(" UTC"):
        ts = ts[:-4]
    try:
        dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
        return dt.replace(tzinfo=timezone.utc).timestamp()
    except ValueError:
        # Fallback: try date-only
        try:
            dt = datetime.strptime(ts, "%Y-%m-%d")
            return dt.replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            return 0.0
