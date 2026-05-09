"""Workflow persistence — YAML CRUD and SQLite execution tracking."""

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from cliver.workflow.workflow_models import Workflow

logger = logging.getLogger(__name__)

DEFAULT_RETENTION_DAYS = 300

_EXECUTIONS_SCHEMA = """
CREATE TABLE IF NOT EXISTS workflow_executions (
    thread_id     TEXT PRIMARY KEY,
    workflow_name TEXT NOT NULL,
    execution_id  TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'running',
    started_at    TEXT NOT NULL,
    finished_at   TEXT,
    inputs        TEXT,
    error         TEXT
);
CREATE INDEX IF NOT EXISTS idx_wf_exec_name
    ON workflow_executions(workflow_name);
"""


class WorkflowStore:
    """YAML-based workflow CRUD + SQLite execution tracking."""

    def __init__(self, workflows_dir: Path):
        self.workflows_dir = Path(workflows_dir)
        self._ensure_dir()

    def _ensure_dir(self) -> None:
        self.workflows_dir.mkdir(parents=True, exist_ok=True)

    def save_workflow(self, workflow: Workflow) -> None:
        path = self.workflows_dir / f"{workflow.name}.yaml"
        data = workflow.model_dump(exclude_none=True)
        path.write_text(
            yaml.dump(data, default_flow_style=False, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

    def load_workflow(self, name: str) -> Optional[Workflow]:
        path = self.workflows_dir / f"{name}.yaml"
        if not path.exists():
            project_path = Path.cwd() / ".cliver" / "workflows" / f"{name}.yaml"
            if project_path.exists():
                path = project_path
            else:
                return None
        return self.load_workflow_from_file(path)

    @staticmethod
    def load_workflow_from_file(path: "str | Path") -> Optional[Workflow]:
        path = Path(path)
        if not path.exists():
            return None
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            return Workflow(**data)
        except Exception as e:
            logger.error("Failed to load workflow from %s: %s", path, e)
            return None

    def list_workflows(self) -> List[str]:
        if not self.workflows_dir.is_dir():
            return []
        names = []
        for p in self.workflows_dir.glob("*.yaml"):
            names.append(p.stem)
        project_dir = Path.cwd() / ".cliver" / "workflows"
        if project_dir.is_dir() and project_dir.resolve() != self.workflows_dir.resolve():
            for p in project_dir.glob("*.yaml"):
                if p.stem not in names:
                    names.append(p.stem)
        return sorted(names)

    def list_all_workflows(self) -> List[tuple]:
        results = {}
        self._collect_workflows(self.workflows_dir, "global", results)
        project_dir = Path.cwd() / ".cliver" / "workflows"
        if project_dir.is_dir() and project_dir.resolve() != self.workflows_dir.resolve():
            self._collect_workflows(project_dir, "project", results)
        return [(name, src, path) for name, (src, _, path) in sorted(results.items())]

    @staticmethod
    def _collect_workflows(directory: Path, source: str, results: dict) -> None:
        if not directory.is_dir():
            return
        for p in directory.glob("*.yaml"):
            name = p.stem
            if name not in results:
                results[name] = (source, name, p)

    def delete_workflow(self, name: str) -> bool:
        path = self.workflows_dir / f"{name}.yaml"
        if path.exists():
            path.unlink()
            return True
        return False

    # ── SQLite Execution Tracking ──

    @staticmethod
    def _ensure_executions_table(db_path: Path) -> None:
        import sqlite3

        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(db_path)) as conn:
            conn.executescript(_EXECUTIONS_SCHEMA)

    @staticmethod
    def record_execution_start(
        db_path: Path,
        thread_id: str,
        workflow_name: str,
        execution_id: str,
        inputs: Optional[Dict[str, Any]] = None,
    ) -> None:
        import sqlite3

        WorkflowStore._ensure_executions_table(db_path)
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO workflow_executions "
                "(thread_id, workflow_name, execution_id, status, started_at, inputs) "
                "VALUES (?, ?, ?, 'running', ?, ?)",
                (
                    thread_id,
                    workflow_name,
                    execution_id,
                    datetime.now(timezone.utc).isoformat(),
                    json.dumps(inputs) if inputs else None,
                ),
            )

    @staticmethod
    def record_execution_end(
        db_path: Path,
        thread_id: str,
        status: str = "completed",
        error: Optional[str] = None,
    ) -> None:
        import sqlite3

        WorkflowStore._ensure_executions_table(db_path)
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute(
                "UPDATE workflow_executions SET status=?, finished_at=?, error=? WHERE thread_id=?",
                (status, datetime.now(timezone.utc).isoformat(), error, thread_id),
            )

    @staticmethod
    def list_executions(db_path: "str | Path", workflow_name: Optional[str] = None) -> List[dict]:
        import sqlite3

        db_path = Path(db_path)
        if not db_path.exists():
            return []
        WorkflowStore._ensure_executions_table(db_path)
        with sqlite3.connect(str(db_path)) as conn:
            conn.row_factory = sqlite3.Row
            if workflow_name:
                rows = conn.execute(
                    "SELECT * FROM workflow_executions WHERE workflow_name=? ORDER BY started_at DESC",
                    (workflow_name,),
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM workflow_executions ORDER BY started_at DESC").fetchall()
            return [dict(r) for r in rows]

    @staticmethod
    def delete_execution(thread_id: str, db_path: "str | Path", checkpointer=None) -> bool:
        import sqlite3

        db_path = Path(db_path)
        if not db_path.exists():
            return False
        with sqlite3.connect(str(db_path)) as conn:
            cursor = conn.execute(
                "DELETE FROM workflow_executions WHERE thread_id=?",
                (thread_id,),
            )
            return cursor.rowcount > 0

    @staticmethod
    def prune_executions(
        db_path: "str | Path",
        retention_days: int = DEFAULT_RETENTION_DAYS,
        checkpointer=None,
    ) -> int:
        import sqlite3

        db_path = Path(db_path)
        if not db_path.exists():
            return 0
        cutoff = (datetime.now(timezone.utc) - timedelta(days=retention_days)).isoformat()
        with sqlite3.connect(str(db_path)) as conn:
            cursor = conn.execute(
                "DELETE FROM workflow_executions WHERE started_at < ?",
                (cutoff,),
            )
            return cursor.rowcount
