"""
Workflow persistence — YAML definitions + execution tracking.

Workflows stored as {workflows_dir}/{name}.yaml
Execution metadata stored in the checkpoint SQLite database.
LangGraph checkpoint/writes tables are managed exclusively via LangGraph API.
"""

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from cliver.db import get_store
from cliver.workflow.workflow_models import Workflow

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

logger = logging.getLogger(__name__)


class WorkflowStore:
    """Manages workflow YAML files and execution tracking."""

    def __init__(self, workflows_dir: Path):
        self.workflows_dir = Path(workflows_dir)

    def _ensure_dir(self) -> None:
        self.workflows_dir.mkdir(parents=True, exist_ok=True)

    # -- Workflow YAML CRUD ----------------------------------------------------

    def save_workflow(self, workflow: Workflow) -> None:
        """Save a workflow definition as YAML."""
        self._ensure_dir()
        path = self.workflows_dir / f"{workflow.name}.yaml"
        data = workflow.model_dump(exclude_none=True, mode="json")
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    def load_workflow(self, name: str) -> Optional[Workflow]:
        """Load a workflow definition from YAML.

        Search order:
        1. {workflows_dir}/{name}.yaml
        2. {workflows_dir}/sub-workflows/{name}.yaml
        3. .cliver/workflows/{name}.yaml  (project-local)
        """
        path = self.workflows_dir / f"{name}.yaml"
        if not path.exists():
            path = self.workflows_dir / "sub-workflows" / f"{name}.yaml"
        if not path.exists():
            project_path = Path(".cliver/workflows") / f"{name}.yaml"
            if project_path.exists():
                path = project_path
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            return Workflow(**data)
        except Exception as e:
            logger.error(f"Failed to load workflow '{name}': {e}")
            return None

    @staticmethod
    def load_workflow_from_file(path: "str | Path") -> Optional[Workflow]:
        """Load a workflow definition from a YAML file path.

        For relative paths, tries the following search order:
        1. The literal path (relative to CWD)
        2. .cliver/workflows/<path>   (project-local)
        """
        file_path = Path(path)
        resolved = WorkflowStore._resolve_workflow_path(file_path)
        if resolved is None:
            logger.error("Workflow file not found: %s", file_path)
            return None
        try:
            with open(resolved, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            return Workflow(**data)
        except Exception as e:
            logger.error("Failed to load workflow from file '%s': %s", resolved, e)
            return None

    @staticmethod
    def _resolve_workflow_path(file_path: Path) -> Optional[Path]:
        """Resolve a workflow file path, searching known directories for relative paths."""
        if file_path.is_absolute():
            return file_path if file_path.exists() else None
        if file_path.exists():
            return file_path
        project_local = Path(".cliver/workflows") / file_path
        if project_local.exists():
            return project_local
        return None

    def list_workflows(self) -> List[str]:
        """List all saved workflow names, including sub-workflows."""
        if not self.workflows_dir.is_dir():
            return []
        names = set()
        for p in self.workflows_dir.glob("*.yaml"):
            if not p.stem.endswith(".state"):
                names.add(p.stem)
        sub_dir = self.workflows_dir / "sub-workflows"
        if sub_dir.is_dir():
            for p in sub_dir.glob("*.yaml"):
                if not p.stem.endswith(".state"):
                    names.add(p.stem)
        return sorted(names)

    def list_all_workflows(self) -> List[tuple[str, str, Path]]:
        """List workflows from both global and project-local directories.

        Returns list of (name, source, path) tuples where source is
        'global' or 'project'.
        """
        results: dict[str, tuple[str, str, Path]] = {}

        # Global workflows (from agent profile)
        self._collect_workflows(self.workflows_dir, "global", results)

        # Project-local workflows (.cliver/workflows/ relative to CWD)
        project_dir = Path(".cliver/workflows")
        if project_dir.is_dir() and project_dir.resolve() != self.workflows_dir.resolve():
            self._collect_workflows(project_dir, "project", results)

        return sorted(results.values(), key=lambda x: x[0])

    @staticmethod
    def _collect_workflows(
        directory: Path,
        source: str,
        results: dict[str, tuple[str, str, Path]],
    ) -> None:
        if not directory.is_dir():
            return
        for p in directory.glob("*.yaml"):
            if not p.stem.endswith(".state") and p.stem not in results:
                results[p.stem] = (p.stem, source, p)

    def delete_workflow(self, name: str, checkpointer=None, db_path: Optional[Path] = None) -> bool:
        """Delete a workflow YAML and all its execution data."""
        path = self.workflows_dir / f"{name}.yaml"
        if not path.exists():
            return False
        path.unlink()
        if db_path:
            self._delete_workflow_executions(name, db_path, checkpointer)
        return True

    # -- Execution tracking (workflow_executions table) ------------------------

    @staticmethod
    def _ensure_executions_table(db_path: Path) -> None:
        """Create the workflow_executions table if it doesn't exist."""
        store = get_store(db_path)
        store.execute_schema(_EXECUTIONS_SCHEMA)

    @staticmethod
    def record_execution_start(
        db_path: Path,
        thread_id: str,
        workflow_name: str,
        execution_id: str,
        inputs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record the start of a workflow execution."""
        WorkflowStore._ensure_executions_table(db_path)
        store = get_store(db_path)
        with store.write() as conn:
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
        """Record the completion or failure of a workflow execution."""
        WorkflowStore._ensure_executions_table(db_path)
        store = get_store(db_path)
        with store.write() as conn:
            conn.execute(
                "UPDATE workflow_executions SET status = ?, finished_at = ?, error = ? WHERE thread_id = ?",
                (status, datetime.now(timezone.utc).isoformat(), error, thread_id),
            )

    @staticmethod
    def list_executions(db_path: "str | Path", workflow_name: Optional[str] = None) -> List[dict]:
        """List executions from the workflow_executions table."""
        db_file = Path(db_path)
        if not db_file.exists():
            return []

        WorkflowStore._ensure_executions_table(db_file)
        store = get_store(db_file)
        try:
            with store.read() as conn:
                if workflow_name:
                    rows = conn.execute(
                        "SELECT thread_id, workflow_name, execution_id, status, "
                        "started_at, finished_at, error "
                        "FROM workflow_executions WHERE workflow_name = ? "
                        "ORDER BY started_at DESC",
                        (workflow_name,),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT thread_id, workflow_name, execution_id, status, "
                        "started_at, finished_at, error "
                        "FROM workflow_executions ORDER BY started_at DESC"
                    ).fetchall()
        except Exception as e:
            logger.warning("Could not list executions: %s", e)
            return []

        return [
            {
                "thread_id": r[0] if isinstance(r, (tuple, list)) else r["thread_id"],
                "workflow_name": r[1] if isinstance(r, (tuple, list)) else r["workflow_name"],
                "execution_id": r[2] if isinstance(r, (tuple, list)) else r["execution_id"],
                "status": r[3] if isinstance(r, (tuple, list)) else r["status"],
                "started_at": r[4] if isinstance(r, (tuple, list)) else r["started_at"],
                "finished_at": r[5] if isinstance(r, (tuple, list)) else r["finished_at"],
                "error": r[6] if isinstance(r, (tuple, list)) else r["error"],
            }
            for r in rows
        ]

    # -- Checkpoint cleanup (via LangGraph API) --------------------------------

    @staticmethod
    def _delete_workflow_executions(workflow_name: str, db_path: Path, checkpointer=None) -> int:
        """Delete all execution records and checkpoints for a workflow.

        Uses the workflow_executions table to find thread_ids, then deletes
        checkpoints via LangGraph's checkpointer API.
        """
        db_file = Path(db_path)
        if not db_file.exists():
            return 0

        WorkflowStore._ensure_executions_table(db_file)
        store = get_store(db_file)

        with store.read() as conn:
            rows = conn.execute(
                "SELECT thread_id FROM workflow_executions WHERE workflow_name = ?",
                (workflow_name,),
            ).fetchall()

        thread_ids = [r[0] if isinstance(r, (tuple, list)) else r["thread_id"] for r in rows]

        if checkpointer:
            for tid in thread_ids:
                try:
                    checkpointer.delete_thread(tid)
                except Exception as e:
                    logger.warning("Could not delete checkpoint thread '%s': %s", tid, e)

        with store.write() as conn:
            conn.execute(
                "DELETE FROM workflow_executions WHERE workflow_name = ?",
                (workflow_name,),
            )

        if thread_ids:
            logger.info("Deleted %d executions for workflow '%s'", len(thread_ids), workflow_name)
        return len(thread_ids)

    @staticmethod
    def delete_execution(thread_id: str, db_path: "str | Path", checkpointer=None) -> bool:
        """Delete a single execution's records and checkpoints."""
        db_file = Path(db_path)
        if not db_file.exists():
            return False

        if checkpointer:
            try:
                checkpointer.delete_thread(thread_id)
            except Exception as e:
                logger.warning("Could not delete checkpoint thread '%s': %s", thread_id, e)

        WorkflowStore._ensure_executions_table(db_file)
        store = get_store(db_file)
        with store.write() as conn:
            c = conn.execute("DELETE FROM workflow_executions WHERE thread_id = ?", (thread_id,))
            return c.rowcount > 0

    @staticmethod
    def prune_executions(
        db_path: "str | Path",
        retention_days: int = DEFAULT_RETENTION_DAYS,
        checkpointer=None,
    ) -> int:
        """Delete executions older than retention_days.

        Uses the workflow_executions table for timestamps, then cleans
        checkpoints via LangGraph's checkpointer API.
        """
        db_file = Path(db_path)
        if not db_file.exists():
            return 0

        cutoff = (datetime.now(timezone.utc) - timedelta(days=retention_days)).isoformat()
        WorkflowStore._ensure_executions_table(db_file)
        store = get_store(db_file)

        try:
            with store.read() as conn:
                rows = conn.execute(
                    "SELECT thread_id FROM workflow_executions "
                    "WHERE started_at < ? AND status IN ('completed', 'failed', 'cancelled')",
                    (cutoff,),
                ).fetchall()
        except Exception as e:
            logger.warning("Could not query executions for pruning: %s", e)
            return 0

        stale_threads = [r[0] if isinstance(r, (tuple, list)) else r["thread_id"] for r in rows]
        if not stale_threads:
            return 0

        if checkpointer:
            for tid in stale_threads:
                try:
                    checkpointer.delete_thread(tid)
                except Exception as e:
                    logger.warning("Could not delete checkpoint thread '%s': %s", tid, e)

        with store.write() as conn:
            for tid in stale_threads:
                conn.execute("DELETE FROM workflow_executions WHERE thread_id = ?", (tid,))

        logger.info(
            "Pruned %d executions older than %d days",
            len(stale_threads),
            retention_days,
        )
        return len(stale_threads)
