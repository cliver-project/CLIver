"""Thin workflow executor -- wraps LangGraph graph.ainvoke()."""

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from cliver.workflow.compiler import WorkflowCompiler
from cliver.workflow.persistence import WorkflowStore
from cliver.workflow.workflow_models import Workflow

logger = logging.getLogger(__name__)


class WorkflowExecutor:
    """Execute compiled workflows via LangGraph."""

    def __init__(
        self,
        app_config=None,
        workflows_dir: Optional[Path] = None,
        on_tool_event=None,
    ):
        self.app_config = app_config
        self.on_tool_event = on_tool_event

        if app_config and getattr(app_config, "workflow_runs_dir", None):
            self._runs_dir = Path(app_config.workflow_runs_dir)
        else:
            self._runs_dir = Path.home() / ".cliver" / "workflow-runs"

        self._compiler = WorkflowCompiler()
        self._store = WorkflowStore(workflows_dir) if workflows_dir else None

    def _workflow_dir(self, workflow_id: str) -> Path:
        return self._runs_dir / workflow_id

    def _make_outputs_dir(self, workflow_id: str, thread_id: str) -> Path:
        path = self._workflow_dir(workflow_id) / thread_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _checkpoints_db_path(self, workflow_id: str, create: bool = False) -> Path:
        path = self._workflow_dir(workflow_id) / "checkpoints"
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path / "checkpoints.db"

    def _executions_db_path(self, workflow_id: str, create: bool = False) -> Path:
        path = self._workflow_dir(workflow_id)
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path / "executions.db"

    async def _get_checkpointer(self, workflow_id: str):
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        db_path = self._checkpoints_db_path(workflow_id, create=True)
        return AsyncSqliteSaver.from_conn_string(str(db_path))

    @staticmethod
    def generate_execution_id() -> str:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        short_id = uuid.uuid4().hex[:8]
        return f"{ts}_{short_id}"

    async def execute(
        self,
        workflow: Workflow,
        inputs: Optional[Dict[str, Any]] = None,
        execution_id: Optional[str] = None,
        base_dir: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Execute a workflow and return the final state."""
        execution_id = execution_id or self.generate_execution_id()
        workflow_id = workflow.name

        outputs_dir = self._make_outputs_dir(workflow_id, execution_id)
        exec_db = self._executions_db_path(workflow_id, create=True)
        merged_inputs = workflow.get_initial_inputs(inputs)

        checkpointer = await self._get_checkpointer(workflow_id)

        try:
            graph = self._compiler.compile(workflow, checkpointer=checkpointer)

            initial_state = {
                "inputs": merged_inputs,
                "steps": {},
                "workflow_id": workflow_id,
                "thread_id": execution_id,
                "outputs_dir": str(outputs_dir),
                "error": None,
                "_app_config": self.app_config,
                "_on_tool_event": self.on_tool_event,
                "_workflow_base_dir": base_dir or ".",
            }

            config = {"configurable": {"thread_id": execution_id}}

            WorkflowStore.record_execution_start(exec_db, execution_id, workflow_id, execution_id, merged_inputs)

            self._log_event(
                outputs_dir,
                "workflow_start",
                {
                    "workflow_id": workflow_id,
                    "execution_id": execution_id,
                    "inputs": merged_inputs,
                },
            )

            result = await graph.ainvoke(initial_state, config)

            WorkflowStore.record_execution_end(exec_db, execution_id, "completed")
            self._log_event(outputs_dir, "workflow_end", {"status": "completed"})

            return result

        except Exception as e:
            logger.error("Workflow %s failed: %s", workflow_id, e)
            WorkflowStore.record_execution_end(exec_db, execution_id, "failed", str(e))
            self._log_event(outputs_dir, "workflow_end", {"status": "failed", "error": str(e)})
            raise
        finally:
            if hasattr(checkpointer, "conn") and checkpointer.conn:
                await checkpointer.conn.close()

    async def execute_by_name(
        self,
        workflow_name: str,
        inputs: Optional[Dict[str, Any]] = None,
        execution_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Load a workflow by name and execute it."""
        if not self._store:
            raise ValueError("No workflows_dir configured -- cannot load by name")
        wf = self._store.load_workflow(workflow_name)
        if not wf:
            raise ValueError(f"Workflow '{workflow_name}' not found")
        return await self.execute(
            wf,
            inputs=inputs,
            execution_id=execution_id,
            base_dir=str(self._store.workflows_dir),
        )

    async def resume(
        self,
        workflow_name: str,
        execution_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Resume a workflow from its last checkpoint."""
        if not self._store:
            raise ValueError("No workflows_dir configured")
        wf = self._store.load_workflow(workflow_name)
        if not wf:
            raise ValueError(f"Workflow '{workflow_name}' not found")

        checkpointer = await self._get_checkpointer(workflow_name)
        try:
            graph = self._compiler.compile(wf, checkpointer=checkpointer)
            config = {"configurable": {"thread_id": execution_id}}
            result = await graph.ainvoke(None, config)
            exec_db = self._executions_db_path(workflow_name)
            WorkflowStore.record_execution_end(exec_db, execution_id, "completed")
            return result
        finally:
            if hasattr(checkpointer, "conn") and checkpointer.conn:
                await checkpointer.conn.close()

    async def get_history(self, workflow_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """List execution history."""
        db_paths = []
        if workflow_name:
            db_path = self._executions_db_path(workflow_name)
            if db_path.exists():
                db_paths.append(db_path)
        else:
            if self._runs_dir.exists():
                for wf_dir in self._runs_dir.iterdir():
                    if wf_dir.is_dir():
                        exec_db = wf_dir / "executions.db"
                        if exec_db.exists():
                            db_paths.append(exec_db)

        all_executions = []
        for db_path in db_paths:
            execs = WorkflowStore.list_executions(db_path, workflow_name)
            all_executions.extend(execs)
        return sorted(all_executions, key=lambda e: e.get("started_at", ""), reverse=True)

    async def get_status(self, execution_id: str, workflow_name: str) -> Optional[Dict[str, Any]]:
        """Get per-step status of a run."""
        exec_db = self._executions_db_path(workflow_name)
        if not exec_db.exists():
            return None
        execs = WorkflowStore.list_executions(exec_db, workflow_name)
        match = next((e for e in execs if e.get("thread_id") == execution_id), None)
        if not match:
            return None

        outputs_dir = self._runs_dir / workflow_name / execution_id
        step_statuses = {}
        if outputs_dir.exists():
            for f in outputs_dir.iterdir():
                if f.is_dir():
                    step_statuses[f.name] = "completed"

        match["step_statuses"] = step_statuses
        return match

    async def delete_run(self, execution_id: str, workflow_name: str) -> bool:
        """Delete a single run's data and outputs."""
        import shutil

        exec_db = self._executions_db_path(workflow_name)
        if exec_db.exists():
            WorkflowStore.delete_execution(execution_id, exec_db)
        run_dir = self._runs_dir / workflow_name / execution_id
        if run_dir.exists():
            shutil.rmtree(run_dir)
            return True
        return False

    async def delete_workflow(self, workflow_name: str) -> bool:
        """Delete a workflow definition and all its runs (checkpoints + outputs)."""
        import shutil

        deleted = False
        if self._store and self._store.delete_workflow(workflow_name):
            deleted = True

        project_path = Path.cwd() / ".cliver" / "workflows" / f"{workflow_name}.yaml"
        if project_path.exists():
            project_path.unlink()
            deleted = True

        wf_dir = self._runs_dir / workflow_name
        if wf_dir.exists():
            shutil.rmtree(wf_dir)
            deleted = True

        return deleted

    async def prune(self, retention_days: int = 300) -> int:
        """Delete runs older than retention_days."""
        count = 0
        if not self._runs_dir.exists():
            return 0
        for wf_dir in self._runs_dir.iterdir():
            if wf_dir.is_dir():
                exec_db = wf_dir / "executions.db"
                if exec_db.exists():
                    count += WorkflowStore.prune_executions(exec_db, retention_days)
        return count

    @staticmethod
    def _log_event(outputs_dir: Path, event_type: str, data: dict):
        log_file = outputs_dir / "workflow.log"
        entry = {"type": event_type, "timestamp": datetime.now(timezone.utc).isoformat(), **data}
        with open(log_file, "a") as f:
            f.write(json.dumps(entry, default=str, ensure_ascii=False) + "\n")
