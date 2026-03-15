"""
Task Manager — thin wrappers around workflow executions with agent ownership.

A task is a YAML definition that references a workflow, provides default
inputs, and optionally specifies a cron schedule. Tasks are stored
per-agent in {config_dir}/agents/{agent_name}/tasks/.

Task YAML format:
    name: daily-research
    description: Research AI trends daily
    workflow: research_analysis
    inputs:
      topic: "AI trends"
    schedule: "0 9 * * *"    # optional cron expression
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class TaskDefinition(BaseModel):
    """A task definition — references a workflow with inputs and optional schedule."""

    name: str = Field(..., description="Unique task name")
    description: Optional[str] = Field(None, description="What this task does")
    workflow: str = Field(..., description="Name of the workflow to execute")
    inputs: Optional[Dict[str, Any]] = Field(None, description="Default inputs for the workflow")
    schedule: Optional[str] = Field(None, description="Cron expression for recurring execution")


class TaskRun(BaseModel):
    """Record of a single task execution."""

    task_name: str
    execution_id: str
    workflow_name: str
    status: str  # completed, failed, cancelled
    started_at: str
    finished_at: Optional[str] = None
    error: Optional[str] = None


class TaskManager:
    """Manages tasks for a specific agent.

    Directory layout:
        {tasks_dir}/
        ├── daily-research.yaml        # task definition
        ├── daily-research.runs.yaml   # execution history
        └── code-review.yaml
    """

    def __init__(self, tasks_dir: Path):
        self.tasks_dir = tasks_dir

    def _ensure_dir(self) -> None:
        self.tasks_dir.mkdir(parents=True, exist_ok=True)

    # -- CRUD ---------------------------------------------------------------

    def list_tasks(self) -> List[TaskDefinition]:
        """List all task definitions."""
        tasks = []
        if not self.tasks_dir.is_dir():
            return tasks
        for path in sorted(self.tasks_dir.glob("*.yaml")):
            if path.name.endswith(".runs.yaml"):
                continue
            try:
                task = self._load_task_file(path)
                if task:
                    tasks.append(task)
            except Exception as e:
                logger.warning(f"Failed to load task {path}: {e}")
        return tasks

    def get_task(self, name: str) -> Optional[TaskDefinition]:
        """Get a task definition by name."""
        path = self.tasks_dir / f"{name}.yaml"
        if not path.exists():
            return None
        return self._load_task_file(path)

    def save_task(self, task: TaskDefinition) -> Path:
        """Save a task definition. Creates or overwrites."""
        self._ensure_dir()
        path = self.tasks_dir / f"{task.name}.yaml"
        data = task.model_dump(exclude_none=True)
        path.write_text(yaml.dump(data, default_flow_style=False, allow_unicode=True), encoding="utf-8")
        logger.info(f"Task '{task.name}' saved to {path}")
        return path

    def remove_task(self, name: str) -> bool:
        """Remove a task definition and its run history."""
        path = self.tasks_dir / f"{name}.yaml"
        runs_path = self.tasks_dir / f"{name}.runs.yaml"
        removed = False
        if path.exists():
            path.unlink()
            removed = True
        if runs_path.exists():
            runs_path.unlink()
        return removed

    # -- Run history --------------------------------------------------------

    def record_run(self, run: TaskRun) -> None:
        """Append a run record to the task's execution history."""
        self._ensure_dir()
        runs_path = self.tasks_dir / f"{run.task_name}.runs.yaml"
        runs = self._load_runs(runs_path)
        runs.append(run.model_dump())
        runs_path.write_text(
            yaml.dump(runs, default_flow_style=False, allow_unicode=True),
            encoding="utf-8",
        )

    def get_runs(self, task_name: str, limit: int = 10) -> List[TaskRun]:
        """Get recent run history for a task."""
        runs_path = self.tasks_dir / f"{task_name}.runs.yaml"
        raw_runs = self._load_runs(runs_path)
        # Most recent first, limited
        return [TaskRun(**r) for r in reversed(raw_runs[-limit:])]

    # -- Helpers ------------------------------------------------------------

    @staticmethod
    def _load_task_file(path: Path) -> Optional[TaskDefinition]:
        content = path.read_text(encoding="utf-8")
        data = yaml.safe_load(content)
        if not data or not isinstance(data, dict):
            return None
        return TaskDefinition(**data)

    @staticmethod
    def _load_runs(path: Path) -> list:
        if not path.exists():
            return []
        content = path.read_text(encoding="utf-8")
        data = yaml.safe_load(content)
        return data if isinstance(data, list) else []

    @staticmethod
    def timestamp_now() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
