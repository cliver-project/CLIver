"""
Task Manager — scheduled prompt-based tasks with agent ownership.

A task is a YAML definition that provides a prompt for the LLM,
optional model override, and optional cron schedule. Tasks are stored
per-agent in {config_dir}/agents/{agent_name}/tasks/.

Task YAML format:
    name: daily-research
    description: Research AI trends daily
    prompt: "Research the latest AI trends and summarize the top 3 developments"
    model: deepseek-r1           # optional model override
    schedule: "0 9 * * *"        # optional cron expression
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class TaskOrigin(BaseModel):
    """Where a task was created and where results should be delivered."""

    source: str = Field(..., description="Origin source: cli, api, slack, telegram, discord, feishu")
    platform: Optional[str] = Field(None, description="Adapter name for IM sources")
    channel_id: Optional[str] = Field(None, description="IM channel to reply to")
    thread_id: Optional[str] = Field(None, description="IM thread to reply in")
    user_id: Optional[str] = Field(None, description="Who created the task")
    session_key: Optional[str] = Field(None, description="Session key for history lookup")


class TaskDefinition(BaseModel):
    """A task definition — a prompt with optional workflow and skill activation."""

    name: str = Field(..., description="Unique task name")
    description: Optional[str] = Field(None, description="What this task does")
    prompt: str = Field(..., description="The prompt to send to the LLM")
    workflow: Optional[str] = Field(
        None, description="Workflow name to execute (if set, runs workflow instead of chat)"
    )
    workflow_inputs: Optional[Dict[str, Any]] = Field(None, description="Extra inputs for workflow execution")
    skills: Optional[List[str]] = Field(None, description="Skills to pre-activate in system prompt")
    model: Optional[str] = Field(None, description="Model override for this task")
    schedule: Optional[str] = Field(None, description="Cron expression for recurring execution")
    run_at: Optional[str] = Field(None, description="ISO 8601 datetime for one-shot execution")
    permissions: Optional[Any] = Field(None, description="Permission overrides for this task")
    origin: Optional[TaskOrigin] = Field(None, description="Where the task was created from")


class TaskRun(BaseModel):
    """Record of a single task execution."""

    task_name: str
    execution_id: str
    status: str  # completed, failed, cancelled
    started_at: str
    finished_at: Optional[str] = None
    error: Optional[str] = None
    result: Optional[str] = None


class TaskManager:
    """Manages task definitions for a specific agent.

    Directory layout:
        {tasks_dir}/
        ├── daily-research.yaml        # task definition
        └── code-review.yaml

    Run history is stored in gateway.db via TaskRunStore.
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
        try:
            return self._load_task_file(path)
        except Exception as e:
            logger.warning("Invalid task file %s: %s", path, e)
            return None

    def save_task(self, task: TaskDefinition) -> Path:
        """Save a task definition. Creates or overwrites.

        Automatically attaches IM origin if running inside a gateway
        message handler and the task doesn't already have an origin.
        """
        if not task.origin:
            task.origin = self._resolve_im_origin()
        self._ensure_dir()
        path = self.tasks_dir / f"{task.name}.yaml"
        data = task.model_dump(exclude_none=True)
        path.write_text(yaml.dump(data, default_flow_style=False, allow_unicode=True), encoding="utf-8")
        logger.info(f"Task '{task.name}' saved to {path}")
        return path

    @staticmethod
    def _resolve_im_origin() -> Optional[TaskOrigin]:
        """Auto-detect IM origin from gateway context if available."""
        try:
            from cliver.gateway.gateway import im_context

            ctx = im_context.get()
            if ctx:
                return TaskOrigin(
                    source=ctx["platform"],
                    platform=ctx["platform"],
                    channel_id=ctx.get("channel_id"),
                    thread_id=ctx.get("thread_id"),
                    user_id=ctx.get("user_id"),
                    session_key=ctx.get("session_key"),
                )
        except ImportError:
            pass
        return None

    def remove_task(self, name: str) -> bool:
        """Remove a task definition."""
        path = self.tasks_dir / f"{name}.yaml"
        if not path.exists():
            return False
        path.unlink()
        return True

    # -- Helpers ------------------------------------------------------------

    @staticmethod
    def _load_task_file(path: Path) -> Optional[TaskDefinition]:
        content = path.read_text(encoding="utf-8")
        data = yaml.safe_load(content)
        if not data or not isinstance(data, dict):
            return None
        return TaskDefinition(**data)

    @staticmethod
    def timestamp_now() -> str:
        from cliver.util import format_datetime

        return format_datetime(fmt="%Y-%m-%d %H:%M:%S")
