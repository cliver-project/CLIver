"""Built-in create_task tool — thin wrapper that auto-attaches IM origin."""

import logging
from pathlib import Path
from typing import Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from cliver.agent_profile import get_current_profile
from cliver.task_manager import TaskDefinition, TaskManager

logger = logging.getLogger(__name__)


class CreateTaskInput(BaseModel):
    name: str = Field(..., description="Task name (snake_case)")
    prompt: str = Field(..., description="Prompt the LLM receives when the task runs")
    description: Optional[str] = Field(None, description="What this task does")
    model: Optional[str] = Field(None, description="Model override")
    schedule: Optional[str] = Field(None, description="Cron expression for recurring execution")
    run_at: Optional[str] = Field(None, description="ISO 8601 datetime for one-shot execution")


class CreateTaskTool(BaseTool):
    name: str = "CreateTask"
    description: str = (
        "Create a task. Use when the user wants to schedule or defer execution. "
        "IM origin is auto-attached for reply-back."
    )
    args_schema: Type[BaseModel] = CreateTaskInput
    tags: list = ["task", "scheduling"]
    tasks_dir: Optional[Path] = None

    def _run(
        self,
        name: str,
        prompt: str,
        description: Optional[str] = None,
        model: Optional[str] = None,
        schedule: Optional[str] = None,
        run_at: Optional[str] = None,
    ) -> str:
        tasks_dir = self.tasks_dir
        if tasks_dir is None:
            profile = get_current_profile()
            if profile is None:
                return "Error: no agent profile available."
            tasks_dir = profile.tasks_dir

        if run_at:
            from datetime import datetime

            try:
                datetime.fromisoformat(run_at)
            except ValueError:
                return f"Error: invalid run_at '{run_at}'. Use ISO 8601 (e.g. 2026-04-25T14:30:00)."

        task = TaskDefinition(
            name=name,
            description=description,
            prompt=prompt,
            model=model,
            schedule=schedule,
            run_at=run_at,
        )

        manager = TaskManager(tasks_dir)
        path = manager.save_task(task)

        origin_info = f" (reply-back: {task.origin.source})" if task.origin else ""
        if schedule:
            schedule_info = f" (schedule: {schedule})"
        elif run_at:
            schedule_info = f" (run at: {run_at})"
        else:
            schedule_info = " (manual)"
        return f"Task '{name}' created{schedule_info}{origin_info}: {path}"


create_task = CreateTaskTool()
