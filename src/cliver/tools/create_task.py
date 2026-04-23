"""Built-in create_task tool for creating scheduled or one-shot tasks.

When called from an IM conversation (gateway), automatically attaches
the IM origin context so task results can be delivered back to the
originating thread.
"""

import logging
from pathlib import Path
from typing import Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from cliver.agent_profile import get_current_profile
from cliver.task_manager import TaskDefinition, TaskManager, TaskOrigin

logger = logging.getLogger(__name__)


class CreateTaskInput(BaseModel):
    name: str = Field(..., description="Unique task name")
    prompt: str = Field(..., description="The prompt to execute")
    description: Optional[str] = Field(None, description="What this task does")
    model: Optional[str] = Field(None, description="Model override")
    schedule: Optional[str] = Field(None, description="Cron expression for recurring execution")


class CreateTaskTool(BaseTool):
    name: str = "CreateTask"
    description: str = (
        "Create a new task. Tasks can be one-shot or scheduled (with a cron expression). "
        "Scheduled tasks run automatically via the gateway daemon. "
        "If you are in an IM conversation, the task will automatically reply back "
        "to this thread when it completes."
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
    ) -> str:
        tasks_dir = self.tasks_dir
        if tasks_dir is None:
            profile = get_current_profile()
            if profile is None:
                return "Error: no agent profile available."
            tasks_dir = profile.tasks_dir

        origin = self._resolve_origin()

        task = TaskDefinition(
            name=name,
            description=description,
            prompt=prompt,
            model=model,
            schedule=schedule,
            origin=origin,
        )

        manager = TaskManager(tasks_dir)
        path = manager.save_task(task)

        origin_info = f" (reply-back: {origin.source})" if origin else ""
        schedule_info = f" (schedule: {schedule})" if schedule else " (one-shot)"
        return f"Task '{name}' created{schedule_info}{origin_info}: {path}"

    @staticmethod
    def _resolve_origin() -> Optional[TaskOrigin]:
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


create_task = CreateTaskTool()
