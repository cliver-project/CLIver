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
    name: str = Field(..., description="Task name in snake_case (e.g. 'daily_greeting')")
    prompt: str = Field(
        ...,
        description="The text/instruction the LLM will execute when the task runs. "
        "This is the task content — NOT called 'content' or 'command'.",
    )
    description: Optional[str] = Field(None, description="What this task does")
    model: Optional[str] = Field(None, description="Model override")
    schedule: Optional[str] = Field(
        None,
        description="Cron expression for RECURRING tasks ONLY (e.g. '0 9 * * *' for daily 9am). "
        "Do NOT put a datetime here — use run_at instead.",
    )
    run_at: Optional[str] = Field(
        None,
        description="ISO 8601 datetime for ONE-TIME execution (e.g. '2026-04-25T14:30:00'). "
        "Use this for 'run at a specific time', 'in 5 minutes', 'at 3pm', etc.",
    )


class CreateTaskTool(BaseTool):
    name: str = "CreateTask"
    description: str = (
        "Create a scheduled or deferred task. "
        "Required parameters: 'name' (snake_case identifier) and 'prompt' (the instruction text). "
        "Do NOT use 'content' or 'command' — the field is called 'prompt'. "
        "Optional: 'schedule' (cron), 'run_at' (ISO 8601 datetime), 'model', 'description'. "
        "ALWAYS use this tool instead of shell commands like 'cliver task create'. "
        "This tool auto-attaches IM origin so task results are delivered back to the conversation."
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

        # Auto-correct: if schedule looks like a datetime, move it to run_at
        if schedule and not run_at:
            from datetime import datetime

            try:
                datetime.fromisoformat(schedule)
                run_at = schedule
                schedule = None
            except ValueError:
                pass

        if run_at:
            from datetime import datetime

            try:
                datetime.fromisoformat(run_at)
            except ValueError:
                return f"Error: invalid run_at '{run_at}'. Use ISO 8601 (e.g. 2026-04-25T14:30:00)."

        # Resolve IM context (origin + session_id) before creating the task
        im_origin, im_session_id = self._resolve_im_context()

        task = TaskDefinition(
            name=name,
            description=description,
            prompt=prompt,
            model=model,
            schedule=schedule,
            run_at=run_at,
            session_id=im_session_id,
        )

        from cliver.gateway.task_store import TaskStore

        profile = get_current_profile()
        db_path = profile.agent_dir / "gateway.db" if profile else tasks_dir.parent / "gateway.db"
        store = TaskStore(db_path)
        manager = TaskManager(tasks_dir, store)
        path = manager.save_task(task)

        # Save origin to database (not YAML) for IM reply-back
        origin_info = ""
        if im_origin:
            try:
                store.save_origin(name, im_origin)
                origin_info = f" (reply-back: {im_origin.source})"
            except Exception as e:
                logger.warning("Failed to save task origin to DB: %s", e)
        if schedule:
            schedule_info = f" (schedule: {schedule})"
        elif run_at:
            schedule_info = f" (run at: {run_at})"
        else:
            schedule_info = " (manual)"
        return f"Task '{name}' created{schedule_info}{origin_info}: {path}"

    @staticmethod
    def _resolve_im_context():
        """Auto-detect IM origin and session_id from gateway context.

        Returns (TaskOrigin, session_id) or (None, None).
        """
        try:
            from cliver.gateway.gateway import im_context
            from cliver.task_manager import TaskOrigin

            ctx = im_context.get()
            if ctx:
                origin = TaskOrigin(
                    source=ctx["platform"],
                    platform=ctx["platform"],
                    channel_id=ctx.get("channel_id"),
                    thread_id=ctx.get("thread_id"),
                    user_id=ctx.get("user_id"),
                )
                return origin, ctx.get("session_id")
        except ImportError:
            pass
        return None, None


create_task = CreateTaskTool()
