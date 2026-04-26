"""Cron scheduler for the gateway daemon.

Ticks every 60 seconds, checks all tasks with a schedule field,
and runs any that are due. "Last run" is derived from the task_runs
table in gateway.db — no separate state file needed.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine, List

from croniter import croniter

from cliver.gateway.task_store import TaskStore
from cliver.task_manager import TaskDefinition, TaskManager

logger = logging.getLogger(__name__)

RunTaskFn = Callable[[TaskDefinition], Coroutine[Any, Any, None]]


class CronScheduler:
    """Evaluates cron schedules and dispatches due tasks.

    Uses TaskStore to determine when each task last ran —
    no separate state file is needed.
    """

    def __init__(
        self,
        task_manager: TaskManager,
        run_store: TaskStore,
        run_task_fn: RunTaskFn,
    ):
        self.task_manager = task_manager
        self.run_store = run_store
        self.run_task_fn = run_task_fn

    def find_due_tasks(self) -> List[TaskDefinition]:
        """Return tasks whose cron schedule or run_at datetime is due."""
        now = datetime.now(timezone.utc)
        due = []

        for task in self.task_manager.list_tasks():
            # One-shot: run_at datetime
            if task.run_at:
                try:
                    scheduled = datetime.fromisoformat(task.run_at)
                    if scheduled.tzinfo is None:
                        from cliver.util import get_effective_timezone

                        scheduled = scheduled.replace(tzinfo=get_effective_timezone())
                    scheduled = scheduled.astimezone(timezone.utc)
                    if scheduled <= now:
                        last_run = self.run_store.get_last_run_time(task.name) or 0.0
                        if last_run < scheduled.timestamp():
                            due.append(task)
                except ValueError as e:
                    logger.warning(f"Invalid run_at for task '{task.name}': {e}")
                continue

            # Recurring: cron schedule
            if not task.schedule:
                continue

            # Auto-correct: if schedule is a datetime, treat as run_at
            try:
                scheduled = datetime.fromisoformat(task.schedule)
                if scheduled.tzinfo is None:
                    from cliver.util import get_effective_timezone

                    scheduled = scheduled.replace(tzinfo=get_effective_timezone())
                scheduled = scheduled.astimezone(timezone.utc)
                if scheduled <= now:
                    last_run = self.run_store.get_last_run_time(task.name) or 0.0
                    if last_run < scheduled.timestamp():
                        due.append(task)
                continue
            except ValueError:
                pass

            last_run = self.run_store.get_last_run_time(task.name) or 0.0
            try:
                cron = croniter(task.schedule, datetime.fromtimestamp(last_run, tz=timezone.utc))
                next_run = cron.get_next(datetime)
                if next_run <= now:
                    due.append(task)
            except (ValueError, KeyError) as e:
                logger.warning(f"Invalid cron expression for task '{task.name}': {e}")

        return due

    async def tick(self) -> int:
        """Run one scheduler tick: find due tasks and execute them.

        Returns the number of tasks executed.
        """
        due_tasks = self.find_due_tasks()
        if not due_tasks:
            return 0

        tasks = []
        for task in due_tasks:
            logger.info(f"Cron: dispatching task '{task.name}'")
            coro = self._run_one(task)
            tasks.append(asyncio.create_task(coro, name=f"cron:{task.name}"))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for task_def, result in zip(due_tasks, results, strict=False):
            if isinstance(result, Exception):
                logger.error(f"Cron: task '{task_def.name}' failed: {result}")

        return len(due_tasks)

    async def _run_one(self, task: TaskDefinition) -> None:
        """Run a single task in its own asyncio.Task for ContextVar isolation."""
        try:
            await self.run_task_fn(task)
        except Exception as e:
            logger.error(f"Cron: task '{task.name}' failed: {e}")
            raise

    def validate_tasks(self) -> None:
        """Log warnings for tasks with invalid configurations.

        Checks:
        - Invalid cron expressions
        - Empty prompts
        - Tasks with schedule but no prompt
        """
        for task in self.task_manager.list_tasks():
            if not task.prompt or not task.prompt.strip():
                logger.warning(
                    "Task '%s' has an empty prompt — it will not produce useful results",
                    task.name,
                )

            if task.schedule:
                try:
                    croniter(task.schedule)
                except (ValueError, KeyError):
                    logger.warning(
                        "Task '%s' has invalid cron expression '%s' — it will never be scheduled",
                        task.name,
                        task.schedule,
                    )

            if task.run_at:
                try:
                    datetime.fromisoformat(task.run_at)
                except ValueError:
                    logger.warning(
                        "Task '%s' has invalid run_at '%s' — it will never be scheduled",
                        task.name,
                        task.run_at,
                    )

    def cleanup_orphan_runs(self) -> int:
        """Delete run records for tasks not registered in the database.

        Returns the number of orphan records deleted.
        """
        registered_names = {r["name"] for r in self.run_store.list_registered_tasks()}
        recorded_names = set(self.run_store.get_all_task_names())
        orphans = recorded_names - registered_names

        total_deleted = 0
        for name in orphans:
            deleted = self.run_store.delete_runs(name)
            total_deleted += deleted
            logger.info(
                "Cleaned up %d orphan run record(s) for unregistered task '%s'",
                deleted,
                name,
            )

        return total_deleted
