"""Cron scheduler for the gateway daemon.

Ticks every 60 seconds, checks all tasks with a schedule field,
and runs any that are due via the provided run_task_fn callback.
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Coroutine, List

from croniter import croniter

from cliver.task_manager import TaskDefinition, TaskManager

logger = logging.getLogger(__name__)

# Type for the task execution callback
RunTaskFn = Callable[[TaskDefinition], Coroutine[Any, Any, None]]


class CronScheduler:
    """Evaluates cron schedules and dispatches due tasks.

    State is persisted to a JSON file to prevent double-execution
    after gateway restarts.
    """

    def __init__(
        self,
        task_manager: TaskManager,
        cron_state_path: Path,
        run_task_fn: RunTaskFn,
    ):
        self.task_manager = task_manager
        self.cron_state_path = Path(cron_state_path)
        self.run_task_fn = run_task_fn
        self._state: dict[str, float] = self._load_state()

    def find_due_tasks(self) -> List[TaskDefinition]:
        """Return tasks whose cron schedule is due since their last run."""
        now = datetime.now(timezone.utc)
        due = []

        for task in self.task_manager.list_tasks():
            if not task.schedule:
                continue

            last_run = self._state.get(task.name, 0.0)
            try:
                cron = croniter(task.schedule, datetime.fromtimestamp(last_run, tz=timezone.utc))
                next_run = cron.get_next(datetime)
                if next_run <= now:
                    due.append(task)
            except (ValueError, KeyError) as e:
                logger.warning(f"Invalid cron expression for task '{task.name}': {e}")

        return due

    def mark_task_run(self, task_name: str) -> None:
        """Record that a task has run at the current time."""
        self._state[task_name] = time.time()
        self._save_state()

    async def tick(self) -> int:
        """Run one scheduler tick: find due tasks and execute them.

        Returns the number of tasks executed.
        """
        due_tasks = self.find_due_tasks()
        if not due_tasks:
            return 0

        executed = 0
        for task in due_tasks:
            logger.info(f"Cron: executing task '{task.name}'")
            try:
                await self.run_task_fn(task)
            except Exception as e:
                logger.error(f"Cron: task '{task.name}' failed: {e}")
            finally:
                # Mark as run even on failure to avoid retry storms
                self.mark_task_run(task.name)
                executed += 1

        return executed

    def _load_state(self) -> dict[str, float]:
        """Load last-run timestamps from disk."""
        if not self.cron_state_path.exists():
            return {}
        try:
            data = json.loads(self.cron_state_path.read_text(encoding="utf-8"))
            return {k: float(v) for k, v in data.items()}
        except Exception as e:
            logger.warning(f"Could not load cron state: {e}")
            return {}

    def _save_state(self) -> None:
        """Persist last-run timestamps to disk."""
        self.cron_state_path.parent.mkdir(parents=True, exist_ok=True)
        self.cron_state_path.write_text(json.dumps(self._state), encoding="utf-8")
