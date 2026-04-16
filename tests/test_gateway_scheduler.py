"""Tests for the gateway cron scheduler."""

from unittest.mock import AsyncMock

import pytest

from cliver.gateway.scheduler import CronScheduler
from cliver.task_manager import TaskDefinition, TaskManager


@pytest.fixture
def tasks_dir(tmp_path):
    d = tmp_path / "tasks"
    d.mkdir()
    return d


@pytest.fixture
def task_manager(tasks_dir):
    return TaskManager(tasks_dir)


@pytest.fixture
def cron_state_path(tmp_path):
    return tmp_path / "cron-state.json"


class TestCronScheduler:
    def test_find_due_tasks(self, task_manager, cron_state_path):
        """Tasks with a schedule that is due should be returned."""
        task_manager.save_task(
            TaskDefinition(
                name="every-minute",
                prompt="test prompt",
                schedule="* * * * *",
            )
        )
        task_manager.save_task(
            TaskDefinition(
                name="no-schedule",
                prompt="test prompt",
            )
        )

        scheduler = CronScheduler(
            task_manager=task_manager,
            cron_state_path=cron_state_path,
            run_task_fn=AsyncMock(),
        )
        due = scheduler.find_due_tasks()
        due_names = [t.name for t in due]
        assert "every-minute" in due_names
        assert "no-schedule" not in due_names

    def test_cron_state_prevents_rerun(self, task_manager, cron_state_path):
        """A task that just ran should not be due again within the same minute."""
        task_manager.save_task(
            TaskDefinition(
                name="once-per-minute",
                prompt="test prompt",
                schedule="* * * * *",
            )
        )

        scheduler = CronScheduler(
            task_manager=task_manager,
            cron_state_path=cron_state_path,
            run_task_fn=AsyncMock(),
        )

        # First check: should be due
        due = scheduler.find_due_tasks()
        assert len(due) == 1

        # Mark as run
        scheduler.mark_task_run("once-per-minute")

        # Second check: should NOT be due
        due = scheduler.find_due_tasks()
        assert len(due) == 0

    def test_cron_state_persists(self, task_manager, cron_state_path):
        """Cron state is saved to and loaded from disk."""
        scheduler = CronScheduler(
            task_manager=task_manager,
            cron_state_path=cron_state_path,
            run_task_fn=AsyncMock(),
        )
        scheduler.mark_task_run("test-task")

        # Create a new scheduler from the same state file
        scheduler2 = CronScheduler(
            task_manager=task_manager,
            cron_state_path=cron_state_path,
            run_task_fn=AsyncMock(),
        )
        assert "test-task" in scheduler2._state

    @pytest.mark.asyncio
    async def test_tick_runs_due_tasks(self, task_manager, cron_state_path):
        """A single tick should execute due tasks."""
        task_manager.save_task(
            TaskDefinition(
                name="tick-test",
                prompt="test prompt",
                schedule="* * * * *",
            )
        )

        run_fn = AsyncMock()
        scheduler = CronScheduler(
            task_manager=task_manager,
            cron_state_path=cron_state_path,
            run_task_fn=run_fn,
        )
        await scheduler.tick()

        run_fn.assert_called_once()
        call_args = run_fn.call_args[0]
        assert call_args[0].name == "tick-test"

    @pytest.mark.asyncio
    async def test_tick_handles_errors(self, task_manager, cron_state_path):
        """A failing task should not stop the scheduler tick."""
        task_manager.save_task(
            TaskDefinition(
                name="failing-task",
                prompt="test prompt",
                schedule="* * * * *",
            )
        )

        run_fn = AsyncMock(side_effect=RuntimeError("workflow exploded"))
        scheduler = CronScheduler(
            task_manager=task_manager,
            cron_state_path=cron_state_path,
            run_task_fn=run_fn,
        )
        # Should not raise
        await scheduler.tick()
        # Task should still be marked as run (to avoid retry storm)
        assert "failing-task" in scheduler._state
