"""Tests for the gateway cron scheduler."""

from unittest.mock import AsyncMock

import pytest

from cliver.gateway.scheduler import CronScheduler
from cliver.gateway.task_run_store import TaskRunStore
from cliver.task_manager import TaskDefinition, TaskManager, TaskRun


@pytest.fixture
def tasks_dir(tmp_path):
    d = tmp_path / "tasks"
    d.mkdir()
    return d


@pytest.fixture
def task_manager(tasks_dir):
    return TaskManager(tasks_dir)


@pytest.fixture
def run_store(tmp_path):
    return TaskRunStore(tmp_path / "gateway.db")


class TestCronScheduler:
    def test_find_due_tasks(self, task_manager, run_store):
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
            run_store=run_store,
            run_task_fn=AsyncMock(),
        )
        due = scheduler.find_due_tasks()
        due_names = [t.name for t in due]
        assert "every-minute" in due_names
        assert "no-schedule" not in due_names

    def test_run_store_prevents_rerun(self, task_manager, run_store):
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
            run_store=run_store,
            run_task_fn=AsyncMock(),
        )

        # First check: should be due
        due = scheduler.find_due_tasks()
        assert len(due) == 1

        # Record a run in the store
        run_store.record_run(
            TaskRun(
                task_name="once-per-minute",
                execution_id="e1",
                status="completed",
                started_at=TaskManager.timestamp_now(),
            )
        )

        # Second check: should NOT be due
        due = scheduler.find_due_tasks()
        assert len(due) == 0

    @pytest.mark.asyncio
    async def test_tick_runs_due_tasks(self, task_manager, run_store):
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
            run_store=run_store,
            run_task_fn=run_fn,
        )
        await scheduler.tick()

        run_fn.assert_called_once()
        call_args = run_fn.call_args[0]
        assert call_args[0].name == "tick-test"

    @pytest.mark.asyncio
    async def test_tick_handles_errors(self, task_manager, run_store):
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
            run_store=run_store,
            run_task_fn=run_fn,
        )
        # Should not raise
        executed = await scheduler.tick()
        assert executed == 1
        # run_fn was called even though it raised
        run_fn.assert_called_once()


class TestTaskValidation:
    def test_warns_on_invalid_cron(self, task_manager, run_store, caplog):
        """Tasks with invalid cron expressions should log a warning."""
        task_manager.save_task(
            TaskDefinition(
                name="bad-cron",
                prompt="test",
                schedule="not-a-cron",
            )
        )
        scheduler = CronScheduler(
            task_manager=task_manager,
            run_store=run_store,
            run_task_fn=AsyncMock(),
        )
        import logging

        with caplog.at_level(logging.WARNING):
            scheduler.validate_tasks()

        assert any("invalid cron expression" in r.message.lower() for r in caplog.records)

    def test_warns_on_empty_prompt(self, task_manager, run_store, caplog):
        """Tasks with empty prompts should log a warning."""
        task_manager.save_task(
            TaskDefinition(
                name="empty-prompt",
                prompt="   ",
            )
        )
        scheduler = CronScheduler(
            task_manager=task_manager,
            run_store=run_store,
            run_task_fn=AsyncMock(),
        )
        import logging

        with caplog.at_level(logging.WARNING):
            scheduler.validate_tasks()

        assert any("empty prompt" in r.message.lower() for r in caplog.records)

    def test_no_warnings_for_valid_tasks(self, task_manager, run_store, caplog):
        """Valid tasks should not produce warnings."""
        task_manager.save_task(
            TaskDefinition(
                name="good-task",
                prompt="do something useful",
                schedule="0 9 * * *",
            )
        )
        scheduler = CronScheduler(
            task_manager=task_manager,
            run_store=run_store,
            run_task_fn=AsyncMock(),
        )
        import logging

        with caplog.at_level(logging.WARNING):
            scheduler.validate_tasks()

        assert len(caplog.records) == 0


class TestOrphanCleanup:
    def test_cleanup_orphan_runs(self, task_manager, run_store):
        """Runs for deleted tasks should be cleaned up."""
        # Create a task and record runs
        task_manager.save_task(TaskDefinition(name="existing", prompt="still here"))
        run_store.record_run(
            TaskRun(
                task_name="existing",
                execution_id="e1",
                status="completed",
                started_at="2026-01-01 09:00:00 UTC",
            )
        )
        # Record runs for a task that no longer has a YAML file
        run_store.record_run(
            TaskRun(
                task_name="deleted-task",
                execution_id="e2",
                status="completed",
                started_at="2026-01-01 09:00:00 UTC",
            )
        )

        scheduler = CronScheduler(
            task_manager=task_manager,
            run_store=run_store,
            run_task_fn=AsyncMock(),
        )
        deleted = scheduler.cleanup_orphan_runs()

        assert deleted == 1
        assert run_store.get_runs("deleted-task") == []
        assert len(run_store.get_runs("existing")) == 1

    def test_no_orphans_no_deletions(self, task_manager, run_store):
        """When all runs match existing tasks, nothing is deleted."""
        task_manager.save_task(TaskDefinition(name="t", prompt="p"))
        run_store.record_run(
            TaskRun(
                task_name="t",
                execution_id="e1",
                status="completed",
                started_at="2026-01-01 09:00:00 UTC",
            )
        )

        scheduler = CronScheduler(
            task_manager=task_manager,
            run_store=run_store,
            run_task_fn=AsyncMock(),
        )
        deleted = scheduler.cleanup_orphan_runs()
        assert deleted == 0
