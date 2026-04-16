"""Tests for TaskManager — task CRUD and run history."""

import pytest

from cliver.task_manager import TaskDefinition, TaskManager, TaskRun


@pytest.fixture
def manager(tmp_path):
    return TaskManager(tmp_path / "tasks")


# ---------------------------------------------------------------------------
# TaskDefinition model
# ---------------------------------------------------------------------------


class TestTaskDefinition:
    def test_minimal(self):
        t = TaskDefinition(name="t", prompt="do something")
        assert t.name == "t"
        assert t.prompt == "do something"
        assert t.model is None
        assert t.schedule is None

    def test_full(self):
        t = TaskDefinition(
            name="daily",
            description="Run daily",
            prompt="Research AI trends",
            model="deepseek-r1",
            schedule="0 9 * * *",
        )
        assert t.schedule == "0 9 * * *"
        assert t.model == "deepseek-r1"


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------


class TestCRUD:
    def test_save_and_get(self, manager):
        task = TaskDefinition(name="my-task", prompt="do x")
        manager.save_task(task)

        loaded = manager.get_task("my-task")
        assert loaded is not None
        assert loaded.name == "my-task"
        assert loaded.prompt == "do x"

    def test_list_empty(self, manager):
        assert manager.list_tasks() == []

    def test_list_tasks(self, manager):
        manager.save_task(TaskDefinition(name="a", prompt="p1"))
        manager.save_task(TaskDefinition(name="b", prompt="p2"))

        tasks = manager.list_tasks()
        names = [t.name for t in tasks]
        assert "a" in names
        assert "b" in names

    def test_get_nonexistent(self, manager):
        assert manager.get_task("nope") is None

    def test_overwrite(self, manager):
        manager.save_task(TaskDefinition(name="t", prompt="old"))
        manager.save_task(TaskDefinition(name="t", prompt="new"))

        loaded = manager.get_task("t")
        assert loaded.prompt == "new"

    def test_remove(self, manager):
        manager.save_task(TaskDefinition(name="t", prompt="p"))
        assert manager.remove_task("t") is True
        assert manager.get_task("t") is None

    def test_remove_nonexistent(self, manager):
        assert manager.remove_task("nope") is False

    def test_remove_also_removes_runs(self, manager):
        manager.save_task(TaskDefinition(name="t", prompt="p"))
        manager.record_run(
            TaskRun(
                task_name="t",
                execution_id="e1",
                status="completed",
                started_at="2026-01-01",
            )
        )
        assert len(manager.get_runs("t")) == 1

        manager.remove_task("t")
        assert manager.get_runs("t") == []

    def test_list_excludes_runs_files(self, manager):
        """Runs files (.runs.yaml) should not appear as tasks."""
        manager.save_task(TaskDefinition(name="t", prompt="p"))
        manager.record_run(
            TaskRun(
                task_name="t",
                execution_id="e1",
                status="completed",
                started_at="2026-01-01",
            )
        )

        tasks = manager.list_tasks()
        assert len(tasks) == 1
        assert tasks[0].name == "t"


# ---------------------------------------------------------------------------
# Run history
# ---------------------------------------------------------------------------


class TestRunHistory:
    def test_record_and_get(self, manager):
        run = TaskRun(
            task_name="t",
            execution_id="e1",
            status="completed",
            started_at="2026-01-01 09:00 UTC",
            finished_at="2026-01-01 09:05 UTC",
        )
        manager.record_run(run)

        runs = manager.get_runs("t")
        assert len(runs) == 1
        assert runs[0].execution_id == "e1"
        assert runs[0].status == "completed"

    def test_multiple_runs_ordered_recent_first(self, manager):
        for i in range(5):
            manager.record_run(
                TaskRun(
                    task_name="t",
                    execution_id=f"e{i}",
                    status="completed",
                    started_at=f"2026-01-0{i + 1}",
                )
            )

        runs = manager.get_runs("t")
        assert len(runs) == 5
        # Most recent first
        assert runs[0].execution_id == "e4"
        assert runs[-1].execution_id == "e0"

    def test_limit(self, manager):
        for i in range(20):
            manager.record_run(
                TaskRun(
                    task_name="t",
                    execution_id=f"e{i}",
                    status="completed",
                    started_at=f"run-{i}",
                )
            )

        runs = manager.get_runs("t", limit=3)
        assert len(runs) == 3

    def test_empty_history(self, manager):
        assert manager.get_runs("nonexistent") == []

    def test_failed_run_recorded(self, manager):
        manager.record_run(
            TaskRun(
                task_name="t",
                execution_id="e1",
                status="failed",
                started_at="2026-01-01",
                error="connection timeout",
            )
        )

        runs = manager.get_runs("t")
        assert runs[0].status == "failed"
        assert runs[0].error == "connection timeout"


# ---------------------------------------------------------------------------
# Task with schedule
# ---------------------------------------------------------------------------


class TestSchedule:
    def test_save_and_load_schedule(self, manager):
        task = TaskDefinition(
            name="cron-task",
            prompt="check status",
            schedule="30 8 * * 1-5",
        )
        manager.save_task(task)

        loaded = manager.get_task("cron-task")
        assert loaded.schedule == "30 8 * * 1-5"

    def test_no_schedule_is_none(self, manager):
        task = TaskDefinition(name="once", prompt="do once")
        manager.save_task(task)

        loaded = manager.get_task("once")
        assert loaded.schedule is None
