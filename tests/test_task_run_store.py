"""Tests for TaskRunStore — SQLite-backed task run history."""

import pytest

from cliver.gateway.task_run_store import TaskRunStore
from cliver.task_manager import TaskRun


@pytest.fixture
def store(tmp_path):
    return TaskRunStore(tmp_path / "gateway.db")


class TestRecordAndQuery:
    def test_record_and_get_runs(self, store):
        run = TaskRun(
            task_name="t",
            execution_id="e1",
            status="completed",
            started_at="2026-01-01 09:00:00 UTC",
            finished_at="2026-01-01 09:05:00 UTC",
        )
        store.record_run(run)

        runs = store.get_runs("t")
        assert len(runs) == 1
        assert runs[0].execution_id == "e1"
        assert runs[0].status == "completed"

    def test_multiple_runs_ordered_recent_first(self, store):
        for i in range(5):
            store.record_run(
                TaskRun(
                    task_name="t",
                    execution_id=f"e{i}",
                    status="completed",
                    started_at=f"2026-01-0{i + 1} 09:00:00 UTC",
                )
            )

        runs = store.get_runs("t")
        assert len(runs) == 5
        assert runs[0].execution_id == "e4"
        assert runs[-1].execution_id == "e0"

    def test_limit(self, store):
        for i in range(20):
            store.record_run(
                TaskRun(
                    task_name="t",
                    execution_id=f"e{i}",
                    status="completed",
                    started_at=f"2026-01-{i + 1:02d} 09:00:00 UTC",
                )
            )

        runs = store.get_runs("t", limit=3)
        assert len(runs) == 3

    def test_empty_history(self, store):
        assert store.get_runs("nonexistent") == []

    def test_failed_run_recorded(self, store):
        store.record_run(
            TaskRun(
                task_name="t",
                execution_id="e1",
                status="failed",
                started_at="2026-01-01 09:00:00 UTC",
                error="connection timeout",
            )
        )

        runs = store.get_runs("t")
        assert runs[0].status == "failed"
        assert runs[0].error == "connection timeout"


class TestLastRunTime:
    def test_no_runs_returns_none(self, store):
        assert store.get_last_run_time("nonexistent") is None

    def test_returns_latest_run_time(self, store):
        store.record_run(
            TaskRun(
                task_name="t",
                execution_id="e1",
                status="completed",
                started_at="2026-01-01 09:00:00 UTC",
            )
        )
        store.record_run(
            TaskRun(
                task_name="t",
                execution_id="e2",
                status="completed",
                started_at="2026-01-02 10:00:00 UTC",
            )
        )

        ts = store.get_last_run_time("t")
        assert ts is not None
        # Should be the second run's timestamp
        from datetime import datetime, timezone

        expected = datetime(2026, 1, 2, 10, 0, 0, tzinfo=timezone.utc).timestamp()
        assert ts == expected

    def test_different_tasks_independent(self, store):
        store.record_run(
            TaskRun(
                task_name="a",
                execution_id="e1",
                status="completed",
                started_at="2026-01-01 09:00:00 UTC",
            )
        )
        store.record_run(
            TaskRun(
                task_name="b",
                execution_id="e2",
                status="completed",
                started_at="2026-01-02 10:00:00 UTC",
            )
        )

        ts_a = store.get_last_run_time("a")
        ts_b = store.get_last_run_time("b")
        assert ts_a != ts_b


class TestDeleteRuns:
    def test_delete_runs(self, store):
        store.record_run(
            TaskRun(
                task_name="t",
                execution_id="e1",
                status="completed",
                started_at="2026-01-01 09:00:00 UTC",
            )
        )
        deleted = store.delete_runs("t")
        assert deleted == 1
        assert store.get_runs("t") == []

    def test_delete_nonexistent(self, store):
        deleted = store.delete_runs("nope")
        assert deleted == 0


class TestGetAllTaskNames:
    def test_empty(self, store):
        assert store.get_all_task_names() == []

    def test_returns_distinct_names(self, store):
        for name in ["a", "b", "a", "c"]:
            store.record_run(
                TaskRun(
                    task_name=name,
                    execution_id="e1",
                    status="completed",
                    started_at="2026-01-01 09:00:00 UTC",
                )
            )
        names = sorted(store.get_all_task_names())
        assert names == ["a", "b", "c"]
