"""Tests for TaskStore — SQLite-backed task registry, run history, and state."""

import pytest

from cliver.gateway.task_store import TaskStore
from cliver.task_manager import TaskRun


@pytest.fixture
def store(tmp_path):
    return TaskStore(tmp_path / "gateway.db")


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------


class TestTaskRegistry:
    def test_register_and_get(self, store):
        store.register_task("my-task", "my-task.yaml")
        row = store.get_registered_task("my-task")
        assert row is not None
        assert row["name"] == "my-task"
        assert row["yaml_path"] == "my-task.yaml"
        assert row["created_at"] is not None
        assert row["updated_at"] is not None

    def test_get_nonexistent(self, store):
        assert store.get_registered_task("nope") is None

    def test_list_registered_tasks(self, store):
        store.register_task("a", "a.yaml")
        store.register_task("b", "b.yaml")
        tasks = store.list_registered_tasks()
        names = [t["name"] for t in tasks]
        assert names == ["a", "b"]

    def test_list_empty(self, store):
        assert store.list_registered_tasks() == []

    def test_upsert_updates_path(self, store):
        store.register_task("t", "old.yaml")
        store.register_task("t", "new.yaml")
        row = store.get_registered_task("t")
        assert row["yaml_path"] == "new.yaml"

    def test_unregister(self, store):
        store.register_task("t", "t.yaml")
        assert store.unregister_task("t") is True
        assert store.get_registered_task("t") is None

    def test_unregister_nonexistent(self, store):
        assert store.unregister_task("nope") is False


# ---------------------------------------------------------------------------
# Session linkage
# ---------------------------------------------------------------------------


class TestSessionLinkage:
    def test_set_and_get_session_id(self, store):
        store.register_task("t", "t.yaml")
        store.set_session_id("t", "sess-abc")
        assert store.get_session_id("t") == "sess-abc"

    def test_session_id_in_registered_task(self, store):
        store.register_task("t", "t.yaml")
        store.set_session_id("t", "sess-123")
        row = store.get_registered_task("t")
        assert row["session_id"] == "sess-123"

    def test_no_session_id_returns_none(self, store):
        store.register_task("t", "t.yaml")
        assert store.get_session_id("t") is None

    def test_nonexistent_task_returns_none(self, store):
        assert store.get_session_id("nope") is None

    def test_session_id_deleted_with_task(self, store):
        store.register_task("t", "t.yaml")
        store.set_session_id("t", "sess-abc")
        store.unregister_task("t")
        assert store.get_session_id("t") is None


# ---------------------------------------------------------------------------
# Run history
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Task state
# ---------------------------------------------------------------------------


class TestTaskState:
    def test_set_and_get_state(self, store):
        store.register_task("my-task", "my-task.yaml")
        store.set_task_state("my-task", "pending")
        state = store.get_task_state("my-task")
        assert state["task_name"] == "my-task"
        assert state["status"] == "pending"
        assert state["suspend_reason"] is None

    def test_update_state(self, store):
        store.register_task("t", "t.yaml")
        store.set_task_state("t", "pending")
        store.set_task_state("t", "running")
        state = store.get_task_state("t")
        assert state["status"] == "running"

    def test_suspend_with_reason(self, store):
        store.register_task("t", "t.yaml")
        store.set_task_state("t", "suspended", reason="Adapter 'slack' not connected")
        state = store.get_task_state("t")
        assert state["status"] == "suspended"
        assert state["suspend_reason"] == "Adapter 'slack' not connected"

    def test_get_nonexistent_state(self, store):
        assert store.get_task_state("nope") is None

    def test_get_tasks_by_status(self, store):
        store.register_task("a", "a.yaml")
        store.register_task("b", "b.yaml")
        store.register_task("c", "c.yaml")
        store.set_task_state("a", "suspended", reason="slack down")
        store.set_task_state("b", "running")
        store.set_task_state("c", "suspended", reason="telegram down")

        suspended = store.get_tasks_by_status("suspended")
        names = [s["task_name"] for s in suspended]
        assert sorted(names) == ["a", "c"]

    def test_delete_task_state(self, store):
        store.register_task("t", "t.yaml")
        store.set_task_state("t", "pending")
        store.delete_task_state("t")
        assert store.get_task_state("t") is None

    def test_state_deleted_with_task(self, store):
        """Unregistering a task also removes its state."""
        store.register_task("t", "t.yaml")
        store.set_task_state("t", "running")
        assert store.get_task_state("t") is not None

        store.unregister_task("t")
        assert store.get_task_state("t") is None


# ---------------------------------------------------------------------------
# Task origin
# ---------------------------------------------------------------------------


class TestTaskOrigin:
    def test_save_and_get_origin(self, store):
        from cliver.task_manager import TaskOrigin

        store.register_task("my-task", "my-task.yaml")
        origin = TaskOrigin(
            source="slack",
            platform="slack",
            channel_id="C12345",
            thread_id="ts123",
            user_id="U67890",
        )
        store.save_origin("my-task", origin)
        loaded = store.get_origin("my-task")
        assert loaded is not None
        assert loaded.platform == "slack"
        assert loaded.channel_id == "C12345"
        assert loaded.user_id == "U67890"

    def test_origin_minimal(self, store):
        from cliver.task_manager import TaskOrigin

        store.register_task("cli-task", "cli-task.yaml")
        origin = TaskOrigin(source="cli")
        store.save_origin("cli-task", origin)
        loaded = store.get_origin("cli-task")
        assert loaded is not None
        assert loaded.source == "cli"
        assert loaded.platform is None

    def test_update_origin(self, store):
        from cliver.task_manager import TaskOrigin

        store.register_task("t", "t.yaml")
        origin = TaskOrigin(source="slack", platform="slack")
        store.save_origin("t", origin)

        origin2 = TaskOrigin(source="telegram", platform="telegram", channel_id="chat_1")
        store.save_origin("t", origin2)

        loaded = store.get_origin("t")
        assert loaded.source == "telegram"
        assert loaded.channel_id == "chat_1"

    def test_delete_origin(self, store):
        from cliver.task_manager import TaskOrigin

        store.register_task("t", "t.yaml")
        origin = TaskOrigin(source="cli")
        store.save_origin("t", origin)
        assert store.get_origin("t") is not None

        store.delete_origin("t")
        assert store.get_origin("t") is None

    def test_get_nonexistent_origin(self, store):
        assert store.get_origin("nope") is None

    def test_origin_deleted_with_task(self, store):
        """Unregistering a task also removes its origin."""
        from cliver.task_manager import TaskOrigin

        store.register_task("t", "t.yaml")
        store.save_origin("t", TaskOrigin(source="slack", platform="slack"))
        assert store.get_origin("t") is not None

        store.unregister_task("t")
        assert store.get_origin("t") is None
