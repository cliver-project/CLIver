"""Tests for TaskManager — DB-first task CRUD."""

import pytest

from cliver.gateway.task_store import TaskStore
from cliver.task_manager import TaskDefinition, TaskManager, TaskOrigin


@pytest.fixture
def store(tmp_path):
    return TaskStore(tmp_path / "gateway.db")


@pytest.fixture
def manager(tmp_path, store):
    return TaskManager(tmp_path / "tasks", store)


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

    def test_remove_cleans_up(self, manager):
        manager.save_task(TaskDefinition(name="t", prompt="p"))
        assert manager.remove_task("t") is True
        assert manager.get_task("t") is None


# ---------------------------------------------------------------------------
# DB-first behavior
# ---------------------------------------------------------------------------


class TestDBFirst:
    def test_yaml_only_not_recognised(self, manager):
        """A YAML file without a DB registration is not a valid task."""
        import yaml

        manager._ensure_dir()
        yaml_path = manager.tasks_dir / "stray.yaml"
        yaml_path.write_text(
            yaml.dump({"name": "stray", "prompt": "hello"}),
            encoding="utf-8",
        )
        assert manager.get_task("stray") is None
        assert all(t.name != "stray" for t in manager.list_tasks())

    def test_yaml_missing_shows_in_entries(self, manager, store):
        """A DB-registered task with missing YAML shows in entries."""
        store.register_task("ghost", "ghost.yaml")
        entries = manager.list_task_entries()
        assert len(entries) == 1
        assert entries[0].name == "ghost"
        assert entries[0].status == "yaml_missing"

    def test_yaml_invalid_shows_in_entries(self, manager, store):
        """A DB-registered task with invalid YAML shows in entries."""
        manager._ensure_dir()
        (manager.tasks_dir / "bad.yaml").write_text("not: valid: yaml: {{", encoding="utf-8")
        store.register_task("bad", "bad.yaml")
        entries = manager.list_task_entries()
        assert len(entries) == 1
        assert entries[0].name == "bad"
        assert entries[0].status == "yaml_invalid"

    def test_active_task_in_entries(self, manager):
        """A properly saved task shows as active in entries."""
        manager.save_task(TaskDefinition(name="ok", prompt="hello"))
        entries = manager.list_task_entries()
        assert len(entries) == 1
        assert entries[0].status == "active"
        assert entries[0].definition is not None
        assert entries[0].definition.prompt == "hello"

    def test_get_task_entry_missing_yaml(self, manager, store):
        store.register_task("missing", "missing.yaml")
        entry = manager.get_task_entry("missing")
        assert entry is not None
        assert entry.status == "yaml_missing"
        assert entry.definition is None

    def test_get_task_entry_active(self, manager):
        manager.save_task(TaskDefinition(name="active", prompt="test"))
        entry = manager.get_task_entry("active")
        assert entry is not None
        assert entry.status == "active"
        assert entry.definition.prompt == "test"

    def test_get_task_entry_nonexistent(self, manager):
        assert manager.get_task_entry("nope") is None

    def test_save_registers_in_db(self, manager, store):
        """save_task writes YAML and registers in DB."""
        manager.save_task(TaskDefinition(name="db-test", prompt="p"))
        row = store.get_registered_task("db-test")
        assert row is not None
        assert row["yaml_path"] == "db-test.yaml"

    def test_remove_unregisters_from_db(self, manager, store):
        """remove_task removes from DB and deletes YAML."""
        manager.save_task(TaskDefinition(name="rm-test", prompt="p"))
        assert manager.remove_task("rm-test") is True
        assert store.get_registered_task("rm-test") is None
        assert not (manager.tasks_dir / "rm-test.yaml").exists()

    def test_remove_missing_yaml_still_works(self, manager, store):
        """remove_task works even if YAML was already deleted."""
        store.register_task("no-yaml", "no-yaml.yaml")
        assert manager.remove_task("no-yaml") is True
        assert store.get_registered_task("no-yaml") is None


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


# ---------------------------------------------------------------------------
# TaskOrigin
# ---------------------------------------------------------------------------


class TestTaskOrigin:
    def test_task_without_origin(self):
        t = TaskDefinition(name="t", prompt="do something")
        assert t.origin is None

    def test_task_with_origin(self):
        origin = TaskOrigin(
            source="slack",
            platform="slack",
            channel_id="C12345",
            thread_id="1234567890.123456",
            user_id="U67890",
        )
        t = TaskDefinition(name="t", prompt="research AI", origin=origin)
        assert t.origin.source == "slack"
        assert t.origin.channel_id == "C12345"
        assert t.origin.thread_id == "1234567890.123456"

    def test_origin_cli(self):
        origin = TaskOrigin(source="cli")
        t = TaskDefinition(name="t", prompt="do x", origin=origin)
        assert t.origin.source == "cli"
        assert t.origin.platform is None
        assert t.origin.channel_id is None

    def test_origin_excluded_from_yaml(self, manager):
        """Origin is stored in DB (origin columns on tasks table), not YAML."""
        origin = TaskOrigin(
            source="telegram",
            platform="telegram",
            channel_id="chat_123",
            thread_id="msg_456",
            user_id="user_789",
        )
        task = TaskDefinition(name="origin-task", prompt="do x", origin=origin)
        manager.save_task(task)

        loaded = manager.get_task("origin-task")
        assert loaded.origin is None

    def test_origin_omitted_in_yaml_when_none(self, manager):
        task = TaskDefinition(name="no-origin", prompt="do x")
        manager.save_task(task)

        loaded = manager.get_task("no-origin")
        assert loaded.origin is None
