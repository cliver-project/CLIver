"""Tests for TaskManager — task CRUD."""

import pytest

from cliver.task_manager import TaskDefinition, TaskManager, TaskOrigin


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

    def test_remove_cleans_up(self, manager):
        manager.save_task(TaskDefinition(name="t", prompt="p"))
        assert manager.remove_task("t") is True
        assert manager.get_task("t") is None


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
            session_key="slack:C12345:1234567890.123456",
        )
        t = TaskDefinition(name="t", prompt="research AI", origin=origin)
        assert t.origin.source == "slack"
        assert t.origin.channel_id == "C12345"
        assert t.origin.session_key == "slack:C12345:1234567890.123456"

    def test_origin_cli(self):
        origin = TaskOrigin(source="cli")
        t = TaskDefinition(name="t", prompt="do x", origin=origin)
        assert t.origin.source == "cli"
        assert t.origin.platform is None
        assert t.origin.channel_id is None

    def test_origin_round_trip_yaml(self, manager):
        origin = TaskOrigin(
            source="telegram",
            platform="telegram",
            channel_id="chat_123",
            thread_id="msg_456",
            user_id="user_789",
            session_key="telegram:chat_123:msg_456",
        )
        task = TaskDefinition(name="origin-task", prompt="do x", origin=origin)
        manager.save_task(task)

        loaded = manager.get_task("origin-task")
        assert loaded.origin is not None
        assert loaded.origin.source == "telegram"
        assert loaded.origin.platform == "telegram"
        assert loaded.origin.channel_id == "chat_123"
        assert loaded.origin.thread_id == "msg_456"
        assert loaded.origin.session_key == "telegram:chat_123:msg_456"

    def test_origin_omitted_in_yaml_when_none(self, manager):
        task = TaskDefinition(name="no-origin", prompt="do x")
        manager.save_task(task)

        loaded = manager.get_task("no-origin")
        assert loaded.origin is None
