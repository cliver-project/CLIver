"""Tests for the create_task builtin tool."""

import pytest

from cliver.task_manager import TaskManager
from cliver.tools.create_task import CreateTaskTool


@pytest.fixture
def tool(tmp_path):
    tasks_dir = tmp_path / "tasks"
    return CreateTaskTool(tasks_dir=tasks_dir)


class TestCreateTaskTool:
    def test_create_basic_task(self, tool):
        result = tool._run(name="my-task", prompt="do something")
        assert "my-task" in result
        assert "created" in result.lower()

    def test_create_task_with_schedule(self, tool):
        result = tool._run(name="cron-task", prompt="check status", schedule="0 9 * * *")
        assert "cron-task" in result

    def test_create_task_without_im_context(self, tool):
        tool._run(name="cli-task", prompt="do x")
        manager = TaskManager(tool.tasks_dir)
        loaded = manager.get_task("cli-task")
        assert loaded.origin is None

    def test_create_task_with_im_context(self, tool):
        from cliver.gateway.gateway import im_context

        im_context.set(
            {
                "platform": "slack",
                "channel_id": "C12345",
                "thread_id": "ts123",
                "user_id": "U67890",
                "session_key": "slack:C12345:ts123",
            }
        )
        try:
            tool._run(name="slack-task", prompt="research AI")
            manager = TaskManager(tool.tasks_dir)
            loaded = manager.get_task("slack-task")
            assert loaded.origin is not None
            assert loaded.origin.source == "slack"
            assert loaded.origin.platform == "slack"
            assert loaded.origin.channel_id == "C12345"
            assert loaded.origin.thread_id == "ts123"
            assert loaded.origin.session_key == "slack:C12345:ts123"
        finally:
            im_context.set(None)

    def test_duplicate_name_overwrites(self, tool):
        tool._run(name="t", prompt="old")
        tool._run(name="t", prompt="new")
        manager = TaskManager(tool.tasks_dir)
        loaded = manager.get_task("t")
        assert loaded.prompt == "new"
