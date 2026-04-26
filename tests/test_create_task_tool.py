"""Tests for the create_task builtin tool."""

import pytest

from cliver.gateway.task_store import TaskStore
from cliver.task_manager import TaskManager
from cliver.tools.create_task import CreateTaskTool


@pytest.fixture(autouse=True)
def _clear_profile():
    """Ensure no stale profile leaks from other tests."""
    from cliver.agent_profile import get_current_profile, set_current_profile

    saved = get_current_profile()
    set_current_profile(None)
    yield
    set_current_profile(saved)


@pytest.fixture
def tool(tmp_path):
    tasks_dir = tmp_path / "tasks"
    return CreateTaskTool(tasks_dir=tasks_dir)


@pytest.fixture
def store(tool):
    """Use the same DB path as the tool (tasks_dir.parent / gateway.db)."""
    return TaskStore(tool.tasks_dir.parent / "gateway.db")


class TestCreateTaskTool:
    def test_create_basic_task(self, tool):
        result = tool._run(name="my-task", prompt="do something")
        assert "my-task" in result
        assert "created" in result.lower()

    def test_create_task_with_schedule(self, tool):
        result = tool._run(name="cron-task", prompt="check status", schedule="0 9 * * *")
        assert "cron-task" in result

    def test_create_task_without_im_context(self, tool, store):
        tool._run(name="cli-task", prompt="do x")
        manager = TaskManager(tool.tasks_dir, store)
        loaded = manager.get_task("cli-task")
        assert loaded is not None
        assert loaded.origin is None

    def test_create_task_with_im_context(self, tool):
        """Origin is saved to DB (not YAML) when IM context is present."""
        from cliver.agent_profile import get_current_profile, set_current_profile
        from cliver.gateway.gateway import im_context
        from cliver.gateway.task_store import TaskStore

        # Ensure no stale profile leaks from other tests
        saved_profile = get_current_profile()
        set_current_profile(None)

        im_context.set(
            {
                "platform": "slack",
                "channel_id": "C12345",
                "thread_id": "ts123",
                "user_id": "U67890",
                "session_id": "sess-abc",
            }
        )
        try:
            result = tool._run(name="slack-task", prompt="research AI")
            assert "reply-back" in result

            db_path = tool.tasks_dir.parent / "gateway.db"
            store = TaskStore(db_path)

            # Origin is NOT in YAML (loaded via DB-first TaskManager)
            manager = TaskManager(tool.tasks_dir, store)
            loaded = manager.get_task("slack-task")
            assert loaded.origin is None

            # Origin in DB (platform/channel/thread/user — no session fields)
            origin = store.get_origin("slack-task")
            assert origin is not None
            assert origin.source == "slack"
            assert origin.platform == "slack"
            assert origin.channel_id == "C12345"
            assert origin.thread_id == "ts123"

            # Session ID on task row (not origin)
            assert store.get_session_id("slack-task") == "sess-abc"
        finally:
            im_context.set(None)
            set_current_profile(saved_profile)

    def test_duplicate_name_overwrites(self, tool, store):
        tool._run(name="t", prompt="old")
        tool._run(name="t", prompt="new")
        manager = TaskManager(tool.tasks_dir, store)
        loaded = manager.get_task("t")
        assert loaded.prompt == "new"
