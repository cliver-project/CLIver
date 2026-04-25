"""Tests for the Gateway Starlette application."""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cliver.gateway.gateway import Gateway, ThreadQueue
from cliver.task_manager import TaskDefinition, TaskOrigin


@pytest.fixture
def config_dir(tmp_path):
    d = tmp_path / "config"
    d.mkdir()
    return d


class TestGateway:
    @pytest.mark.asyncio
    async def test_create_app_returns_application(self, config_dir):
        gw = Gateway(config_dir=config_dir, agent_name="test")
        with patch.object(gw, "_create_agent_core", return_value=MagicMock()):
            with patch.object(gw, "_get_config_manager") as mock_cm:
                mock_cm.return_value.config.gateway = None
                app = gw.create_app()
                from starlette.applications import Starlette

                assert isinstance(app, Starlette)

    @pytest.mark.asyncio
    async def test_startup_acquires_flock(self, config_dir):
        gw = Gateway(config_dir=config_dir, agent_name="test")
        with patch.object(gw, "_create_agent_core", return_value=MagicMock()):
            with patch.object(gw, "_load_adapters", return_value=[]):
                await gw._on_startup()
                assert gw._pid_path.exists()
                pid = int(gw._pid_path.read_text().strip())
                assert pid > 0
                await gw._on_cleanup()
                assert not gw._pid_path.exists()

    @pytest.mark.asyncio
    async def test_get_status(self, config_dir):
        gw = Gateway(config_dir=config_dir, agent_name="test")
        gw._start_time = 100.0
        gw._tasks_run = 5
        gw._adapter_manager = MagicMock()
        gw._adapter_manager.connected_platforms = ["slack"]
        with patch("time.monotonic", return_value=160.0):
            status = gw._get_status()
        assert status["uptime"] == 60
        assert status["tasks_run"] == 5
        assert status["platforms"] == ["slack"]

    @pytest.mark.asyncio
    async def test_cleanup_releases_flock(self, config_dir):
        gw = Gateway(config_dir=config_dir, agent_name="test")
        with patch.object(gw, "_create_agent_core", return_value=MagicMock()):
            with patch.object(gw, "_load_adapters", return_value=[]):
                await gw._on_startup()
                assert gw._pid_file is not None
                await gw._on_cleanup()
                assert gw._pid_file is None
                assert not gw._pid_path.exists()


class TestThreadQueue:
    @pytest.mark.asyncio
    async def test_same_key_serializes(self):
        queue = ThreadQueue()
        order = []

        async def work(key, label, delay):
            async with queue.get_lock(key):
                order.append(f"{label}-start")
                await asyncio.sleep(delay)
                order.append(f"{label}-end")

        await asyncio.gather(
            work("thread-1", "A", 0.05),
            work("thread-1", "B", 0.01),
        )
        assert order == ["A-start", "A-end", "B-start", "B-end"]

    @pytest.mark.asyncio
    async def test_different_keys_parallel(self):
        queue = ThreadQueue()
        order = []

        async def work(key, label, delay):
            async with queue.get_lock(key):
                order.append(f"{label}-start")
                await asyncio.sleep(delay)
                order.append(f"{label}-end")

        await asyncio.gather(
            work("thread-1", "A", 0.05),
            work("thread-2", "B", 0.01),
        )
        assert "B-end" in order
        b_end_idx = order.index("B-end")
        a_end_idx = order.index("A-end")
        assert b_end_idx < a_end_idx

    @pytest.mark.asyncio
    async def test_cleanup_removes_stale_locks(self):
        queue = ThreadQueue()
        async with queue.get_lock("old-thread"):
            pass
        queue._last_used["old-thread"] = time.monotonic() - 7200
        queue.cleanup(max_idle_seconds=3600)
        assert "old-thread" not in queue._locks

    @pytest.mark.asyncio
    async def test_cleanup_keeps_recent_locks(self):
        queue = ThreadQueue()
        async with queue.get_lock("recent"):
            pass
        queue.cleanup(max_idle_seconds=3600)
        assert "recent" in queue._locks


class TestOriginAwareExecution:
    @pytest.mark.asyncio
    async def test_run_task_without_origin(self, config_dir):
        """CLI-originated task runs statelessly, no reply-back."""
        gw = Gateway(config_dir=config_dir, agent_name="test")
        gw._agent_core = MagicMock()
        gw._agent_core.process_user_input = AsyncMock(return_value=MagicMock(content="done"))
        gw._run_store = MagicMock()
        gw._run_store.set_task_state = MagicMock()

        task = TaskDefinition(name="cli-task", prompt="do x")
        await gw._run_task(task)

        gw._agent_core.process_user_input.assert_awaited_once()
        gw._run_store.record_run.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_task_with_im_origin_adapter_connected(self, config_dir):
        """IM-originated task delivers result back to thread."""
        gw = Gateway(config_dir=config_dir, agent_name="test")
        gw._agent_core = MagicMock()
        gw._agent_core.process_user_input = AsyncMock(return_value=MagicMock(content="AI trends summary"))
        gw._run_store = MagicMock()
        gw._run_store.set_task_state = MagicMock()

        mock_adapter = AsyncMock()
        mock_adapter.name = "slack"
        mock_adapter.format_message = MagicMock(return_value="AI trends summary")
        mock_adapter.max_message_length = MagicMock(return_value=4000)
        gw._adapter_manager = MagicMock()
        gw._adapter_manager.connected_platforms = ["slack"]
        gw._adapter_manager._adapters = [mock_adapter]

        gw._session_manager = MagicMock()
        gw._session_manager.load_turns = MagicMock(return_value=[])
        gw._session_manager.list_sessions = MagicMock(return_value=[])
        gw._session_manager.append_turn = MagicMock()
        gw._session_manager.create_session = MagicMock(return_value="sess-123")

        origin = TaskOrigin(
            source="slack",
            platform="slack",
            channel_id="C123",
            thread_id="ts456",
            user_id="U789",
            session_key="slack:C123:ts456",
        )
        task = TaskDefinition(name="im-task", prompt="research AI", origin=origin)
        await gw._run_task(task)

        mock_adapter.send_text.assert_awaited()
        call_args = mock_adapter.send_text.call_args
        assert call_args[0][0] == "C123"

    @pytest.mark.asyncio
    async def test_run_task_suspended_when_adapter_disconnected(self, config_dir):
        """IM-originated task gets suspended if adapter is not connected."""
        gw = Gateway(config_dir=config_dir, agent_name="test")
        gw._agent_core = MagicMock()
        gw._agent_core.process_user_input = AsyncMock()
        gw._run_store = MagicMock()
        gw._run_store.set_task_state = MagicMock()
        gw._adapter_manager = MagicMock()
        gw._adapter_manager.connected_platforms = []

        origin = TaskOrigin(
            source="slack",
            platform="slack",
            channel_id="C123",
            thread_id="ts456",
        )
        task = TaskDefinition(name="suspended-task", prompt="do x", origin=origin)
        await gw._run_task(task)

        gw._run_store.set_task_state.assert_called_with(
            "suspended-task", "suspended", reason="Adapter 'slack' not connected"
        )
        gw._agent_core.process_user_input.assert_not_awaited()
