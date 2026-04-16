"""Tests for the Gateway orchestrator."""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cliver.gateway.gateway import Gateway


@pytest.fixture
def config_dir(tmp_path):
    d = tmp_path / "config"
    d.mkdir()
    return d


@pytest.fixture
def short_socket_dir():
    """Short temp dir for Unix socket (macOS path length limit)."""
    with tempfile.TemporaryDirectory(prefix="gw") as d:
        yield Path(d)


class TestGateway:
    @pytest.mark.asyncio
    async def test_start_and_stop(self, config_dir, short_socket_dir):
        """Gateway can start and stop cleanly."""
        gw = Gateway(config_dir=config_dir, agent_name="test-agent")
        # Override socket/pid paths to use short dir
        gw._control_server.socket_path = short_socket_dir / "gw.sock"
        gw._control_server.pid_path = config_dir / "gw.pid"

        with patch.object(gw, "_create_task_executor", return_value=MagicMock()):
            await gw.start()

            assert gw.is_running
            assert (config_dir / "gw.pid").exists()

            await gw.stop()

            assert not gw.is_running
            assert not (config_dir / "gw.pid").exists()

    @pytest.mark.asyncio
    async def test_run_with_shutdown(self, config_dir, short_socket_dir):
        """Gateway.run() exits when shutdown is requested via control server."""
        gw = Gateway(config_dir=config_dir, agent_name="test-agent")
        gw._control_server.socket_path = short_socket_dir / "gw.sock"
        gw._control_server.pid_path = config_dir / "gw.pid"

        with patch.object(gw, "_create_task_executor", return_value=MagicMock()):
            await gw.start()

            # Request shutdown after a short delay
            async def shutdown_after_delay():
                await asyncio.sleep(0.2)
                gw._control_server.shutdown_requested = True

            shutdown_task = asyncio.create_task(shutdown_after_delay())
            await gw.run(tick_interval=0.1)
            await shutdown_task
            await gw.stop()

            assert not gw.is_running
