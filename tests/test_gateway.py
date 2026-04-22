"""Tests for the Gateway aiohttp application."""

import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from aiohttp import web

from cliver.gateway.gateway import Gateway


@pytest.fixture
def config_dir(tmp_path):
    d = tmp_path / "config"
    d.mkdir()
    return d


class TestGateway:
    @pytest.mark.asyncio
    async def test_create_app_returns_application(self, config_dir):
        gw = Gateway(config_dir=config_dir, agent_name="test")
        app = gw.create_app()
        assert isinstance(app, web.Application)

    @pytest.mark.asyncio
    async def test_startup_acquires_flock(self, config_dir):
        gw = Gateway(config_dir=config_dir, agent_name="test")
        app = web.Application()
        with patch.object(gw, "_create_task_executor", return_value=MagicMock()):
            with patch.object(gw, "_load_adapters", return_value=[]):
                await gw._on_startup(app)
                assert gw._pid_path.exists()
                pid = int(gw._pid_path.read_text().strip())
                assert pid > 0
                await gw._on_cleanup(app)
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
        app = web.Application()
        with patch.object(gw, "_create_task_executor", return_value=MagicMock()):
            with patch.object(gw, "_load_adapters", return_value=[]):
                await gw._on_startup(app)
                assert gw._pid_file is not None
                await gw._on_cleanup(app)
                assert gw._pid_file is None
                assert not gw._pid_path.exists()
