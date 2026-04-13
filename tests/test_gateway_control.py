"""Tests for the gateway control server (Unix socket)."""

import asyncio
import json
import tempfile
from pathlib import Path

import pytest

from cliver.gateway.control import ControlServer


@pytest.fixture
def socket_path():
    # Unix socket paths have a length limit (104 bytes on macOS).
    # pytest's tmp_path is too long, so use a short temp dir.
    with tempfile.TemporaryDirectory(prefix="gw") as d:
        yield Path(d) / "gw.sock"


@pytest.fixture
def pid_path(tmp_path):
    return tmp_path / "test-gateway.pid"


class TestControlServer:
    @pytest.mark.asyncio
    async def test_status_command(self, socket_path, pid_path):
        """Status command returns running state."""
        server = ControlServer(socket_path=socket_path, pid_path=pid_path)
        await server.start()
        try:
            reader, writer = await asyncio.open_unix_connection(str(socket_path))
            writer.write(json.dumps({"cmd": "status"}).encode() + b"\n")
            await writer.drain()
            data = await reader.readline()
            response = json.loads(data)
            assert response["status"] == "running"
            assert "uptime" in response
            assert "tasks_run" in response
            writer.close()
            await writer.wait_closed()
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_stop_command(self, socket_path, pid_path):
        """Stop command sets the shutdown flag."""
        server = ControlServer(socket_path=socket_path, pid_path=pid_path)
        await server.start()
        try:
            reader, writer = await asyncio.open_unix_connection(str(socket_path))
            writer.write(json.dumps({"cmd": "stop"}).encode() + b"\n")
            await writer.drain()
            data = await reader.readline()
            response = json.loads(data)
            assert response["status"] == "stopping"
            assert server.shutdown_requested
            writer.close()
            await writer.wait_closed()
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_pid_file_lifecycle(self, socket_path, pid_path):
        """PID file is created on start and removed on stop."""
        server = ControlServer(socket_path=socket_path, pid_path=pid_path)
        assert not pid_path.exists()
        await server.start()
        assert pid_path.exists()
        pid = int(pid_path.read_text().strip())
        assert pid > 0
        await server.stop()
        assert not pid_path.exists()

    @pytest.mark.asyncio
    async def test_unknown_command(self, socket_path, pid_path):
        """Unknown commands return an error response."""
        server = ControlServer(socket_path=socket_path, pid_path=pid_path)
        await server.start()
        try:
            reader, writer = await asyncio.open_unix_connection(str(socket_path))
            writer.write(json.dumps({"cmd": "unknown"}).encode() + b"\n")
            await writer.drain()
            data = await reader.readline()
            response = json.loads(data)
            assert response["status"] == "error"
            writer.close()
            await writer.wait_closed()
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_stale_socket_cleanup(self, socket_path, pid_path):
        """Starting a server cleans up a stale socket file."""
        socket_path.write_text("stale")
        server = ControlServer(socket_path=socket_path, pid_path=pid_path)
        await server.start()
        assert pid_path.exists()
        await server.stop()
