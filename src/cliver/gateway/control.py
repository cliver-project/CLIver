"""Unix socket control server for the gateway daemon."""

import asyncio
import json
import logging
import os
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class ControlServer:
    """Unix domain socket server for gateway status and lifecycle control.

    Protocol: line-delimited JSON. Client sends a request, server responds
    with one JSON line.

    Commands:
        {"cmd": "status"} -> {"status": "running", "uptime": ..., "tasks_run": ..., "platforms": [...]}
        {"cmd": "stop"}   -> {"status": "stopping"}
    """

    def __init__(self, socket_path: Path, pid_path: Path):
        self.socket_path = Path(socket_path)
        self.pid_path = Path(pid_path)
        self.shutdown_requested = False
        self.tasks_run = 0
        self.platforms: list[str] = []
        self._server: asyncio.AbstractServer | None = None
        self._start_time = 0.0

    async def start(self) -> None:
        """Start listening on the Unix socket."""
        # Clean up stale socket file
        if self.socket_path.exists():
            self.socket_path.unlink()

        self._start_time = time.monotonic()
        self._server = await asyncio.start_unix_server(self._handle_client, path=str(self.socket_path))

        # Write PID file
        self.pid_path.write_text(str(os.getpid()))
        logger.info(f"Control server listening on {self.socket_path}")

    async def stop(self) -> None:
        """Stop the control server and clean up."""
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

        if self.socket_path.exists():
            self.socket_path.unlink()
        if self.pid_path.exists():
            self.pid_path.unlink()

        logger.info("Control server stopped")

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Handle a single client connection."""
        try:
            while True:
                line = await reader.readline()
                if not line:
                    break
                try:
                    request = json.loads(line.decode().strip())
                except json.JSONDecodeError:
                    response = {"status": "error", "message": "invalid JSON"}
                    writer.write(json.dumps(response).encode() + b"\n")
                    await writer.drain()
                    continue

                response = self._dispatch(request)
                writer.write(json.dumps(response).encode() + b"\n")
                await writer.drain()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.debug(f"Control client error: {e}")
        finally:
            writer.close()
            await writer.wait_closed()

    def _dispatch(self, request: dict) -> dict:
        """Dispatch a control command and return the response."""
        cmd = request.get("cmd", "")

        if cmd == "status":
            uptime = int(time.monotonic() - self._start_time)
            return {
                "status": "running",
                "uptime": uptime,
                "tasks_run": self.tasks_run,
                "platforms": list(self.platforms),
            }
        elif cmd == "stop":
            self.shutdown_requested = True
            return {"status": "stopping"}
        else:
            return {"status": "error", "message": f"unknown command: {cmd}"}
