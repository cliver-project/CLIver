"""Gateway daemon management commands."""

import asyncio
import json
import logging
import os
import signal
import sys

import click

from cliver.cli import Cliver, pass_cliver
from cliver.util import get_config_dir

logger = logging.getLogger(__name__)


def _get_socket_path():
    return get_config_dir() / "cliver-gateway.sock"


def _get_pid_path():
    return get_config_dir() / "cliver-gateway.pid"


def _send_control_command(cmd: str) -> dict | None:
    """Send a command to a running gateway via its control socket.

    Returns the response dict, or None if the gateway is not reachable.
    """
    socket_path = _get_socket_path()
    if not socket_path.exists():
        return None

    try:

        async def _send():
            reader, writer = await asyncio.open_unix_connection(str(socket_path))
            writer.write(json.dumps({"cmd": cmd}).encode() + b"\n")
            await writer.drain()
            data = await reader.readline()
            writer.close()
            await writer.wait_closed()
            return json.loads(data)

        return asyncio.run(_send())
    except Exception:
        return None


@click.group(name="gateway", help="Manage the gateway daemon (cron scheduler & platform adapters)")
def gateway_cmd():
    """Gateway daemon management."""
    pass


@gateway_cmd.command(name="start", help="Start the gateway daemon")
@click.option("--daemon", "-d", is_flag=True, default=False, help="Run in the background (fork)")
@pass_cliver
def start(cliver: Cliver, daemon: bool):
    """Start the gateway process."""
    from cliver.gateway import Gateway

    # Check if already running
    pid_path = _get_pid_path()
    if pid_path.exists():
        try:
            pid = int(pid_path.read_text().strip())
            os.kill(pid, 0)  # check if process exists
            cliver.output(f"Gateway already running (PID {pid})")
            return 1
        except (ProcessLookupError, ValueError):
            # Stale PID file — clean up
            pid_path.unlink(missing_ok=True)

    if daemon:
        # Fork to background
        pid = os.fork()
        if pid > 0:
            cliver.output(f"Gateway started in background (PID {pid})")
            return 0
        # Child process continues below
        os.setsid()
        # Redirect stdout/stderr to log file
        log_path = get_config_dir() / "gateway.log"
        sys.stdout = open(log_path, "a")
        sys.stderr = sys.stdout

    config_dir = get_config_dir()
    agent_name = cliver.agent_name

    gw = Gateway(config_dir=config_dir, agent_name=agent_name)

    if not daemon:
        cliver.output(f"Gateway starting (agent: {agent_name}, Ctrl+C to stop)...")

    async def _run():
        await gw.start()
        # Handle signals for graceful shutdown
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, lambda: setattr(gw._control_server, "shutdown_requested", True))
        await gw.run()
        await gw.stop()

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass

    if not daemon:
        cliver.output("Gateway stopped.")
    return 0


@gateway_cmd.command(name="stop", help="Stop the running gateway daemon")
@pass_cliver
def stop(cliver: Cliver):
    """Stop the gateway process."""
    response = _send_control_command("stop")
    if response is None:
        cliver.output("No gateway running.")
        return 1
    cliver.output(f"Gateway stopping (status: {response.get('status', 'unknown')})")
    return 0


@gateway_cmd.command(name="status", help="Show gateway daemon status")
@pass_cliver
def status(cliver: Cliver):
    """Show gateway status."""
    response = _send_control_command("status")
    if response is None:
        cliver.output("Gateway is not running.")
        return 1

    uptime = response.get("uptime", 0)
    hours, remainder = divmod(uptime, 3600)
    minutes, seconds = divmod(remainder, 60)
    uptime_str = f"{hours}h {minutes}m {seconds}s" if hours else f"{minutes}m {seconds}s"

    cliver.output(f"  Status: {response.get('status', 'unknown')}")
    cliver.output(f"  Uptime: {uptime_str}")
    cliver.output(f"  Tasks run: {response.get('tasks_run', 0)}")
    platforms = response.get("platforms", [])
    cliver.output(f"  Platforms: {', '.join(platforms) if platforms else 'none'}")
    return 0
