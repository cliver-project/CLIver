"""Gateway daemon management commands."""

import json
import logging
import os
import signal
import subprocess
import sys
import time

import click

from cliver.cli import Cliver, pass_cliver
from cliver.util import get_config_dir

logger = logging.getLogger(__name__)


def _get_pid_path():
    return get_config_dir() / "cliver-gateway.pid"


# ── Logic Functions ──


def _start_gateway(cliver: Cliver):
    """Start the gateway daemon as a subprocess.

    Spawns a fresh Python process (no os.fork) to avoid macOS segfaults
    in socket.getaddrinfo and Security framework caused by forking a
    process with active threads or mach port connections.
    """
    pid_path = _get_pid_path()

    # Check if already running via PID file
    if pid_path.exists():
        try:
            pid = int(pid_path.read_text().strip())
            os.kill(pid, 0)
            cliver.output(f"Gateway already running (PID {pid})")
            return 1
        except (ProcessLookupError, ValueError):
            pid_path.unlink(missing_ok=True)

    # Read host/port from config
    cfg = cliver.config_manager.config
    gw_cfg = cfg.gateway
    host = gw_cfg.host if gw_cfg else "127.0.0.1"
    port = gw_cfg.port if gw_cfg else 8321

    # Check if port is already in use (catches orphan processes without PID file)
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex((host, port)) == 0:
            cliver.output(f"[red]Port {port} is already in use. An orphan gateway may be running.[/red]")
            cliver.output(f"[dim]Find it with: lsof -i :{port}[/dim]")
            return 1

    cliver.output("[dim]Starting gateway...[/dim]")

    # Spawn gateway as a fresh subprocess — avoids all fork-related issues
    # (macOS mach port invalidation, DNS resolver segfaults, thread state).
    # The child process handles its own secret resolution and AgentCore init.
    log_path = get_config_dir() / "gateway.log"
    proc = subprocess.Popen(
        [sys.executable, "-m", "cliver.gateway.main", "--agent", cliver.agent_name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

    # Wait for the gateway to become healthy (PID file + HTTP /health)
    for _ in range(20):
        time.sleep(0.5)
        # Check if process died
        if proc.poll() is not None:
            cliver.output(f"[red]Gateway process exited with code {proc.returncode}[/red]")
            cliver.output(f"[dim]Check log: {log_path}[/dim]")
            return 1
        # Check for health endpoint
        if pid_path.exists():
            try:
                import urllib.request

                url = f"http://{host}:{port}/health"
                with urllib.request.urlopen(url, timeout=2) as resp:
                    if resp.status == 200:
                        cliver.output(f"Gateway started (PID {proc.pid})")
                        cliver.output(f"  http://{host}:{port}")
                        cliver.output(f"  Log: {log_path}")
                        return 0
            except Exception:
                pass

    # Timed out waiting for health
    if proc.poll() is None:
        cliver.output(f"Gateway process alive (PID {proc.pid}) but not yet healthy")
        cliver.output(f"  Check log: {log_path}")
    else:
        cliver.output(f"[red]Gateway process exited with code {proc.returncode}[/red]")
        cliver.output(f"[dim]Check log: {log_path}[/dim]")
    return 1


def _stop_gateway(cliver: Cliver):
    """Stop the gateway daemon via SIGTERM."""
    pid_path = _get_pid_path()
    if not pid_path.exists():
        cliver.output("No gateway running.")
        return 1

    try:
        pid = int(pid_path.read_text().strip())
        os.kill(pid, 0)
    except (ProcessLookupError, ValueError):
        pid_path.unlink(missing_ok=True)
        cliver.output("No gateway running (stale PID file cleaned up).")
        return 1

    os.kill(pid, signal.SIGTERM)

    for _ in range(10):
        time.sleep(0.5)
        if not pid_path.exists():
            cliver.output("Gateway stopped.")
            return 0

    cliver.output(f"SIGTERM sent to PID {pid}. Still shutting down.")
    return 0


def _restart_gateway(cliver: Cliver):
    """Restart the gateway daemon."""
    pid_path = _get_pid_path()
    if pid_path.exists():
        try:
            pid = int(pid_path.read_text().strip())
            os.kill(pid, 0)
            cliver.output(f"Stopping gateway (PID {pid})...")
            os.kill(pid, signal.SIGTERM)
            for _ in range(10):
                time.sleep(0.5)
                if not pid_path.exists():
                    break
        except (ProcessLookupError, ValueError):
            pid_path.unlink(missing_ok=True)

    _start_gateway(cliver)


def _list_platforms(cliver: Cliver):
    """List configured platform adapters with live connection status."""
    from rich import box
    from rich.table import Table

    cfg = cliver.config_manager.config
    gw = cfg.gateway
    if not gw or not gw.platforms:
        cliver.output("[dim]No platforms configured.[/dim]")
        cliver.output("[dim]Set one up with: /gateway platform setup[/dim]")
        return

    # Fetch live status from running gateway if available
    live_statuses = {}
    gw_cfg = cfg.gateway
    host = gw_cfg.host if gw_cfg else "127.0.0.1"
    port = gw_cfg.port if gw_cfg else 8321
    pid_path = _get_pid_path()
    if pid_path.exists():
        try:
            import urllib.request

            url = f"http://{host}:{port}/health"
            with urllib.request.urlopen(url, timeout=2) as resp:
                data = json.loads(resp.read())
            for a in data.get("adapters", []):
                live_statuses[a["name"]] = a
        except Exception:
            pass

    table = Table(title="Platform Adapters", box=box.SQUARE)
    table.add_column("Name", style="green")
    table.add_column("Type", style="cyan")
    table.add_column("Status")
    table.add_column("Token", style="dim")
    table.add_column("Home Channel", style="yellow")

    for name, p in gw.platforms.items():
        token_display = _mask_token(p.token) if p.token else "-"
        live = live_statuses.get(name)
        if live:
            state = live.get("state", "")
            err = live.get("error", "")
            if state == "connected":
                status_str = "[green]● connected[/green]"
            elif state == "connecting":
                status_str = "[yellow]◌ connecting[/yellow]"
            elif state == "error":
                short_err = (err[:30] + "...") if len(err) > 30 else err
                status_str = f"[red]✗ {short_err}[/red]"
            else:
                status_str = f"[dim]{state}[/dim]"
        elif pid_path.exists():
            status_str = "[dim]not loaded[/dim]"
        else:
            status_str = "[dim]gateway stopped[/dim]"
        table.add_row(name, p.type, status_str, token_display, p.home_channel or "-")

    cliver.output(table)


def _add_platform(
    cliver: Cliver,
    name: str,
    ptype: str,
    token: str = None,
    app_token: str = None,
    home_channel: str = None,
    allowed_users: str = None,
):
    """Add a new platform adapter configuration."""
    from cliver.config import PlatformConfig

    cfg = cliver.config_manager.config
    if cfg.gateway is None:
        from cliver.config import GatewayConfig

        cfg.gateway = GatewayConfig()

    if name in cfg.gateway.platforms:
        cliver.output(f"[red]Platform '{name}' already exists. Use 'setup' to reconfigure.[/red]")
        return

    users = [u.strip() for u in allowed_users.split(",")] if allowed_users else None
    platform = PlatformConfig(
        name=name,
        type=ptype,
        token=token,
        app_token=app_token,
        home_channel=home_channel,
        allowed_users=users,
    )
    cfg.gateway.platforms[name] = platform
    cliver.config_manager._save_config()
    cliver.output(f"[green]✓[/green] Platform '{name}' added (type: {ptype}).")
    cliver.output("[dim]Restart gateway to apply: /gateway stop && /gateway start[/dim]")


# -- Platform type metadata for the setup wizard --

_PLATFORM_INFO = {
    "slack": {
        "label": "Slack",
        "description": "Connect to a Slack workspace via Socket Mode",
        "fields": [
            (
                "token",
                "Bot User OAuth Token",
                "Find it in Slack App → OAuth & Permissions → Bot User OAuth Token.\n"
                "   [dim]Starts with xoxb-...[/dim]",
            ),
            (
                "app_token",
                "App-Level Token",
                "Find it in Slack App → Basic Information → App-Level Tokens.\n"
                "   [dim]Starts with xapp-... Required for Socket Mode.[/dim]",
            ),
        ],
    },
    "telegram": {
        "label": "Telegram",
        "description": "Connect to Telegram via Bot API",
        "fields": [
            (
                "token",
                "Bot Token",
                "Get it from @BotFather on Telegram → /newbot.\n   [dim]Format: 123456789:ABCdef...[/dim]",
            ),
        ],
    },
    "discord": {
        "label": "Discord",
        "description": "Connect to Discord as a bot",
        "fields": [
            (
                "token",
                "Bot Token",
                "Find it in Discord Developer Portal → Bot → Token.\n"
                "   [dim]Enable Message Content Intent in the Bot settings.[/dim]",
            ),
        ],
    },
    "feishu": {
        "label": "Feishu (飞书/Lark)",
        "description": "Connect to Feishu / Lark workspace",
        "fields": [
            ("token", "App Secret", "Find it in Feishu Open Platform → your app → Credentials."),
            ("app_id", "App ID", "Find it in Feishu Open Platform → your app → Credentials."),
        ],
    },
}


def _setup_platform(cliver: Cliver):
    """Interactive platform setup wizard."""
    from cliver.config import GatewayConfig, PlatformConfig
    from cliver.gateway.adapters import BUILTIN_ADAPTERS

    console = cliver.console
    console.print("\n[bold]Platform Setup[/bold]")
    console.print("[dim]─────────────────[/dim]")

    # 1. Choose platform type
    types = list(BUILTIN_ADAPTERS.keys())
    console.print("\n[bold]1. Platform type[/bold]")
    for i, t in enumerate(types, 1):
        info = _PLATFORM_INFO.get(t, {})
        label = info.get("label", t)
        desc = info.get("description", "")
        console.print(f"   [green]{i}[/green]) {label}  [dim]{desc}[/dim]")

    choice = cliver.ui.ask_input("   Choose [1-5] > ")
    if not choice:
        console.print("[yellow]Cancelled.[/yellow]")
        return
    try:
        ptype = types[int(choice) - 1]
    except (ValueError, IndexError):
        console.print(f"[red]Invalid choice: {choice}[/red]")
        return

    info = _PLATFORM_INFO.get(ptype, {})
    label = info.get("label", ptype)

    # 2. Name (default to type)
    console.print("\n[bold]2. Adapter name[/bold] [dim](used as identifier)[/dim]")
    name = cliver.ui.ask_input(f"   Name [{ptype}] > ") or ptype

    # Check for existing — offer to reconfigure
    cfg = cliver.config_manager.config
    if cfg.gateway and name in cfg.gateway.platforms:
        response = cliver.ui.ask_input(
            f"   Platform '{name}' already exists. Reconfigure? [y/n] > ",
            choices=["y", "yes", "n", "no"],
        )
        if response.lower() not in ("y", "yes"):
            console.print("[yellow]Cancelled.[/yellow]")
            return

    # 3. Type-specific required fields
    collected = {"type": ptype}
    fields = info.get("fields", [])
    for i, (field_key, field_label, field_help) in enumerate(fields, 3):
        console.print(f"\n[bold]{i}. {field_label}[/bold]")
        console.print(f"   {field_help}")

        # Show existing value if reconfiguring
        existing = ""
        if cfg.gateway and name in cfg.gateway.platforms:
            existing = getattr(cfg.gateway.platforms[name], field_key, "") or ""

        if existing:
            is_secret = "token" in field_key.lower() or "secret" in field_key.lower()
            masked = _mask_token(existing) if is_secret else existing
            prompt = f"   [{masked}] > "
        else:
            prompt = "   > "

        value = cliver.ui.ask_input(prompt)
        if not value and existing:
            value = existing
        if not value and field_key in ("token",):
            console.print(f"[red]{field_label} is required.[/red]")
            return
        if value:
            collected[field_key] = value

    # 4. Optional: home channel
    next_num = 3 + len(fields)
    console.print(f"\n[bold]{next_num}. Home channel[/bold] [dim](optional — default channel for cron output)[/dim]")
    existing_channel = ""
    if cfg.gateway and name in cfg.gateway.platforms:
        existing_channel = cfg.gateway.platforms[name].home_channel or ""
    if existing_channel:
        home_channel = cliver.ui.ask_input(f"   [{existing_channel}] > ") or existing_channel
    else:
        home_channel = cliver.ui.ask_input("   Channel ID (Enter to skip) > ")

    # 5. Optional: allowed users
    next_num += 1
    console.print(
        f"\n[bold]{next_num}. Allowed users[/bold] [dim](optional — comma-separated user IDs, empty = all)[/dim]"
    )
    existing_users = ""
    if cfg.gateway and name in cfg.gateway.platforms:
        existing_users = ",".join(cfg.gateway.platforms[name].allowed_users or [])
    if existing_users:
        allowed_users = cliver.ui.ask_input(f"   [{existing_users}] > ") or existing_users
    else:
        allowed_users = cliver.ui.ask_input("   User IDs (Enter to skip) > ")

    # Build the config
    users_list = [u.strip() for u in allowed_users.split(",") if u.strip()] if allowed_users else None

    # Extract standard fields vs extra fields
    token = collected.pop("token", None)
    ptype = collected.pop("type")
    app_token = collected.pop("app_token", None)

    platform = PlatformConfig(
        name=name,
        type=ptype,
        token=token,
        app_token=app_token,
        home_channel=home_channel or None,
        allowed_users=users_list,
        **collected,
    )

    # Save
    if cfg.gateway is None:
        cfg.gateway = GatewayConfig()
    cfg.gateway.platforms[name] = platform
    cliver.config_manager._save_config()

    console.print(f"\n[green]✓[/green] Platform '{name}' ({label}) configured.")
    cliver.output("[dim]Restart gateway to apply: /gateway stop && /gateway start[/dim]")


def _set_platform(cliver: Cliver, name: str, **kwargs):
    """Update an existing platform adapter configuration."""
    cfg = cliver.config_manager.config
    if not cfg.gateway or name not in cfg.gateway.platforms:
        cliver.output(f"[red]Platform '{name}' not found.[/red]")
        return

    p = cfg.gateway.platforms[name]
    for key, value in kwargs.items():
        if value is not None:
            if key == "allowed_users":
                value = [u.strip() for u in value.split(",")]
            setattr(p, key, value)

    cliver.config_manager._save_config()
    cliver.output(f"[green]✓[/green] Platform '{name}' updated.")
    cliver.output("[dim]Restart gateway to apply.[/dim]")


def _remove_platform(cliver: Cliver, name: str):
    """Remove a platform adapter configuration."""
    cfg = cliver.config_manager.config
    if not cfg.gateway or name not in cfg.gateway.platforms:
        cliver.output(f"[red]Platform '{name}' not found.[/red]")
        return

    response = cliver.ui.ask_input(f"  Remove platform '{name}'? [y/n] > ", choices=["y", "yes", "n", "no"])
    if response.lower() not in ("y", "yes"):
        cliver.output("Cancelled.")
        return

    del cfg.gateway.platforms[name]
    cliver.config_manager._save_config()
    cliver.output(f"Removed platform '{name}'.")


def _mask_token(token: str) -> str:
    """Mask token for display."""
    if "{{" in token and "}}" in token:
        return token
    if len(token) <= 8:
        return "****"
    return f"{token[:4]}****{token[-4:]}"


def _status_gateway(cliver: Cliver):
    """Show gateway status via PID check + HTTP /health."""
    pid_path = _get_pid_path()
    if not pid_path.exists():
        cliver.output("Gateway is not running.")
        return 1

    try:
        pid = int(pid_path.read_text().strip())
        os.kill(pid, 0)
    except (ProcessLookupError, ValueError):
        pid_path.unlink(missing_ok=True)
        cliver.output("Gateway is not running (stale PID file cleaned up).")
        return 1

    cfg = cliver.config_manager.config
    gw_cfg = cfg.gateway
    host = gw_cfg.host if gw_cfg else "127.0.0.1"
    port = gw_cfg.port if gw_cfg else 8321

    import urllib.request

    try:
        url = f"http://{host}:{port}/health"
        with urllib.request.urlopen(url, timeout=3) as resp:
            data = json.loads(resp.read())

        uptime = data.get("uptime", 0)
        days, remainder = divmod(uptime, 86400)
        hours, remainder = divmod(remainder, 3600)
        minutes, seconds = divmod(remainder, 60)
        parts = []
        if days:
            parts.append(f"{days}d")
        if hours or days:
            parts.append(f"{hours}h")
        if minutes or hours or days:
            parts.append(f"{minutes}m")
        parts.append(f"{seconds}s")
        uptime_str = " ".join(parts)

        cliver.output(f"  Status: running (PID {pid})")
        cliver.output(f"  URL: http://{host}:{port}")
        cliver.output(f"  Uptime: {uptime_str}")
        cliver.output(f"  Tasks run: {data.get('tasks_run', 0)}")

        adapters = data.get("adapters", [])
        if adapters:
            cliver.output("  Adapters:")
            for a in adapters:
                state = a.get("state", "unknown")
                name = a.get("name", "?")
                if state == "connected":
                    cliver.output(f"    [green]●[/green] {name} [dim](running)[/dim]")
                elif state == "connecting":
                    cliver.output(f"    [yellow]◌[/yellow] {name} [dim](connecting...)[/dim]")
                elif state == "error":
                    err = a.get("error", "")
                    cliver.output(f"    [red]✗[/red] {name} [dim]({err})[/dim]")
                else:
                    cliver.output(f"    [dim]?[/dim] {name} [dim]({state})[/dim]")
        else:
            cliver.output("  Adapters: none")
    except Exception:
        cliver.output(f"  Status: process alive (PID {pid}) but not responding")
        cliver.output(f"  URL: http://{host}:{port} (unreachable)")

    return 0


# ── Dispatch ──


def dispatch(cliver: Cliver, args: str):
    """Manage the gateway daemon — start, stop, restart, status, platforms."""
    parts = args.strip().split(None, 1) if args.strip() else []
    sub = parts[0] if parts else "status"
    rest = parts[1] if len(parts) > 1 else ""

    if sub == "start":
        _start_gateway(cliver)
    elif sub == "stop":
        _stop_gateway(cliver)
    elif sub == "restart":
        _restart_gateway(cliver)
    elif sub == "status":
        _status_gateway(cliver)
    elif sub == "platform":
        _dispatch_platform(cliver, rest)
    elif sub in ("--help", "help"):
        cliver.output("Manage the gateway daemon. The gateway runs as a background process that")
        cliver.output("handles cron-scheduled tasks and platform adapters (Slack, Telegram, etc.).")
        cliver.output("")
        cliver.output("Usage: /gateway [start|stop|restart|status|platform] [arguments]")
        cliver.output("")
        cliver.output("Subcommands:")
        cliver.output("  start     — Start the gateway daemon as a background subprocess.")
        cliver.output("              Checks for existing PID file and port conflicts before starting.")
        cliver.output("              No parameters.")
        cliver.output("  stop      — Stop the running gateway daemon via SIGTERM.")
        cliver.output("              No parameters.")
        cliver.output("  restart   — Stop and then start the gateway daemon.")
        cliver.output("              No parameters.")
        cliver.output("  status    — Show gateway daemon status: PID, uptime, tasks run, adapter states.")
        cliver.output("              Checks both PID file and HTTP /health endpoint.")
        cliver.output("              No parameters.")
        cliver.output("  platform  — Manage messaging platform adapters (Slack, Telegram, Discord, Feishu).")
        cliver.output("              See '/gateway platform help' for subcommands.")
        cliver.output("")
        cliver.output("Default subcommand: status (when /gateway is called with no arguments)")
        cliver.output("")
        cliver.output("Examples:")
        cliver.output("  /gateway start    — start the daemon")
        cliver.output("  /gateway status   — check if running and show health info")
        cliver.output("  /gateway restart  — restart after config changes")
    else:
        cliver.output(f"[yellow]Unknown: /gateway {sub}[/yellow]")


def _dispatch_platform(cliver: Cliver, args: str):
    """Dispatch /gateway platform subcommands."""
    parts = args.strip().split(None, 1) if args.strip() else []
    sub = parts[0] if parts else "list"
    rest = parts[1] if len(parts) > 1 else ""

    if sub == "list":
        _list_platforms(cliver)
    elif sub in ("setup", "add"):
        _setup_platform(cliver)
    elif sub == "set":
        if not rest:
            cliver.output("[yellow]Usage: /gateway platform set <name> --token T ...[/yellow]")
            return
        from shlex import split as shlex_split

        try:
            p = shlex_split(rest)
        except ValueError:
            p = rest.split()
        name = p[0]
        kwargs = {}
        i = 1
        while i < len(p):
            if p[i] == "--token" and i + 1 < len(p):
                kwargs["token"] = p[i + 1]
                i += 2
            elif p[i] == "--app-token" and i + 1 < len(p):
                kwargs["app_token"] = p[i + 1]
                i += 2
            elif p[i] == "--home-channel" and i + 1 < len(p):
                kwargs["home_channel"] = p[i + 1]
                i += 2
            elif p[i] == "--allowed-users" and i + 1 < len(p):
                kwargs["allowed_users"] = p[i + 1]
                i += 2
            else:
                cliver.output(f"[yellow]Unknown option: {p[i]}[/yellow]")
                return
        _set_platform(cliver, name, **kwargs)
    elif sub == "remove":
        if not rest:
            cliver.output("[yellow]Usage: /gateway platform remove <name>[/yellow]")
            return
        _remove_platform(cliver, rest.strip())
    elif sub in ("--help", "help"):
        cliver.output("Manage messaging platform adapters that connect CLIver to Slack, Telegram, etc.")
        cliver.output("")
        cliver.output("Usage: /gateway platform [list|setup|set|remove] [arguments]")
        cliver.output("")
        cliver.output("Subcommands:")
        cliver.output("  list              — List configured platforms with name, type, connection status,")
        cliver.output("                      and home channel. Shows live status if gateway is running.")
        cliver.output("  setup             — Start an interactive wizard to add or reconfigure a platform.")
        cliver.output("                      Guides through: platform type, adapter name, credentials,")
        cliver.output("                      home channel, and allowed users.")
        cliver.output("  set <name>        — Update specific fields of an existing platform adapter.")
        cliver.output("    name  STRING (required) — Platform adapter name from '/gateway platform list'.")
        cliver.output("    --token         STRING (optional) — New bot token.")
        cliver.output("    --app-token     STRING (optional) — New app-level token (Slack Socket Mode).")
        cliver.output("    --home-channel  STRING (optional) — New default output channel ID.")
        cliver.output("    --allowed-users STRING (optional) — Comma-separated user IDs (empty = all).")
        cliver.output("    Example: /gateway platform set slack --home-channel C012345")
        cliver.output("  remove <name>     — Remove a platform adapter (requires confirmation).")
        cliver.output("    name  STRING (required) — Platform adapter name to remove.")
        cliver.output("")
        cliver.output("Default subcommand: list")
        cliver.output("")
        cliver.output("Note: Restart the gateway after any platform changes to apply them.")
    else:
        cliver.output(f"[yellow]Unknown: /gateway platform {sub}[/yellow]")


@click.group(
    name="gateway",
    help="Manage the gateway daemon (cron scheduler for tasks & platform adapters)",
)
def gateway_cmd():
    """Gateway daemon management."""
    pass


@gateway_cmd.command(name="start", help="Start the gateway daemon as a background subprocess")
@pass_cliver
def start(cliver: Cliver):
    """Start the gateway daemon."""
    return _start_gateway(cliver)


@gateway_cmd.command(name="stop", help="Stop the running gateway daemon via SIGTERM")
@pass_cliver
def stop(cliver: Cliver):
    """Stop the gateway process."""
    return _stop_gateway(cliver)


@gateway_cmd.command(name="restart", help="Stop and restart the gateway daemon (apply config changes)")
@pass_cliver
def restart(cliver: Cliver):
    """Restart the gateway daemon."""
    return _restart_gateway(cliver)


@gateway_cmd.command(
    name="status",
    help="Show gateway daemon status: PID, uptime, tasks run, adapter states",
)
@pass_cliver
def status(cliver: Cliver):
    """Show gateway status."""
    return _status_gateway(cliver)
