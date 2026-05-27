"""CLI /mcp command — manage MCP servers via config.yaml."""

import click
from rich import box
from rich.table import Table

from cliver.cli import Cliver, pass_cliver
from cliver.commands import click_help, wants_help
from cliver.util import parse_key_value_options


@click.group(
    name="mcp",
    help="Manage MCP (Model Context Protocol) server connections that provide tools to the agent",
)
@click.pass_context
def mcp(ctx: click.Context):
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())
        ctx.exit()


# ---------------------------------------------------------------------------
# Business logic functions
# ---------------------------------------------------------------------------


def _list_mcp_servers(cliver: Cliver):
    servers = cliver.config_manager.list_mcp_servers()
    if servers:
        table = Table(title="Configured MCP Servers", box=box.SQUARE)
        table.add_column("Name", style="green")
        table.add_column("Transport")
        table.add_column("Info", style="blue")
        for name, srv in servers.items():
            if srv.transport == "stdio":
                info = f"{srv.command or ''}"
                if srv.args:
                    info += f" {srv.args}"
            else:
                info = srv.url or ""
                if hasattr(srv, "headers") and srv.headers:
                    info += " [headers]"
            table.add_row(name, srv.transport, info)
        cliver.output(table)
    else:
        cliver.output("No MCP servers configured.")


def _add_mcp_server(
    cliver: Cliver,
    name: str,
    transport: str,
    command: str = None,
    args: str = None,
    env: tuple = None,
    url: str = None,
    header: tuple = None,
):
    if name in cliver.config_manager.list_mcp_servers():
        cliver.output(f"MCP server with name '{name}' already exists.")
        return

    args_list = None
    if args:
        args_list = [a.strip() for a in args.split(",") if a.strip()]

    env_dict = None
    if env:
        env_dict = parse_key_value_options(env, cliver.console)

    headers_dict = None
    if header:
        headers_dict = parse_key_value_options(header, cliver.console)

    try:
        cliver.config_manager.add_or_update_mcp_server(
            name=name,
            transport=transport,
            command=command,
            args=args_list,
            env=env_dict,
            url=url,
            headers=headers_dict,
        )
    except ValueError as e:
        cliver.output(f"{e}")
        return

    if transport == "sse":
        cliver.output("Note: SSE transport is deprecated, consider using streamable_http instead")
    cliver.output(f"Added MCP server: {name} of transport {transport}")


def _set_mcp_server(
    cliver: Cliver,
    name: str,
    command: str = None,
    args: str = None,
    url: str = None,
    header: tuple = None,
    env: tuple = None,
):
    existing = cliver.config_manager.list_mcp_servers().get(name)
    if not existing:
        cliver.output(f"No MCP server found with name: {name}")
        return

    args_list = None
    if args is not None:
        args_list = [a.strip() for a in args.split(",") if a.strip()]

    env_dict = None
    if env is not None:
        env_dict = parse_key_value_options(env, cliver.console)

    headers_dict = None
    if header is not None:
        headers_dict = parse_key_value_options(header, cliver.console)

    try:
        cliver.config_manager.add_or_update_mcp_server(
            name=name,
            transport=existing.transport,
            command=command or getattr(existing, "command", None),
            args=args_list if args_list else getattr(existing, "args", None),
            env=env_dict or getattr(existing, "env", None),
            url=url or getattr(existing, "url", None),
            headers=headers_dict or getattr(existing, "headers", None),
        )
    except ValueError as e:
        cliver.output(f"{e}")
        return

    cliver.output(f"Updated MCP server: {name}")


def _show_mcp_server(cliver: Cliver, name: str):
    """Show detailed information about a specific MCP server."""
    servers = cliver.config_manager.list_mcp_servers()
    srv = servers.get(name)
    if not srv:
        cliver.output(f"No MCP server found with name: {name}")
        return

    from rich.panel import Panel

    t = Table(box=None, show_header=False, padding=(0, 2))
    t.add_column("Key", style="dim", min_width=12)
    t.add_column("Value")

    t.add_row("Name", name)
    t.add_row("Transport", srv.transport)
    if srv.transport == "stdio":
        t.add_row("Command", getattr(srv, "command", "-"))
        args = getattr(srv, "args", None)
        t.add_row("Args", " ".join(args) if args else "-")
        env = getattr(srv, "env", None)
        t.add_row("Env", ", ".join(f"{k}={v}" for k, v in env.items()) if env else "-")
    else:
        t.add_row("URL", getattr(srv, "url", "-"))
        headers = getattr(srv, "headers", None)
        t.add_row("Headers", ", ".join(f"{k}: {v}" for k, v in headers.items()) if headers else "-")

    cliver.output(Panel(t, title=f"MCP Server: {name}", border_style="green", padding=(0, 1)))


def _remove_mcp_server(cliver: Cliver, name: str):
    if cliver.config_manager.remove_mcp_server(name):
        cliver.output(f"Removed MCP server: {name}")
    else:
        cliver.output(f"No MCP server found with name: {name}")


# ---------------------------------------------------------------------------
# Dispatch function
# ---------------------------------------------------------------------------

_SUBCOMMANDS: dict[str, click.Command] = {}


def _parse_mcp_flags(rest: str) -> dict:
    """Parse --flag value pairs from a rest string for MCP commands."""
    from shlex import split as shlex_split

    try:
        tokens = shlex_split(rest)
    except ValueError:
        tokens = rest.split()

    opts: dict = {}
    envs = []
    headers_list = []
    i = 0
    flag_map = {
        "--name": "name",
        "-n": "name",
        "--transport": "transport",
        "-t": "transport",
        "--command": "command",
        "-c": "command",
        "--args": "args",
        "-a": "args",
        "--url": "url",
        "-u": "url",
    }
    while i < len(tokens):
        if tokens[i] in flag_map and i + 1 < len(tokens):
            opts[flag_map[tokens[i]]] = tokens[i + 1]
            i += 2
        elif tokens[i] in ("--env", "-e") and i + 1 < len(tokens):
            envs.append(tokens[i + 1])
            i += 2
        elif tokens[i] == "--header" and i + 1 < len(tokens):
            headers_list.append(tokens[i + 1])
            i += 2
        else:
            i += 1
    if envs:
        opts["env"] = tuple(envs)
    if headers_list:
        opts["header"] = tuple(headers_list)
    return opts


def dispatch(cliver: Cliver, args: str):
    """Manage MCP server connections — list, add, set, remove."""
    parts = args.strip().split(None, 1) if args.strip() else []
    sub = parts[0] if parts else "list"
    rest = parts[1] if len(parts) > 1 else ""

    if sub in ("--help", "-h", "help"):
        cliver.output(click_help(mcp, "/mcp"))
        return

    if sub in _SUBCOMMANDS and wants_help(rest):
        cliver.output(click_help(_SUBCOMMANDS[sub], f"/mcp {sub}"))
        return

    if sub == "list":
        _list_mcp_servers(cliver)
    elif sub == "show":
        name = rest.strip()
        if not name:
            cliver.output(click_help(_SUBCOMMANDS["show"], "/mcp show"))
            return
        _show_mcp_server(cliver, name)
    elif sub == "add":
        _dispatch_add(cliver, rest)
    elif sub == "set":
        _dispatch_set(cliver, rest)
    elif sub == "remove":
        name = rest.strip()
        if not name:
            cliver.output(click_help(_SUBCOMMANDS["remove"], "/mcp remove"))
            return
        _remove_mcp_server(cliver, name)
    else:
        cliver.output(f"Unknown subcommand: /mcp {sub}")
        cliver.output("Run '/mcp help' for usage.")


def _dispatch_add(cliver: Cliver, rest: str) -> None:
    opts = _parse_mcp_flags(rest)
    name = opts.get("name")
    transport = opts.get("transport", "stdio")
    if not name:
        cliver.output("Usage: /mcp add -n NAME [-t TRANSPORT] [-c COMMAND] [-a ARGS] [-u URL] [-e K=V] [--header K=V]")
        return
    _add_mcp_server(
        cliver,
        name=name,
        transport=transport,
        command=opts.get("command"),
        args=opts.get("args"),
        url=opts.get("url"),
        env=opts.get("env"),
        header=opts.get("header"),
    )


def _dispatch_set(cliver: Cliver, rest: str) -> None:
    opts = _parse_mcp_flags(rest)
    name = opts.get("name")
    if not name:
        cliver.output("Usage: /mcp set -n NAME [-c COMMAND] [-a ARGS] [-u URL] [-e K=V] [--header K=V]")
        return
    _set_mcp_server(
        cliver,
        name=name,
        command=opts.get("command"),
        args=opts.get("args"),
        url=opts.get("url"),
        env=opts.get("env"),
        header=opts.get("header"),
    )


# ---------------------------------------------------------------------------
# Click commands (thin wrappers)
# ---------------------------------------------------------------------------


@mcp.command(name="list", help="List all configured MCP servers with name, transport type, and connection info")
@pass_cliver
def list_mcp_servers(cliver: Cliver):
    _list_mcp_servers(cliver)


@mcp.command(name="set", help="Update an existing MCP server's configuration (only provided values are changed)")
@click.option("--name", "-n", type=str, required=True, help="Name of the MCP server to update")
@click.option("--command", "-c", type=str, help="Executable command for stdio transport (e.g. 'npx', 'uvx')")
@click.option("--args", "-a", type=str, help="Comma-separated arguments for the stdio command (e.g. '-y,package-name')")
@click.option("--env", "-e", multiple=True, type=str, help="Environment vars as KEY=VALUE (repeatable)")
@click.option("--url", "-u", type=str, help="URL for SSE/streamable_http/websocket transport")
@click.option("--header", multiple=True, type=str, help="HTTP headers as Key=Value (repeatable)")
@pass_cliver
def set_mcp_server(cliver: Cliver, name: str, command: str, args: str, env: tuple, url: str, header: tuple):
    _set_mcp_server(cliver, name, command, args, url, header, env)


@mcp.command(name="remove", help="Remove a configured MCP server and its tools from the agent")
@click.option("--name", "-n", type=str, required=True, help="Name of the MCP server to remove")
@pass_cliver
def remove_mcp_server(cliver: Cliver, name: str):
    _remove_mcp_server(cliver, name)


@mcp.command(
    name="add",
    help="Add a new MCP server connection with transport-specific options (stdio, sse, streamable, websocket)",
)
@click.option("--name", "-n", type=str, required=True, help="Server name (e.g. 'github', 'web-fetch')")
@click.option(
    "--transport",
    "-t",
    type=click.Choice(["stdio", "sse", "streamable", "websocket"]),
    default="stdio",
    help="Transport protocol",
)
@click.option("--command", "-c", type=str, help="Executable command (stdio transport, e.g. 'npx', 'uvx')")
@click.option("--args", "-a", type=str, help="Comma-separated arguments for stdio command (e.g. '-y,package')")
@click.option(
    "--env", "-e", multiple=True, type=str, help="Environment variables as KEY=VALUE (repeatable, stdio transport)"
)
@click.option("--url", "-u", type=str, help="URL for SSE/streamable_http/websocket transport")
@click.option("--header", multiple=True, type=str, help="HTTP headers as Key=Value (repeatable, SSE/streamable)")
@pass_cliver
def add_mcp_server(
    cliver: Cliver, name: str, transport: str, command: str, args: str, env: tuple, url: str, header: tuple
):
    _add_mcp_server(cliver, name, transport, command, args, env, url, header)


@mcp.command(name="show", help="Show detailed information about a specific MCP server")
@click.option("--name", "-n", type=str, required=True, help="MCP server name")
@pass_cliver
def show_mcp_server(cliver: Cliver, name: str):
    _show_mcp_server(cliver, name)


# Populate subcommand map for dispatch help
_SUBCOMMANDS.update(
    {
        "list": list_mcp_servers,
        "show": show_mcp_server,
        "add": add_mcp_server,
        "set": set_mcp_server,
        "remove": remove_mcp_server,
    }
)
