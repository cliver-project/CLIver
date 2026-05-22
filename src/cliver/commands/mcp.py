"""CLI /mcp command — manage MCP servers via the database."""

import json

import click
from rich import box
from rich.table import Table

from cliver.cli import Cliver, pass_cliver
from cliver.commands import click_help, wants_help
from cliver.util import parse_key_value_options


def _get_store(cliver: Cliver):
    """Get MCPServerStore for the current profile's database."""
    from cliver.mcp.store import MCPServerStore

    return MCPServerStore.from_config_dir(cliver.config_dir)


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
    store = _get_store(cliver)
    servers = store.list_servers()
    if servers:
        table = Table(title="Configured MCP Servers", box=box.SQUARE)
        table.add_column("ID", style="dim")
        table.add_column("Name", style="green")
        table.add_column("Transport")
        table.add_column("Info", style="blue")
        for srv in servers:
            if srv.transport == "stdio":
                info = f"{srv.command or ''}"
                if srv.args:
                    try:
                        args_list = json.loads(srv.args)
                        info += f" {args_list}"
                    except (json.JSONDecodeError, TypeError):
                        info += f" {srv.args}"
            else:
                info = srv.url or ""
                if srv.headers:
                    info += " [headers]"
                if srv.auth:
                    try:
                        auth_data = json.loads(srv.auth)
                        info += f" [auth: {auth_data.get('type', 'token')}]"
                    except (json.JSONDecodeError, TypeError):
                        info += " [auth]"
            table.add_row(srv.id, srv.name, srv.transport, info)
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
    store = _get_store(cliver)
    # Check for name collision
    existing = store.list_servers()
    if any(s.name == name for s in existing):
        cliver.output(f"MCP server with name '{name}' already exists.")
        return

    # Build args/envs/headers as JSON strings
    args_json = None
    if args:
        args_list = [a.strip() for a in args.split(",") if a.strip()]
        args_json = json.dumps(args_list)

    envs_json = None
    if env:
        env_dict = parse_key_value_options(env, cliver.console)
        if env_dict:
            envs_json = json.dumps(env_dict)

    headers_json = None
    if header:
        header_dict = parse_key_value_options(header, cliver.console)
        if header_dict:
            headers_json = json.dumps(header_dict)

    if transport == "stdio":
        if command is None:
            cliver.output("Command is required for stdio transport")
            return
        store.create_server(
            name=name,
            transport="stdio",
            command=command,
            args=args_json,
            envs=envs_json,
        )
    elif transport in ("sse", "streamable", "websocket"):
        if url is None:
            cliver.output(f"URL is required for {transport} transport")
            return
        db_transport = "streamable_http" if transport == "streamable" else transport
        store.create_server(
            name=name,
            transport=db_transport,
            url=url,
            headers=headers_json,
        )
        if transport == "sse":
            cliver.output("Note: SSE transport is deprecated, consider using streamable_http instead")
    else:
        cliver.output(f"Unsupported MCP server transport: {transport}")
        return

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
    store = _get_store(cliver)
    existing = store.list_servers()
    match = next((s for s in existing if s.name == name), None)
    if not match:
        cliver.output(f"No MCP server found with name: {name}")
        return

    kwargs = {}
    if command is not None:
        kwargs["command"] = command
    if args is not None:
        args_list = [a.strip() for a in args.split(",") if a.strip()]
        kwargs["args"] = json.dumps(args_list)
    if env is not None:
        env_dict = parse_key_value_options(env, cliver.console)
        kwargs["envs"] = json.dumps(env_dict) if env_dict else None
    if url is not None:
        kwargs["url"] = url
    if header is not None:
        header_dict = parse_key_value_options(header, cliver.console)
        kwargs["headers"] = json.dumps(header_dict) if header_dict else None

    updated = store.update_server(match.id, **kwargs)
    if updated:
        cliver.output(f"Updated MCP server: {name}")


def _remove_mcp_server(cliver: Cliver, name: str):
    store = _get_store(cliver)
    existing = store.list_servers()
    match = next((s for s in existing if s.name == name), None)
    if not match:
        cliver.output(f"No MCP server found with name: {name}")
        return
    store.delete_server(match.id)
    cliver.output(f"Removed MCP server: {name}")


# ---------------------------------------------------------------------------
# Dispatch function
# ---------------------------------------------------------------------------

_SUBCOMMANDS: dict[str, click.Command] = {}


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
    else:
        cliver.output(f"Unknown subcommand: /mcp {sub}")
        cliver.output("Run '/mcp help' for usage.")


# ---------------------------------------------------------------------------
# Click commands (thin wrappers)
# ---------------------------------------------------------------------------


@mcp.command(name="list", help="List all configured MCP servers with name, transport type, and connection info")
@pass_cliver
def list_mcp_servers(cliver: Cliver):
    _list_mcp_servers(cliver)


@mcp.command(name="set", help="Update an existing MCP server's configuration (only provided values are changed)")
@click.option(
    "--name",
    "-n",
    type=str,
    required=True,
    help="Name of the MCP server to update. Must match an existing server from 'mcp list'",
)
@click.option(
    "--command",
    "-c",
    type=str,
    help="Executable command for stdio transport (e.g. 'npx', 'uvx')",
)
@click.option(
    "--args",
    "-a",
    type=str,
    help="Comma-separated arguments for the stdio command (e.g. '-y,@modelcontextprotocol/server-github')",
)
@click.option(
    "--env",
    "-e",
    multiple=True,
    type=str,
    help="Environment variable as KEY=VALUE (repeatable, e.g. -e GITHUB_TOKEN=ghp_xxx)",
)
@click.option(
    "--url",
    "-u",
    type=str,
    help="Server URL for sse/streamable/websocket transport (e.g. 'http://localhost:3000/mcp')",
)
@click.option(
    "--header",
    "-H",
    multiple=True,
    type=str,
    help="HTTP header as KEY=VALUE for sse/streamable/websocket (repeatable, e.g. -H Authorization=Bearer...)",
)
@pass_cliver
def set_mcp_server(
    cliver: Cliver,
    name: str,
    command: str = None,
    args: str = None,
    url: str = None,
    header: tuple = None,
    env: tuple = None,
):
    _set_mcp_server(cliver, name, command, args, url, header, env)


@mcp.command(name="add", help="Add a new MCP server connection with transport-specific configuration")
@click.option(
    "--transport",
    "-t",
    type=click.Choice(["stdio", "sse", "streamable", "websocket"]),
    default="stdio",
    help="Transport protocol (default: stdio). 'sse' is deprecated, use 'streamable' instead",
)
@click.option(
    "--name",
    "-n",
    type=str,
    required=True,
    help="Unique server name used as identifier (e.g. 'github', 'jira'). Must not already exist",
)
@click.option(
    "--command",
    "-c",
    type=str,
    help="Executable command for stdio transport (required for stdio, e.g. 'npx', 'uvx')",
)
@click.option(
    "--args",
    "-a",
    type=str,
    help="Comma-separated arguments for the stdio command (e.g. '-y,@modelcontextprotocol/server-github')",
)
@click.option(
    "--env",
    "-e",
    multiple=True,
    type=str,
    help="Environment variable as KEY=VALUE for stdio (repeatable, e.g. -e GITHUB_TOKEN=ghp_xxx)",
)
@click.option(
    "--url",
    "-u",
    type=str,
    help="Server URL for sse/streamable/websocket transport (required for non-stdio, e.g. 'http://localhost:3000/mcp')",
)
@click.option(
    "--header",
    "-H",
    multiple=True,
    type=str,
    help="HTTP header as KEY=VALUE for sse/streamable/websocket (repeatable, e.g. -H Authorization=Bearer...)",
)
@pass_cliver
def add_mcp_server(
    cliver: Cliver,
    name: str,
    transport: str,
    command: str = None,
    args: str = None,
    env: tuple = None,
    url: str = None,
    header: tuple = None,
):
    _add_mcp_server(cliver, name, transport, command, args, env, url, header)


@mcp.command(name="remove", help="Remove an MCP server connection by name")
@click.option(
    "--name",
    "-n",
    type=str,
    required=True,
    help="Name of the MCP server to remove. Must match an existing server from 'mcp list'",
)
@pass_cliver
def remove_mcp_server(cliver: Cliver, name: str):
    _remove_mcp_server(cliver, name)


# Populate subcommand map for dispatch help generation.
_SUBCOMMANDS.update(
    {
        "list": list_mcp_servers,
        "add": add_mcp_server,
        "set": set_mcp_server,
        "remove": remove_mcp_server,
    }
)
