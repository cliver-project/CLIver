import click
from rich import box
from rich.table import Table

from cliver.cli import Cliver, pass_cliver
from cliver.config import (
    SSEMCPServerConfig,
    StdioMCPServerConfig,
    StreamableHttpMCPServerConfig,
    WebSocketMCPServerConfig,
)
from cliver.util import parse_key_value_options


@click.group(name="mcp", help="Manage MCP (Model Context Protocol) server connections that provide tools to the agent")
@click.pass_context
def mcp(ctx: click.Context):
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())
        ctx.exit()


# ---------------------------------------------------------------------------
# Business logic functions
# ---------------------------------------------------------------------------


def _list_mcp_servers(cliver: Cliver):
    """List all MCP servers."""
    mcp_servers = cliver.config_manager.list_mcp_servers()
    if mcp_servers:
        table = Table(title="Configured MCP Servers", box=box.SQUARE)
        table.add_column("Name", style="green")
        table.add_column("Transport")
        table.add_column("Info", style="blue")
        for name, _mcp_server in mcp_servers.items():
            if isinstance(_mcp_server, StdioMCPServerConfig):
                info = f"{_mcp_server.command} {_mcp_server.args or ''} {_mcp_server.env or ''}"
            elif (
                isinstance(_mcp_server, SSEMCPServerConfig)
                or isinstance(_mcp_server, StreamableHttpMCPServerConfig)
                or isinstance(_mcp_server, WebSocketMCPServerConfig)
            ):
                info = f"{_mcp_server.url} {_mcp_server.headers or ''}"
            else:
                info = "Unknown server type"
            table.add_row(
                name,
                _mcp_server.transport,
                info,
            )
        cliver.output(table)
    else:
        cliver.output("No MCP servers configured.")


def _set_mcp_server(
    cliver: Cliver,
    name: str,
    command: str = None,
    args: str = None,
    url: str = None,
    header: tuple = None,
    env: tuple = None,
):
    """Update an MCP server."""
    mcp_server = cliver.config_manager.get_mcp_server(name)
    if not mcp_server:
        cliver.output(f"No MCP server found with name: {name}")
        return

    # Update server based on its type
    if isinstance(mcp_server, StdioMCPServerConfig):
        if command:
            mcp_server.command = command
        if args:
            mcp_server.args = args.split(",")
        if env:
            # Parse env variables from key=value format
            env_dict = parse_key_value_options(env, cliver.console)
            mcp_server.env = env_dict
    elif (
        isinstance(mcp_server, SSEMCPServerConfig)
        or isinstance(mcp_server, StreamableHttpMCPServerConfig)
        or isinstance(mcp_server, WebSocketMCPServerConfig)
    ):
        if url:
            mcp_server.url = url
        if header:
            # Parse headers from key=value format
            header_dict = parse_key_value_options(header, cliver.console)
            mcp_server.headers = header_dict
    cliver.config_manager.add_or_update_server(name, mcp_server)
    cliver.output(f"Updated MCP server: {name}")


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
    """Add a new MCP server."""
    mcp_server = cliver.config_manager.get_mcp_server(name)
    if mcp_server:
        cliver.output(f"MCP server with name {name} already exists.")
        return
    if transport == "stdio":
        if command is None:
            cliver.output("Command is required for stdio transport")
            return
        # Parse env variables from key=value format
        env_dict = parse_key_value_options(env, cliver.console)
        cliver.config_manager.add_or_update_stdio_mcp_server(
            name=name,
            command=command,
            args=args.split(",") if args else None,
            env=env_dict,
        )
    elif transport == "sse":
        cliver.output("Warning: SSE transport is deprecated, consider using streamable instead")
        if url is None:
            cliver.output("URL is required for sse transport")
            return
        # Parse headers from key=value format
        header_dict = parse_key_value_options(header, cliver.console)
        cliver.config_manager.add_or_update_sse_mcp_server(name=name, url=url, headers=header_dict)
    elif transport == "streamable":
        if url is None:
            cliver.output("URL is required for streamable transport")
            return
        # Parse headers from key=value format
        header_dict = parse_key_value_options(header, cliver.console)
        cliver.config_manager.add_or_update_streamable_mcp_server(name=name, url=url, headers=header_dict)
    elif transport == "websocket":
        if url is None:
            cliver.output("URL is required for websocket transport")
            return
        # Parse headers from key=value format
        header_dict = parse_key_value_options(header, cliver.console)
        cliver.config_manager.add_or_update_websocket_mcp_server(name=name, url=url, headers=header_dict)
    else:
        cliver.output(f"Unsupported MCP server transport: {transport}")
    cliver.output(f"Added MCP server: {name} of transport {transport}")


def _remove_mcp_server(cliver: Cliver, name: str):
    """Remove an MCP server."""
    mcp_server = cliver.config_manager.get_mcp_server(name)
    if not mcp_server:
        cliver.output(f"No MCP server found with name: {name}")
        return
    cliver.config_manager.remove_mcp_server(name)
    cliver.output(f"Removed MCP server: {name}")


# ---------------------------------------------------------------------------
# Dispatch function
# ---------------------------------------------------------------------------


def dispatch(cliver: Cliver, args: str):
    """Manage MCP server connections — list, add, set, remove."""
    parts = args.strip().split(None, 1) if args.strip() else []
    sub = parts[0] if parts else "list"

    if sub == "list":
        _list_mcp_servers(cliver)
    elif sub in ("--help", "help"):
        cliver.output("Manage MCP (Model Context Protocol) server connections.")
        cliver.output("MCP servers provide additional tools to the LLM agent.")
        cliver.output("")
        cliver.output("Usage: /mcp [list|add|set|remove] [options]")
        cliver.output("")
        cliver.output("Subcommands:")
        cliver.output("  list   — List all configured MCP servers with name, transport, and connection info.")
        cliver.output("           No parameters.")
        cliver.output("")
        cliver.output("  add    — Add a new MCP server connection.")
        cliver.output("    --name, -n       STRING (required) — Unique server name (e.g. 'github', 'jira').")
        cliver.output("    --transport, -t  CHOICE(stdio|sse|streamable|websocket) (optional, default: stdio)")
        cliver.output("                     Transport protocol. 'sse' is deprecated, use 'streamable' instead.")
        cliver.output("    For stdio transport:")
        cliver.output("      --command, -c  STRING (required for stdio) — Executable command (e.g. 'npx').")
        cliver.output("      --args, -a     STRING (optional) — Comma-separated arguments for the command.")
        cliver.output("      --env, -e      STRING (optional, repeatable) — Env var as KEY=VALUE.")
        cliver.output("    For sse/streamable/websocket transport:")
        cliver.output("      --url, -u      STRING (required) — Server URL (e.g. 'http://localhost:3000/mcp').")
        cliver.output("      --header, -H   STRING (optional, repeatable) — HTTP header as KEY=VALUE.")
        cliver.output("    Example: /mcp add --name github --command npx --args -y,@modelcontextprotocol/server-github")
        cliver.output("    Example: /mcp add --name api --transport streamable --url http://localhost:3000/mcp")
        cliver.output("")
        cliver.output("  set    — Update an existing MCP server's configuration.")
        cliver.output("    --name, -n  STRING (required) — Name of the server to update.")
        cliver.output("    (same transport-specific options as 'add', only provided values are changed)")
        cliver.output("    Example: /mcp set --name github --env GITHUB_TOKEN=ghp_xxx")
        cliver.output("")
        cliver.output("  remove — Remove an MCP server connection.")
        cliver.output("    --name, -n  STRING (required) — Name of the server to remove.")
        cliver.output("    Example: /mcp remove --name github")
        cliver.output("")
        cliver.output("Default subcommand: list (when /mcp is called with no arguments)")
    else:
        cliver.output(f"[yellow]Unknown subcommand: /mcp {sub}[/yellow]")
        cliver.output("Run '/mcp help' for usage.")


# ---------------------------------------------------------------------------
# Click commands (thin wrappers)
# ---------------------------------------------------------------------------


# noinspection PyUnresolvedReferences
@mcp.command(name="list", help="List all configured MCP servers with name, transport type, and connection info")
@pass_cliver
def list_mcp_servers(cliver: Cliver):
    _list_mcp_servers(cliver)


# noinspection PyUnresolvedReferences
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


# noinspection PyUnresolvedReferences
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


# noinspection PyUnresolvedReferences
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
