import click
import json
from cliver.cli import Cliver, pass_cliver
from cliver.config import MCPServerStdio, MCPServerSSE
from rich.table import Table
from rich import box


@click.group(name="config", help="Manage configuration settings.")
def config():
    """
    Configuration command group.
    This group contains commands to manage configuration settings.
    """
    pass


@config.group(name="mcp", help="Manage the MCP Servers")
def mcp():
    pass


def post_group():
    # config.add_command(mcp)
    pass


@mcp.command(name="list", help="List MCP servers")
@pass_cliver
def list_mcp_servers(cliver: Cliver):
    mcp_servers = cliver.config_manager.list_mcp_servers()
    if mcp_servers:
        table = Table(title="Configured MCP Servers", box=box.SQUARE)
        table.add_column("Name", style="green")
        table.add_column("Type")
        table.add_column("Info", style="blue")
        for mcp in mcp_servers:
            table.add_row(
                mcp.name,
                mcp.type,
                mcp.info(),
            )
        cliver.console.print(table)
    else:
        cliver.console.print("No MCP servers configured.")


@mcp.command(name="set", help="Update the MCP server")
@click.option(
    "--name",
    "-n",
    type=str,
    required=True,
    help="Name of the MCP server",
)
@click.option(
    "--command",
    "-c",
    type=str,
    help="Command of the stdio MCP server",
)
@click.option(
    "--args",
    "-a",
    type=str,
    help="Comma separated arguments for the stdio MCP server",
)
@click.option(
    "--envs",
    "-e",
    type=str,
    help="Environment settings in JSON format for stdio MCP server",
)
@click.option(
    "--url",
    "-u",
    type=str,
    help="The URL for the sse MCP server",
)
@click.option(
    "--headers",
    "-h",
    type=str,
    help="HTTP headers in JSON format for sse MCP server",
)
@pass_cliver
def set_mcp_server(cliver: Cliver, name: str, command: str = None, args: str = None, url: str = None, headers: str = None, envs: str = None):
    mcp_server = cliver.config_manager.get_mcp_server(name)
    if not mcp_server:
        cliver.console.print(f"No MCP server found with name: {name}")
        return

    if mcp_server.type == 'stdio':
        if command:
            mcp_server.command = command
        if args:
            mcp_server.args = args.split(",")
        if envs:
            mcp_server.env = json.loads(envs)
    elif mcp_server.type == 'sse':
        if url:
            mcp_server.url = url
        if headers:
            mcp_server.headers = json.loads(headers)
    cliver.config_manager.add_or_update_server(mcp_server)
    cliver.console.print(f"Updated MCP server: {name}")


@mcp.command(name="add", help="Add a MCP server")
@click.option(
    "--type",
    "-t",
    required=True,
    type=click.Choice(['stdio', 'sse']),
    help="Type of the MCP server",
)
@click.option(
    "--name",
    "-n",
    type=str,
    required=True,
    help="Name of the MCP server",
)
@click.option(
    "--command",
    "-c",
    type=str,
    help="Command of the stdio MCP server",
)
@click.option(
    "--args",
    "-a",
    type=str,
    help="Comma separated arguments for the stdio MCP server",
)
@click.option(
    "--envs",
    "-e",
    type=str,
    help="Environment settings in JSON format for stdio MCP server",
)
@click.option(
    "--url",
    "-u",
    type=str,
    help="The URL for the sse MCP server",
)
@click.option(
    "--headers",
    "-h",
    type=str,
    help="HTTP headers in JSON format for sse MCP server",
)
@pass_cliver
def add_mcp_server(cliver: Cliver, name: str, type: str, command: str = None, args: str = None, envs: str = None, url: str = None, headers: str = None):
    mcp_server = cliver.config_manager.get_mcp_server(name)
    if mcp_server:
        cliver.console.print(f"MCP server with name {name} already exists.")
        return
    if type == 'stdio':
        cliver.config_manager.add_or_update_stdio_mcp_server(
            name=name, command=command, args=args.split(","), env=json.loads(envs) if envs else None)
    elif type == 'sse':
        cliver.config_manager.add_or_update_sse_mcp_server(
            name=name, url=url, headers=json.loads(headers) if headers else None)
    else:
        click.echo(f"Unsupported MCP server type: {type}")
    cliver.console.print(f"Added MCP server: {name} of type {type}")


@mcp.command(name="remove", help="Remove a MCP server")
@click.option(
    "--name",
    "-n",
    type=str,
    required=True,
    help="Name of the MCP server",
)
@pass_cliver
def remove_mcp_server(cliver: Cliver, name: str):
    mcp_server = cliver.config_manager.get_mcp_server(name)
    if not mcp_server:
        cliver.console.print(f"No MCP server found with name: {name}")
        return
    cliver.config_manager.remove_mcp_server(name)
    cliver.console.print(f"Removed MCP server: {name}")
