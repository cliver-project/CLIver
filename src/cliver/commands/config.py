import click
from cliver.cli import Cliver, pass_cliver


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
    for mcp in cliver.config_manager.list_mcp_servers():
        click.echo(f"mcp server: {mcp.name}")
