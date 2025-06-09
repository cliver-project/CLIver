import click
import json
from cliver.cli import Cliver, pass_cliver
from rich.table import Table
from rich import box


@click.group(name="config", help="Manage configuration settings.")
@click.pass_context
def config(ctx: click.Context):
    """
    Configuration command group.
    This group contains commands to manage configuration settings.
    """
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())
        ctx.exit()


@config.group(name="mcp", help="Manage the MCP Servers")
@click.pass_context
def mcp(ctx: click.Context):
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())
        ctx.exit()


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
        table.add_column("Transport")
        table.add_column("Info", style="blue")
        for name, mcp in mcp_servers.items():
            table.add_row(
                name,
                mcp.get("transport"),
                f"{mcp.get("command") or ''} {mcp.get("args") or ''} {mcp.get("env") or ''}",
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

    if mcp_server.get("transport") == 'stdio':
        if command:
            mcp_server["command"] = command
        if args:
            mcp_server["args"] = args.split(",")
        if envs:
            mcp_server["env"] = json.loads(envs)
    elif mcp_server.get("transport") == 'sse':
        if url:
            mcp_server["url"] = url
        if headers:
            mcp_server["headers"] = json.loads(headers)
    cliver.config_manager.add_or_update_server(mcp_server)
    cliver.console.print(f"Updated MCP server: {name}")


@mcp.command(name="add", help="Add a MCP server")
@click.option(
    "--transport",
    "-t",
    required=True,
    type=click.Choice(['stdio', 'sse']),
    help="Transport of the MCP server",
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
def add_mcp_server(cliver: Cliver, name: str, transport: str, command: str = None, args: str = None, envs: str = None, url: str = None, headers: str = None):
    mcp_server = cliver.config_manager.get_mcp_server(name)
    if mcp_server:
        cliver.console.print(f"MCP server with name {name} already exists.")
        return
    if transport == 'stdio':
        cliver.config_manager.add_or_update_stdio_mcp_server(
            name=name, command=command, args=args.split(","), env=json.loads(envs) if envs else None)
    elif transport == 'sse':
        cliver.config_manager.add_or_update_sse_mcp_server(
            name=name, url=url, headers=json.loads(headers) if headers else None)
    else:
        click.echo(f"Unsupported MCP server transport: {transport}")
    cliver.console.print(f"Added MCP server: {name} of transport {transport}")


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


#
#  Models sub commands
#
@config.group(name="llm", help="Manage the LLM Models")
@click.pass_context
def llm(ctx: click.Context):
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())
        ctx.exit()


@llm.command(name="list", help="List LLM Models")
@pass_cliver
def list_llm_models(cliver: Cliver):
    models = cliver.config_manager.list_llm_models()
    if models:
        table = Table(title="Configured LLM Models", box=box.SQUARE)
        table.add_column("Name", style="green")
        table.add_column("Name In Provider", style="green")
        table.add_column("Provider")
        table.add_column("API Key", style="red")
        table.add_column("Type", style="blue")
        table.add_column("URL")
        table.add_column("Options", style="blue")
        for model in models:
            table.add_row(
                model.name,
                model.name_in_provider,
                model.provider,
                model.api_key,
                model.type,
                model.url,
                model.options.model_dump_json() if model.options else ""
            )
        cliver.console.print(table)
    else:
        cliver.console.print("No LLM Models configured.")


@llm.command(name="remove", help="Remove a LLM Model")
@click.option(
    "--name",
    "-n",
    type=str,
    required=True,
    help="Name of the LLM Model",
)
@pass_cliver
def remove_llm_model(cliver: Cliver, name: str):
    model = cliver.config_manager.get_llm_model(name)
    if not model:
        cliver.console.print(f"No LLM Model found with name: {name}")
        return
    cliver.config_manager.remove_llm_model(name)
    cliver.console.print(f"Removed LLM Model: {name}")


@llm.command(name="add", help="Add a LLM Model")
@click.option(
    "--name",
    "-n",
    type=str,
    required=True,
    help="Name of the LLM Model",
)
@click.option(
    "--provider",
    "-p",
    type=str,
    required=True,
    help="The provider of the LLM Model",
)
@click.option(
    "--api-key",
    "-k",
    type=str,
    help="The api_key of the LLM Model",
)
@click.option(
    "--url",
    "-u",
    type=str,
    required=True,
    help="The url of the LLM Provider service",
)
@click.option(
    "--options",
    "-o",
    type=str,
    help="The options in json format of the LLM Provider",
)
@click.option(
    "--name-in-provider",
    "-N",
    type=str,
    help="The name of the LLM within the Provider",
)
@click.option(
    "--type",
    "-t",
    type=str,
    default="TEXT_TO_TEXT",
    show_default=True,
    help="The type of the LLM Provider",
)
@pass_cliver
def add_llm_model(cliver: Cliver, name: str, provider: str, api_key: str, url: str, options: str, name_in_provider: str, type: str):
    model = cliver.config_manager.get_llm_model(name)
    if model:
        cliver.console.print(
            f"LLM Model found with name: {name} already exists.")
        return
    cliver.config_manager.add_or_update_llm_model(
        name, provider, api_key, url, options, name_in_provider, type)
    cliver.console.print(f"Added LLM Model: {name}")


@llm.command(name="set", help="Update a LLM Model")
@click.option(
    "--name",
    "-n",
    type=str,
    required=True,
    help="Name of the LLM Model",
)
@click.option(
    "--provider",
    "-p",
    type=str,
    help="The provider of the LLM Model",
)
@click.option(
    "--api-key",
    "-k",
    type=str,
    help="The api_key of the LLM Model",
)
@click.option(
    "--url",
    "-u",
    type=str,
    help="The url of the LLM Provider service",
)
@click.option(
    "--options",
    "-o",
    type=str,
    help="The options in json format of the LLM Provider",
)
@click.option(
    "--name-in-provider",
    "-N",
    type=str,
    help="The name of the LLM within the Provider",
)
@click.option(
    "--type",
    "-t",
    type=str,
    help="The type of the LLM Provider",
)
@pass_cliver
def update_llm_model(cliver: Cliver, name: str, provider: str, api_key: str, url: str, options: str, name_in_provider: str, type: str = "TEXT_TO_TEXT"):
    model = cliver.config_manager.get_llm_model(name)
    if not model:
        cliver.console.print(
            f"LLM Model with name: {name} was not found.")
        return
    cliver.config_manager.add_or_update_llm_model(
        name, provider, api_key, url, options, name_in_provider, type)
    cliver.console.print(f"Added LLM Model: {name}")
