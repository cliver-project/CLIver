import json
import click
from rich import box
from rich.table import Table

from cliver.cli import Cliver, pass_cliver
from cliver.util import parse_key_value_options
from cliver.config import (
    SSEMCPServerConfig,
    StdioMCPServerConfig,
    StreamableHttpMCPServerConfig,
    WebSocketMCPServerConfig,
)
from cliver.model_capabilities import ProviderEnum


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


# noinspection PyUnresolvedReferences
@mcp.command(name="list", help="List MCP servers")
@pass_cliver
def list_mcp_servers(cliver: Cliver):
    mcp_servers = cliver.config_manager.list_mcp_servers()
    if mcp_servers:
        table = Table(title="Configured MCP Servers", box=box.SQUARE)
        table.add_column("Name", style="green")
        table.add_column("Transport")
        table.add_column("Info", style="blue")
        for name, _mcp_server in mcp_servers.items():
            if isinstance(_mcp_server, StdioMCPServerConfig):
                info = f"{_mcp_server.command} {_mcp_server.args or ''} {_mcp_server.env or ''}"
            elif isinstance(_mcp_server, SSEMCPServerConfig) or isinstance(_mcp_server, StreamableHttpMCPServerConfig) or isinstance(_mcp_server, WebSocketMCPServerConfig):
                info = f"{_mcp_server.url} {_mcp_server.headers or ''}"
            else:
                info = "Unknown server type"
            table.add_row(
                name,
                _mcp_server.transport,
                info,
            )
        cliver.console.print(table)
    else:
        cliver.console.print("No MCP servers configured.")


# noinspection PyUnresolvedReferences
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
    "--env",
    "-e",
    multiple=True,
    type=str,
    help="Environment variables in key=value format (can be specified multiple times)",
)
@click.option(
    "--url",
    "-u",
    type=str,
    help="The URL for the sse MCP server",
)
@click.option(
    "--header",
    "-H",
    multiple=True,
    type=str,
    help="HTTP headers in key=value format (can be specified multiple times)",
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
    mcp_server = cliver.config_manager.get_mcp_server(name)
    if not mcp_server:
        cliver.console.print(f"No MCP server found with name: {name}")
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
    elif isinstance(mcp_server, SSEMCPServerConfig) or isinstance(mcp_server, StreamableHttpMCPServerConfig) or isinstance(mcp_server, WebSocketMCPServerConfig):
        if url:
            mcp_server.url = url
        if header:
            # Parse headers from key=value format
            header_dict = parse_key_value_options(header, cliver.console)
            mcp_server.headers = header_dict
    cliver.config_manager.add_or_update_server(name, mcp_server)
    cliver.console.print(f"Updated MCP server: {name}")


# noinspection PyUnresolvedReferences
@mcp.command(name="add", help="Add a MCP server")
@click.option(
    "--transport",
    "-t",
    type=click.Choice(["stdio", "sse", "streamable", "websocket"]),
    default="stdio",
    help="Transport of the MCP server (sse is deprecated, use streamable instead)",
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
    "--env",
    "-e",
    multiple=True,
    type=str,
    help="Environment variables in key=value format (can be specified multiple times)",
)
@click.option(
    "--url",
    "-u",
    type=str,
    help="The URL for the sse MCP server",
)
@click.option(
    "--header",
    "-H",
    multiple=True,
    type=str,
    help="HTTP headers in key=value format (can be specified multiple times)",
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
    mcp_server = cliver.config_manager.get_mcp_server(name)
    if mcp_server:
        cliver.console.print(f"MCP server with name {name} already exists.")
        return
    if transport == "stdio":
        if command is None:
            cliver.console.print("Command is required for stdio transport")
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
        cliver.console.print(
            "Warning: SSE transport is deprecated, consider using streamable instead"
        )
        if url is None:
            cliver.console.print("URL is required for sse transport")
            return
        # Parse headers from key=value format
        header_dict = parse_key_value_options(header, cliver.console)
        cliver.config_manager.add_or_update_sse_mcp_server(
            name=name, url=url, headers=header_dict
        )
    elif transport == "streamable":
        if url is None:
            cliver.console.print("URL is required for streamable transport")
            return
        # Parse headers from key=value format
        header_dict = parse_key_value_options(header, cliver.console)
        cliver.config_manager.add_or_update_streamable_mcp_server(
            name=name, url=url, headers=header_dict
        )
    elif transport == "websocket":
        if url is None:
            cliver.console.print("URL is required for websocket transport")
            return
        # Parse headers from key=value format
        header_dict = parse_key_value_options(header, cliver.console)
        cliver.config_manager.add_or_update_websocket_mcp_server(
            name=name, url=url, headers=header_dict
        )
    else:
        click.echo(f"Unsupported MCP server transport: {transport}")
    cliver.console.print(f"Added MCP server: {name} of transport {transport}")


# noinspection PyUnresolvedReferences
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


# noinspection PyUnresolvedReferences
@llm.command(name="list", help="List LLM Models")
@pass_cliver
def list_llm_models(cliver: Cliver):
    models = cliver.config_manager.list_llm_models()
    if models:
        table = Table(title="Configured LLM Models", box=box.SQUARE)
        table.add_column("Name", style="green")
        table.add_column("Name In Provider", style="green")
        table.add_column("Provider")
        table.add_column("URL")
        table.add_column("Capabilities", style="blue")
        table.add_column("File Upload", style="yellow")
        for _, model in models.items():
            # Get model capabilities
            capabilities = model.get_capabilities()

            # Format capabilities as a comma-separated string
            capabilities_str = (
                ", ".join([cap.value for cap in capabilities])
                if capabilities
                else "N/A"
            )

            # Check if file upload is supported
            from cliver.model_capabilities import ModelCapability

            file_upload_supported = ModelCapability.FILE_UPLOAD in capabilities

            table.add_row(
                model.name,
                model.name_in_provider,
                model.provider,
                model.url,
                capabilities_str,
                "Yes" if file_upload_supported else "No",
            )
        cliver.console.print(table)
    else:
        cliver.console.print("No LLM Models configured.")


# noinspection PyUnresolvedReferences
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


# noinspection PyUnresolvedReferences
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
    type=click.Choice([p.value for p in ProviderEnum]),
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
    "--option",
    "-o",
    multiple=True,
    type=str,
    help="Model options in key=value format (can be specified multiple times)",
)
@click.option(
    "--name-in-provider",
    "-N",
    type=str,
    help="The name of the LLM within the Provider",
)
@click.option(
    "--capabilities",
    "-c",
    type=str,
    help="Comma-separated list of model capabilities (e.g., text_to_text,image_to_text,tool_calling)",
)
@pass_cliver
def add_llm_model(
    cliver: Cliver,
    name: str,
    provider: str,
    api_key: str,
    url: str,
    option: tuple,
    name_in_provider: str,
    capabilities: str,
):
    model = cliver.config_manager.get_llm_model(name)
    if model:
        cliver.console.print(f"LLM Model found with name: {name} already exists.")
        return

    # Convert key=value options to JSON string
    options_json = None
    if option:
        options_dict = parse_key_value_options(option, cliver.console)
        options_json = json.dumps(options_dict)

    cliver.config_manager.add_or_update_llm_model(
        name, provider, api_key, url, options_json, name_in_provider, capabilities
    )
    cliver.console.print(f"Added LLM Model: {name}")


# noinspection PyUnresolvedReferences
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
    type=click.Choice([p.value for p in ProviderEnum]),
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
    "--option",
    "-o",
    multiple=True,
    type=str,
    help="Model options in key=value format (can be specified multiple times)",
)
@click.option(
    "--name-in-provider",
    "-N",
    type=str,
    help="The name of the LLM within the Provider",
)
@click.option(
    "--capabilities",
    "-c",
    type=str,
    help="Comma-separated list of model capabilities (e.g., text_to_text,image_to_text,tool_calling)",
)
@pass_cliver
def update_llm_model(
    cliver: Cliver,
    name: str,
    provider: str,
    api_key: str,
    url: str,
    option: tuple,
    name_in_provider: str,
    capabilities: str,
):
    model = cliver.config_manager.get_llm_model(name)
    if not model:
        cliver.console.print(f"LLM Model with name: {name} was not found.")
        return

    # Convert key=value options to JSON string
    options_json = None
    if option:
        options_dict = parse_key_value_options(option, cliver.console)
        options_json = json.dumps(options_dict)

    cliver.config_manager.add_or_update_llm_model(
        name, provider, api_key, url, options_json, name_in_provider, capabilities
    )
    cliver.console.print(f"LLM Model: {name} updated")
