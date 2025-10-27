import asyncio

import click
from langchain_core.messages import BaseMessage

from cliver.cli import Cliver, pass_cliver
from cliver.mcp_server_caller import MCPServersCaller

@click.group(name="my_command", help="My Command")
@click.pass_context
# here the function name is the same as the file name
def my_command(ctx: click.Context):
    """
    Tasks for My Command
    """
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())
        ctx.exit()


@my_command.command(name="sub_command", help="A sub command")
@click.argument("query", nargs=-1)
@pass_cliver
def sub_command(cliver: Cliver, query: str):
    task_executor = cliver.task_executor
    sentence = " ".join(query)
    response = task_executor.process_user_input_sync(
        user_input=sentence,
        filter_tools=lambda tn, tools: [
            tool for tool in tools if "my_command" in str(tool)
        ],
        enhance_prompt=enhance_prompt,
    )
    if response:
        if isinstance(response, str):
            click.echo(response)
        else:
            if response.content:
                click.echo(response.content)

def enhance_prompt(
    query: str, mcp_caller: MCPServersCaller
) -> list[BaseMessage]:
    return asyncio.run(mcp_caller.get_mcp_prompt(
        "my-mcp-server", "auto_template", {"query": query}
    ))
