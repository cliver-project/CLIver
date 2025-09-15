import click
from langchain_core.messages import BaseMessage

from cliver.cli import Cliver, pass_cliver
from cliver.mcp_server_caller import MCPServersCaller


@click.group(name="hypershift", help="Hypershift related commands")
@click.pass_context
def hypershift(ctx: click.Context):
    """
    Tasks for hypershift related
    """
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())
        ctx.exit()


@hypershift.command(name="auto", help="Generate Auto testcase codes for Hypershift")
@click.argument("query", nargs=-1)
@pass_cliver
def auto(cliver: Cliver, query: str):
    task_executor = cliver.task_executor
    sentence = " ".join(query)
    response = task_executor.process_user_input_sync(
        user_input=sentence,
        filter_tools=lambda tn, tools: [
            tool for tool in tools if "hypershift" in str(tool)
        ],
        enhance_prompt=enhance_prompt_hypershift_auto,
    )
    if response:
        if isinstance(response, str):
            click.echo(response)
        else:
            if response.content:
                click.echo(response.content)


# This requires a MCP server called hypershift which provides a prompt template called 'auto_template'
async def enhance_prompt_hypershift_auto(
    query: str, mcp_caller: MCPServersCaller
) -> list[BaseMessage]:
    # a prompt to assistant on auto case codes generation for hypershift
    return await mcp_caller.get_mcp_prompt(
        "hypershift", "auto_template", {"query": query}
    )
