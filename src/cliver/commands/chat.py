import click
from typing import Optional
from cliver.cli import Cliver, pass_cliver
import asyncio

from cliver.llm import TaskExecutor


@click.command(name="chat", help="Chat with LLM models")
@click.option(
    "--model",
    "-m",
    type=str,
    required=False,
    help="Which LLM model to use",
)
@click.argument("query")
@pass_cliver
def chat(cliver: Cliver, model: Optional[str], query: str):
    """
    Configuration command group.
    This group contains commands to manage configuration settings.
    """
    task_executor = cliver.task_executor
    asyncio.run(_async_chat(task_executor, query, model))

async def _async_chat(task_executor: TaskExecutor, user_input: str, model: str):
    response = await task_executor.process_user_input(user_input=user_input, model=model)
    click.echo(response)
    if response:
        if isinstance(response, str):
            click.echo(response)
        else:
            if response.content:
                click.echo(response.content)