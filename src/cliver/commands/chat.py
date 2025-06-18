import click
from typing import Optional
from cliver.cli import Cliver, pass_cliver

from cliver.llm import TaskExecutor


@click.command(name="chat", help="Chat with LLM models")
@click.option(
    "--model",
    "-m",
    type=str,
    required=False,
    help="Which LLM model to use",
)
@click.argument("query", nargs=-1)
@pass_cliver
def chat(cliver: Cliver, model: Optional[str], query: str):
    """
    Configuration command group.
    This group contains commands to manage configuration settings.
    """
    task_executor = cliver.task_executor
    sentence = " ".join(query)
    _async_chat(task_executor, sentence, model)

async def _async_chat(task_executor: TaskExecutor, user_input: str, model: str):
    response = task_executor.process_user_input_sync(user_input=user_input, model=model)
    click.echo(response)
    if response:
        if isinstance(response, str):
            click.echo(response)
        else:
            if response.content:
                click.echo(response.content)