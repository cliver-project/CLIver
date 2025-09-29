import click
import asyncio
import time
from typing import Optional, List
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
@click.option(
    "--stream",
    "-s",
    is_flag=True,
    default=False,
    help="Stream the response",
)
@click.option(
    "--skill-set",
    "-ss",
    multiple=True,
    help="Skill sets to apply to the chat session",
)
@click.option(
    "--template",
    "-t",
    type=str,
    help="Template to use for the prompt",
)
@click.option(
    "--param",
    "-p",
    multiple=True,
    type=(str, str),
    help="Parameters for skill sets and templates (key=value)",
)
@click.argument("query", nargs=-1)
@pass_cliver
def chat(cliver: Cliver, model: Optional[str], stream: bool, skill_set: List[str], template: Optional[str], param: List[tuple], query: str):
    """
    Configuration command group.
    This group contains commands to manage configuration settings.
    """
    task_executor = cliver.task_executor
    sentence = " ".join(query)
    # Convert param tuples to dictionary
    params = dict(param)
    _async_chat(task_executor, sentence, model, stream, skill_set, template, params)


def _async_chat(
    task_executor: TaskExecutor,
    user_input: str,
    model: str,
    stream: bool = False,
    skill_sets: List[str] = None,
    template: Optional[str] = None,
    params: dict = None
):
    if stream:
        # For streaming, we need to run the async generator
        asyncio.run(_stream_chat(task_executor, user_input, model, skill_sets, template, params))
    else:
        response = task_executor.process_user_input_sync(
            user_input=user_input, model=model, skill_sets=skill_sets, template=template, params=params
        )
        if response:
            if isinstance(response, str):
                click.echo(response)
            else:
                if response.content:
                    click.echo(response.content)


async def _stream_chat(task_executor: TaskExecutor, user_input: str, model: str, skill_sets: List[str] = None, template: Optional[str] = None, params: dict = None):
    """Stream the chat response character by character."""
    try:
        async for chunk in task_executor.stream_user_input(
            user_input=user_input, model=model, skill_sets=skill_sets, template=template, params=params
        ):
            if hasattr(chunk, "content") and chunk.content:
                # Print each character with a small delay to simulate streaming
                import sys

                for char in chunk.content:
                    sys.stdout.write(char)
                    sys.stdout.flush()
                    time.sleep(0.01)
        print()  # New line at the end
    except Exception as e:
        click.echo(f"Error: {e}")
