"""
Cliver CLI Module

The main entrance of the cliver application
"""

from shlex import split as shell_split
import click
from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.panel import Panel

from cliver.config import ConfigManager
from cliver.core import Core
from cliver.commands import loads_commands, list_commands_names
from cliver.util import get_config_dir, stdin_is_piped, read_piped_input
from cliver.constants import *


console = Console()


class Cliver:
    """
    The global App is the box for all capabilities.

    """

    def __init__(self):
        """Initialize the Cliver application.
        """
        # load config
        self.config_dir = get_config_dir()
        self.config_manager = ConfigManager(self.config_dir)
        self.core = Core(self.config_manager)
        # TODO loads LLM functions

        # prepare console for interaction
        self.history_path = self.config_dir / "history"
        self.session = None
        self.piped = stdin_is_piped()

    def init_session(self, group: click.Group):
        if self.piped or self.session is not None:
            return
        # Set up prompt session with history
        self.session = PromptSession(
            history=FileHistory(str(self.history_path)),
            auto_suggest=AutoSuggestFromHistory(),
            completer=WordCompleter(
                self.load_commands_names(group), ignore_case=True),
            style=Style.from_dict(
                {
                    "prompt": "ansigreen bold",
                }
            ),
        )

    def load_commands_names(self, group: click.Group) -> list[str]:
        return list_commands_names(group)

    def run(self) -> None:
        """Run the Cliver client."""
        if self.piped:
            user_data = read_piped_input()
            if user_data is None:
                console.print(
                    "[bold yellow]No data received from stdin.[/bold yellow]")
            else:
                if not user_data.lower() in ("exit", "quit"):
                    self.call_cmd(user_data)
        else:
            console.print(
                Panel.fit(
                    "[bold blue]Cliver[/bold blue] - AI Agent Command Line Interface",
                    border_style="blue",
                )
            )
            console.print(
                "Type [bold green]/help[/bold green] to see available commands or start typing to interact with the AI."
            )

            while True:
                try:
                    # Get user input
                    line = self.session.prompt("Cliver> ").strip()
                    if line.lower() in ("exit", "quit"):
                        break
                    if line.startswith("/") and len(line) > 1:
                        # possibly a command
                        line = line[1:]
                    elif not line.lower().startswith(f"{CMD_CHAT} "):
                        line = f"{CMD_CHAT} {line}"

                    self.call_cmd(line)

                except KeyboardInterrupt:
                    console.print(
                        "\n[yellow]Use 'exit' or 'quit' to exit[/yellow]")
                except EOFError:
                    break
                except Exception as e:
                    console.print(f"[red]Error: {e}[/red]")

        # Clean up
        self.cleanup()

    def call_cmd(self, line: str):
        """
        Call a command with the given name and arguments.
        """
        parts = shell_split(line)
        cliver.main(args=parts,
                    prog_name="cliver",
                    standalone_mode=False,
                    obj=self
                    )

    def cleanup(self):
        """
        Clean up the resources opened by the application like the mcp server processes or connections to remote mcp servers, llm providers
        """
        # TODO
        pass


pass_cliver = click.make_pass_decorator(Cliver)

cli = Cliver()


@click.group(invoke_without_command=True)
@click.pass_context
def cliver(ctx: click.Context):
    """
    Cliver: An application aims to make your CLI clever
    """
    if ctx.obj is None:
        ctx.obj = cli

    if ctx.invoked_subcommand is None:
        # If no subcommand is invoked, show the help message
        _interact()


def _interact():
    """
    Start an interactive session with the AI agent.
    """
    cli.init_session(cliver)
    cli.run()


def main():
    # loading all click groups and commands before calling it
    loads_commands(cliver)
    # bootstrap the cliver application
    cliver()
