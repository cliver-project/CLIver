"""
Cliver CLI Module

The main entrance of the cliver application
"""

from shlex import split as shell_split
import click
import sys
from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.panel import Panel

from cliver.config import ConfigManager
from cliver.llm import TaskExecutor
from cliver import commands
from cliver.util import get_config_dir, stdin_is_piped, read_piped_input
from cliver.constants import *


class Cliver:
    """
    The global App is the box for all capabilities.

    """

    def __init__(self):
        """Initialize the Cliver application."""
        # load config
        self.config_dir = get_config_dir()
        dir_str = str(self.config_dir.absolute())
        if dir_str not in sys.path:
            sys.path.append(dir_str)
        self.config_manager = ConfigManager(self.config_dir)
        self.task_executor = TaskExecutor(
            llm_models=self.config_manager.list_llm_models(),
            mcp_servers=self.config_manager.list_mcp_servers_for_mcp_caller(),
            default_model=self.config_manager.get_llm_model(),
        )
        # prepare console for interaction
        self.history_path = self.config_dir / "history"
        self.session = None
        self.console = Console()
        self.piped = stdin_is_piped()

    def init_session(self, group: click.Group):
        if self.piped or self.session is not None:
            return
        self.session = PromptSession(
            history=FileHistory(str(self.history_path)),
            auto_suggest=AutoSuggestFromHistory(),
            completer=WordCompleter(self.load_commands_names(group), ignore_case=True),
            style=Style.from_dict(
                {
                    "prompt": "ansigreen bold",
                }
            ),
        )

    def load_commands_names(self, group: click.Group) -> list[str]:
        return commands.list_commands_names(group)

    def run(self) -> None:
        """Run the Cliver client."""
        if self.piped:
            user_data = read_piped_input()
            if user_data is None:
                self.console.print(
                    "[bold yellow]No data received from stdin.[/bold yellow]"
                )
            else:
                if not user_data.lower() in ("exit", "quit"):
                    self.call_cmd(user_data)
        else:
            self.console.print(
                Panel.fit(
                    "[bold blue]Cliver[/bold blue] - AI Agent Command Line Interface",
                    border_style="blue",
                )
            )
            self.console.print(
                "Type [bold green]/help[/bold green] to see available commands or start typing to interact with the AI."
            )

            while True:
                try:
                    # Get user input
                    line = self.session.prompt("Cliver> ").strip()
                    if line.lower() in ("exit", "quit", "/exit", "/quit"):
                        break
                    if line.startswith("/") and len(line) > 1:
                        # possibly a command
                        line = line[1:]
                    elif not line.lower().startswith(f"{CMD_CHAT} "):
                        line = f"{CMD_CHAT} {line}"

                    self.call_cmd(line)

                except KeyboardInterrupt:
                    self.console.print(
                        "\n[yellow]Use 'exit' or 'quit' to exit[/yellow]"
                    )
                except EOFError:
                    break
                except click.exceptions.NoArgsIsHelpError as e:
                    self.console.print(f"{e.format_message()}")
                except Exception as e:
                    self.console.print(f"[red]Error: {e}[/red]")

        # Clean up
        self.cleanup()

    def call_cmd(self, line: str):
        """
        Call a command with the given name and arguments.
        """
        parts = shell_split(line)
        cliver(args=parts, prog_name="cliver", standalone_mode=False, obj=self)

    def cleanup(self):
        """
        Clean up the resources opened by the application like the mcp server processes or connections to remote mcp servers, llm providers
        """
        # TODO
        pass


pass_cliver = click.make_pass_decorator(Cliver)


@click.group(invoke_without_command=True)
@click.pass_context
def cliver(ctx: click.Context):
    """
    Cliver: An application aims to make your CLI clever
    """
    cli = None
    if ctx.obj is None:
        cli = Cliver()
        ctx.obj = cli

    if ctx.invoked_subcommand is None:
        # If no subcommand is invoked, show the help message
        _interact(cli)


def _interact(cli: Cliver):
    """
    Start an interactive session with the AI agent.
    """
    cli.init_session(cliver)
    cli.run()


def loads_commands():
    commands.loads_commands(cliver)
    # Loads extended commands from config dir
    commands.loads_external_commands(cliver)


def cliver_main(*args, **kwargs):
    # loading all click groups and commands before calling it
    loads_commands()
    # bootstrap the cliver application
    cliver()
