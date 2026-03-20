"""
Cliver CLI Module

The main entrance of the cliver application
"""

import sys
from pathlib import Path
from shlex import split as shell_split
from typing import Any, Dict

import click
from langchain_core.messages.base import BaseMessage
from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style
from rich.console import Console

from cliver import __version__, commands
from cliver.agent_profile import AgentProfile
from cliver.cli_tool_progress import ThinkingIndicator, create_tool_progress_handler
from cliver.cli_ui import print_banner
from cliver.config import ConfigManager
from cliver.constants import CMD_CHAT
from cliver.llm import TaskExecutor
from cliver.permissions import PermissionManager
from cliver.util import get_config_dir, read_piped_input, stdin_is_piped
from cliver.workflow.workflow_executor import WorkflowExecutor
from cliver.workflow.workflow_manager_local import LocalDirectoryWorkflowManager


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
        self.console = Console()

        # Create agent profile for instance-scoped resources (memory, identity)
        agent_name = self.config_manager.config.agent_name
        self.agent_profile = AgentProfile(agent_name, self.config_dir)

        # Create token tracker for usage auditing
        from cliver.token_tracker import TokenTracker

        self.token_tracker = TokenTracker(
            audit_dir=self.config_dir / "audit_logs",
            agent_name=agent_name,
        )

        # Create permission manager (global + local project settings)
        self.permission_manager = PermissionManager(
            global_config_dir=self.config_dir,
            local_dir=Path.cwd() / ".cliver",
        )

        # Thinking spinner shown while LLM is processing
        self.thinking = ThinkingIndicator(self.console)

        self.task_executor = TaskExecutor(
            llm_models=self.config_manager.list_llm_models(),
            mcp_servers=self.config_manager.list_mcp_servers_for_mcp_caller(),
            default_model=self.config_manager.get_llm_model().name if self.config_manager.get_llm_model() else None,
            user_agent=self.config_manager.config.user_agent,
            agent_name=agent_name,
            on_tool_event=create_tool_progress_handler(self.console, thinking=self.thinking),
            agent_profile=self.agent_profile,
            token_tracker=self.token_tracker,
            permission_manager=self.permission_manager,
            on_permission_prompt=_create_permission_prompt(self.console),
        )

        # Initialize workflow components
        workflow_config = self.config_manager.config.workflow
        workflow_dirs = workflow_config.workflow_dirs if workflow_config else None
        self.workflow_manager = LocalDirectoryWorkflowManager(workflow_dirs)
        self.workflow_executor = WorkflowExecutor(
            task_executor=self.task_executor, workflow_manager=self.workflow_manager
        )

        # prepare console for interaction
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.history_path = self.config_dir / "history"
        self.session = None
        self.piped = stdin_is_piped()
        # Session options that persist across chat commands in interactive mode
        self.session_options = {}
        # Conversation session tracking (interactive mode only)
        self.current_session_id = None
        self.session_history: list[dict] = []  # loaded turns for context
        # LLM-ready conversation history for multi-turn context (BaseMessage objects)
        self.conversation_messages: list[BaseMessage] = []

    def init_session(self, group: click.Group, session_options: Dict[str, Any] = None):
        if self.piped or self.session is not None:
            return
        self._group = group

        def _bottom_toolbar():
            cwd = str(Path.cwd())
            model = self.session_options.get("model") or self.config_manager.config.default_model or "—"
            mode = self.permission_manager.effective_mode.value
            return [
                ("class:toolbar-cwd", f" {cwd} "),
                ("class:toolbar-sep", " │ "),
                ("class:toolbar-mode", f" {mode} "),
                ("class:toolbar-sep", " │ "),
                ("class:toolbar-model", f" ◆ {model} "),
            ]

        self.session = PromptSession(
            history=FileHistory(str(self.history_path)),
            auto_suggest=AutoSuggestFromHistory(),
            completer=WordCompleter(self.load_commands_names(group), ignore_case=True),
            style=Style.from_dict(
                {
                    "prompt": "ansigreen bold",
                    "toolbar-cwd": "bg:#333333 #aaaaaa",
                    "toolbar-sep": "bg:#333333 #555555",
                    "toolbar-mode": "bg:#333333 #88aa88",
                    "toolbar-model": "bg:#333333 #88ccff bold",
                }
            ),
            bottom_toolbar=_bottom_toolbar,
        )
        self.session_options = session_options or {}

    def load_commands_names(self, group: click.Group) -> list[str]:
        return commands.list_commands_names(group)

    def get_session_manager(self):
        """Get the SessionManager for this agent's sessions."""
        from cliver.session_manager import SessionManager

        return SessionManager(self.agent_profile.sessions_dir)

    def record_turn(self, role: str, content: str) -> None:
        """Record a conversation turn to the current session.

        Auto-creates a session on the first turn if none exists.
        Called by chat.py after each user input and LLM response.
        """
        if not content:
            return

        # Auto-create session on first chat in interactive mode
        if not self.current_session_id:
            sm = self.get_session_manager()
            self.current_session_id = sm.create_session()

        sm = self.get_session_manager()
        sm.append_turn(self.current_session_id, role, content)
        self.session_history.append({"role": role, "content": content})

    def _get_commands(self) -> set[str]:
        """Get registered command names for slash-command validation."""
        if hasattr(self, "_group") and self._group:
            return set(self._group.commands.keys())
        return set()

    def run(self) -> None:
        """Run the Cliver client."""
        if self.piped:
            user_data = read_piped_input()
            if user_data is None:
                self.console.print("[bold yellow]No data received from stdin.[/bold yellow]")
            else:
                if user_data.lower() not in ("exit", "quit"):
                    self.call_cmd(user_data)
        else:
            default_model = self.config_manager.config.default_model
            agent_name = self.config_manager.config.agent_name
            print_banner(self.console, agent_name, default_model)

            while True:
                try:
                    # Get user input
                    line = self.session.prompt("Cliver> ").strip()
                    if line.lower() in ("exit", "quit", "/exit", "/quit"):
                        break
                    if line.startswith("/"):
                        if len(line) == 1:
                            continue
                        cmd = line[1:]
                        if cmd.lower().startswith("help"):
                            # /help -> --help, /help llm -> llm --help
                            args = cmd[4:].strip()
                            line = f"{args} --help".strip()
                        else:
                            # Slash prefix = explicit command — validate it exists
                            cmd_name = cmd.split()[0] if cmd.strip() else ""
                            if cmd_name not in self._get_commands():
                                self.console.print(f"[yellow]Unknown command: /{cmd_name}[/yellow]")
                                continue
                            line = cmd
                    elif not line.lower().startswith(f"{CMD_CHAT} "):
                        line = f"{CMD_CHAT} {line}"

                    if len(line.strip()) > 0:
                        self.call_cmd(line)

                except SystemExit:
                    pass
                except KeyboardInterrupt:
                    self.console.print("\n[yellow]Use 'exit' or 'quit' to exit[/yellow]")
                except EOFError:
                    break
                except click.exceptions.UsageError as e:
                    self.console.print(f"{e.format_message()}")
                except Exception as e:
                    self.console.print(f"[red]Error: {e}[/red]")

        # Clean up
        self.cleanup()

    def call_cmd(self, line: str):
        """
        Call a command with the given name and arguments.
        """
        # Parse the command line to get parts
        try:
            parts = shell_split(line)
        except ValueError:
            parts = line.split()
        if not parts or len(parts) == 0:
            return
        if parts[0].lower() == "chat" and len(parts) <= 1:
            # chat command requires at least one more
            return

        cliver(args=parts, prog_name="cliver", standalone_mode=False, obj=self)

    def cleanup(self):
        """
        Clean up the resources opened by the application like the mcp server processes
        or connections to remote mcp servers, llm providers
        """
        self.session_options = {}
        self.session = None
        self.conversation_messages = []


pass_cliver = click.make_pass_decorator(Cliver)


class CliverGroup(click.Group):
    """Custom Click group that treats unrecognized commands as chat queries.

    e.g., `cliver "tell me a joke"` is equivalent to `cliver chat "tell me a joke"`
    """

    def resolve_command(self, ctx, args):
        # If first arg matches a registered command, resolve normally
        cmd_name = args[0] if args else None
        if cmd_name and cmd_name in self.commands:
            return super().resolve_command(ctx, args)

        # Check for flags like --help, --version (start with -)
        if cmd_name and cmd_name.startswith("-"):
            return super().resolve_command(ctx, args)

        # Unrecognized first arg — treat entire args as a chat query
        if args:
            args = ["chat"] + list(args)
            return super().resolve_command(ctx, args)

        return super().resolve_command(ctx, args)


@click.group(cls=CliverGroup, invoke_without_command=True)
@click.version_option(version=__version__, prog_name="cliver")
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
        # If no subcommand is invoked, start interactive mode
        interact(cli)


@cliver.command(name="help", hidden=True)
@click.argument("command", nargs=-1)
@click.pass_context
def help_command(ctx: click.Context, command: tuple):
    """Show help for cliver or a specific command."""
    group = cliver
    # Walk down subcommand chain: cliver help model list → model list --help
    for cmd_name in command:
        sub = group.get_command(ctx, cmd_name)
        if sub is None:
            click.echo(f"No such command '{cmd_name}'.")
            return
        if isinstance(sub, click.Group):
            group = sub
        else:
            click.echo(sub.get_help(ctx))
            return
    click.echo(group.get_help(ctx))


def interact(cli: Cliver, session_options: Dict[str, Any] = None) -> None:
    """
    Start an interactive session with the AI agent.
    """
    if cli.session:
        # already initialized
        return
    cli.init_session(cliver, session_options)
    cli.run()


def loads_commands():
    commands.loads_commands(cliver)
    # Loads extended commands from config dir
    commands.loads_external_commands(cliver)


def _create_permission_prompt(console: Console):
    """Create a Rich-formatted permission prompt callback with arrow-key selection."""
    from prompt_toolkit.formatted_text import HTML
    from prompt_toolkit.shortcuts import radiolist_dialog

    from cliver.permissions import ActionKind, get_tool_meta

    _ACTION_LABELS = {
        ActionKind.READ: ("Read", "blue"),
        ActionKind.WRITE: ("Write", "yellow"),
        ActionKind.EXECUTE: ("Execute", "red"),
        ActionKind.FETCH: ("Fetch", "magenta"),
        ActionKind.SAFE: ("Safe", "green"),
    }

    def prompt(tool_name: str, args: dict) -> str:
        bare = tool_name.split("#")[-1] if "#" in tool_name else tool_name
        meta = get_tool_meta(bare)
        resource = str(args.get(meta.resource_param, "")) if meta.resource_param else ""
        label, color = _ACTION_LABELS.get(meta.action_kind, ("Unknown", "white"))

        # Build the info panel
        console.print()
        console.print("  [bold yellow]⚠  Permission Required[/bold yellow]")
        console.print("  ┌─────────────────────────────────────────────")
        console.print(f"  │  Tool:     [bold]{tool_name}[/bold]")
        console.print(f"  │  Action:   [bold {color}]{label}[/bold {color}]")
        if resource:
            console.print(f"  │  Resource: [cyan]{resource}[/cyan]")
        if event_args_summary := _format_args_summary(args, meta.resource_param):
            console.print(f"  │  Args:     [dim]{event_args_summary}[/dim]")
        console.print("  └─────────────────────────────────────────────")

        try:
            result = radiolist_dialog(
                title=HTML(f"<b>Grant permission for <style fg='ansicyan'>{tool_name}</style>?</b>"),
                text="Use ↑/↓ arrows to select, Enter to confirm:",
                values=[
                    ("allow", "Allow (this time only)"),
                    ("allow_always", "Always allow (this session)"),
                    ("deny", "Deny (this time only)"),
                    ("deny_always", "Always deny (this session)"),
                ],
                default="allow",
            ).run()
            return result if result else "deny"
        except (EOFError, KeyboardInterrupt):
            return "deny"

    return prompt


def _format_args_summary(args: dict, skip_key: str | None = None) -> str:
    """Format tool args into a brief summary, skipping the resource param."""
    if not args:
        return ""
    parts = []
    for k, v in args.items():
        if k == skip_key:
            continue
        val = str(v)
        if len(val) > 50:
            val = val[:47] + "…"
        parts.append(f"{k}={val}")
    return ", ".join(parts[:4])


def cliver_main(*args, **kwargs):
    # loading all click groups and commands before calling it
    loads_commands()
    # bootstrap the cliver application
    cliver()
