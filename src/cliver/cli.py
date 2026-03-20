"""
Cliver CLI Module

The main entrance of the cliver application
"""

import asyncio
import shutil
import sys
from pathlib import Path
from shlex import split as shell_split
from typing import Any, Dict

import click
from langchain_core.messages.base import BaseMessage
from prompt_toolkit import Application
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import FormattedTextControl, HSplit, Layout, Window
from prompt_toolkit.layout.controls import BufferControl
from prompt_toolkit.layout.processors import BeforeInput
from prompt_toolkit.patch_stdout import patch_stdout
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
        self._app: Application | None = None
        self.piped = stdin_is_piped()
        # Session options that persist across chat commands in interactive mode
        self.session_options = {}
        # Conversation session tracking (interactive mode only)
        self.current_session_id = None
        self.session_history: list[dict] = []  # loaded turns for context
        # LLM-ready conversation history for multi-turn context (BaseMessage objects)
        self.conversation_messages: list[BaseMessage] = []
        # RSS feed state for scrolling headlines
        self._rss_headlines: list[str] = []
        self._rss_index = 0

    def init_session(self, group: click.Group, session_options: Dict[str, Any] = None):
        if self.piped:
            return
        self._group = group
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

    # ─── Status Bar ──────────────────────────────────────────────────────────

    def _statusbar_content(self):
        """Build status bar content parts."""
        try:
            from wcwidth import wcswidth

            def _dw(s):
                w = wcswidth(s)
                return w if w >= 0 else len(s)
        except ImportError:
            _dw = len

        cwd = str(Path.cwd())
        model = self.session_options.get("model") or self.config_manager.config.default_model or "—"
        mode = self.permission_manager._effective_mode().value
        tw = shutil.get_terminal_size().columns

        left = f" 📂 {cwd} "
        right_mode = f" 🔒 {mode} "
        right_model = f" ◆ {model} "

        middle_text = ""
        if self._rss_headlines:
            headline = self._rss_headlines[self._rss_index % len(self._rss_headlines)]
            self._rss_index += 1
            middle_text = f" {headline} "

        fixed = _dw(left) + _dw(middle_text) + _dw(right_mode) + _dw(right_model) + 5
        padding = max(0, tw - fixed)

        if middle_text:
            pad_left = padding // 2
            pad_right = padding - pad_left
            middle = f"{' ' * pad_left}{middle_text}{' ' * pad_right}"
        else:
            middle = " " * padding

        return tw, left, middle, right_mode, right_model

    def _get_toolbar_parts(self) -> FormattedText:
        """Build the toolbar formatted text for prompt_toolkit."""
        sb = self.session_options.get("statusbar", {})
        if not sb.get("visible", True):
            return FormattedText([])

        tw, left, middle, right_mode, right_model = self._statusbar_content()

        mid_line = "├" + "─" * (tw - 2) + "┤"
        bot_line = "╰" + "─" * (tw - 2) + "╯"

        return FormattedText(
            [
                ("class:toolbar-border", mid_line),
                ("class:toolbar-border", "\n│"),
                ("class:toolbar-cwd", left),
                ("class:toolbar-border", "│"),
                ("class:toolbar-rss", middle),
                ("class:toolbar-border", "│"),
                ("class:toolbar-mode", right_mode),
                ("class:toolbar-border", "│"),
                ("class:toolbar-model", right_model),
                ("class:toolbar-border", "│"),
                ("class:toolbar-border", f"\n{bot_line}"),
            ]
        )

    def _get_prompt_prefix(self) -> FormattedText:
        """Build the input line top border."""
        tw = shutil.get_terminal_size().columns
        label = " Cliver "
        top = "╭─" + label + "─" * max(0, tw - len(label) - 3) + "╮"
        return FormattedText(
            [
                ("class:toolbar-border", top),
                ("class:toolbar-border", "\n│ "),
            ]
        )

    # ─── Command Processing ──────────────────────────────────────────────────

    def _preprocess_line(self, line: str) -> str | None:
        """Preprocess input line: handle slash commands, validation, routing."""
        if line.lower() in ("exit", "quit", "/exit", "/quit"):
            return None  # signal to exit

        if line.startswith("/"):
            if len(line) == 1:
                return None
            cmd = line[1:]
            if cmd.lower().startswith("help"):
                args = cmd[4:].strip()
                return f"{args} --help".strip()
            else:
                cmd_name = cmd.split()[0] if cmd.strip() else ""
                if cmd_name not in self._get_commands():
                    print(f"Unknown command: /{cmd_name}")
                    return None
                return cmd
        elif not line.lower().startswith(f"{CMD_CHAT} "):
            return f"{CMD_CHAT} {line}"

        return line

    def call_cmd(self, line: str):
        """Call a command with the given name and arguments."""
        try:
            parts = shell_split(line)
        except ValueError:
            parts = line.split()
        if not parts or len(parts) == 0:
            return
        if parts[0].lower() == "chat" and len(parts) <= 1:
            return

        cliver_cli(args=parts, prog_name="cliver", standalone_mode=False, obj=self)

    def cleanup(self):
        """Clean up resources."""
        self.session_options = {}
        self.session = None
        self._app = None
        self.conversation_messages = []

    # ─── Run Methods ─────────────────────────────────────────────────────────

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
            self._run_tui()

    def _run_tui(self) -> None:
        """Run the persistent TUI with input+toolbar always at the bottom."""
        # Print banner to real stdout before TUI takes over
        default_model = self.config_manager.config.default_model
        agent_name = self.config_manager.config.agent_name
        print_banner(self.console, agent_name, default_model)

        # Input buffer with history and completion
        history = FileHistory(str(self.history_path))
        completer = WordCompleter(
            self.load_commands_names(self._group) if hasattr(self, "_group") else [],
            ignore_case=True,
        )

        def on_accept(buff):
            line = buff.text.strip()
            buff.reset()
            if not line:
                return
            processed = self._preprocess_line(line)
            if processed is None:
                if line.lower() in ("exit", "quit", "/exit", "/quit"):
                    app.exit()
                return
            app.create_background_task(_run_command(processed))

        async def _run_command(line):
            loop = asyncio.get_event_loop()
            try:
                await loop.run_in_executor(None, lambda: self.call_cmd(line))
            except SystemExit:
                pass
            except Exception as e:
                print(f"Error: {e}")

        input_buffer = Buffer(
            accept_handler=on_accept,
            multiline=False,
            history=history,
            auto_suggest=AutoSuggestFromHistory(),
            completer=completer,
        )

        # Key bindings
        kb = KeyBindings()

        @kb.add("c-c")
        def _ctrl_c(event):
            """Ctrl+C exits."""
            event.app.exit()

        @kb.add("c-g")
        def _open_editor(event):
            """Ctrl+G opens $EDITOR for extended input."""
            event.current_buffer.open_in_editor()

        # Layout: input line (with top border) + toolbar (with borders)
        input_window = Window(
            content=BufferControl(
                buffer=input_buffer,
                input_processors=[BeforeInput(self._get_prompt_prefix)],
            ),
            height=2,  # top border line + input line
            dont_extend_height=True,
        )

        toolbar_window = Window(
            content=FormattedTextControl(self._get_toolbar_parts),
            height=3,  # top border + content + bottom border
            dont_extend_height=True,
        )

        layout = Layout(
            HSplit(
                [
                    input_window,
                    toolbar_window,
                ]
            ),
            focused_element=input_window,
        )

        from prompt_toolkit.styles import Style

        style = Style.from_dict(
            {
                "prompt": "ansigreen bold",
                "toolbar-border": "#555555",
                "toolbar-cwd": "#aaaaaa",
                "toolbar-mode": "#88aa88",
                "toolbar-rss": "#cccc88 italic",
                "toolbar-model": "#88ccff bold",
            }
        )

        app = Application(
            layout=layout,
            key_bindings=kb,
            style=style,
            full_screen=False,
        )
        self._app = app

        with patch_stdout(raw=True):
            app.run()

        # Clean up
        self.cleanup()


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
def cliver_cli(ctx: click.Context):
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


@cliver_cli.command(name="help", hidden=True)
@click.argument("command", nargs=-1)
@click.pass_context
def help_command(ctx: click.Context, command: tuple):
    """Show help for cliver or a specific command."""
    group = cliver_cli
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
    """Start an interactive session with the AI agent."""
    cli.init_session(cliver_cli, session_options)
    cli.run()


def loads_commands():
    commands.loads_commands(cliver_cli)
    # Loads extended commands from config dir
    commands.loads_external_commands(cliver_cli)


def _create_permission_prompt(console: Console):
    """Create a Rich-formatted permission prompt callback."""
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

        # Build the info panel (printed via stdout → appears above TUI via patch_stdout)
        print()
        print("  ⚠  Permission Required")
        print("  ┌─────────────────────────────────────────────")
        print(f"  │  Tool:     {tool_name}")
        print(f"  │  Action:   {label}")
        if resource:
            print(f"  │  Resource: {resource}")
        if event_args_summary := _format_args_summary(args, meta.resource_param):
            print(f"  │  Args:     {event_args_summary}")
        print("  └─────────────────────────────────────────────")

        # Simple text prompt (works in background thread with patch_stdout)
        while True:
            try:
                print("  [y]es / [n]o / [a]lways allow / [d]eny always")
                response = input("  > ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                return "deny"
            if response in ("y", "yes"):
                return "allow"
            elif response in ("a", "always"):
                return "allow_always"
            elif response in ("n", "no"):
                return "deny"
            elif response in ("d", "deny"):
                return "deny_always"

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
    cliver_cli()
