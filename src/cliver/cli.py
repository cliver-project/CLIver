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


class _IndentedStdout:
    """Wraps stdout to add left margin, while preserving fileno() for Rich width detection."""

    INDENT = "   "

    def __init__(self, real_stdout):
        self._real = real_stdout
        self._at_line_start = True

    def write(self, text):
        if not text:
            return 0
        out = []
        for ch in text:
            if self._at_line_start and ch != "\n":
                out.append(self.INDENT)
            out.append(ch)
            self._at_line_start = ch in ("\n", "\r")
        result = "".join(out)
        return self._real.write(result)

    def flush(self):
        self._real.flush()

    def fileno(self):
        return self._real.fileno()

    @property
    def encoding(self):
        return self._real.encoding

    def isatty(self):
        return self._real.isatty()

    def __getattr__(self, name):
        return getattr(self._real, name)


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

        # Create permission manager (global + local project settings)
        self.permission_manager = PermissionManager(
            global_config_dir=self.config_dir,
            local_dir=Path.cwd() / ".cliver",
        )

        # Thinking spinner shown while LLM is processing
        self.thinking = ThinkingIndicator(self.console)

        # Initialize agent (may be overridden by --agent flag before run)
        self._init_agent(self.config_manager.config.default_agent_name)

        # Initialize workflow components — must be after _init_agent sets task_executor
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
        self._cancel_requested = False

    def init_session(self, group: click.Group, session_options: Dict[str, Any] = None):
        if self.piped:
            return
        self._group = group
        self.session_options = session_options or {}

    def output(self, *args, **kwargs) -> None:
        """Centralized output method. All CLI display should go through this."""
        self.console.print(*args, **kwargs)

    def _init_agent(self, agent_name: str) -> None:
        """Initialize (or reinitialize) all agent-scoped components.

        Ensures the agent directory and a default identity.md exist.
        """
        from cliver.token_tracker import TokenTracker

        self.agent_profile = AgentProfile(agent_name, self.config_dir)
        self.agent_profile.ensure_dirs()

        # Create a default identity file if it doesn't exist yet
        if not self.agent_profile.identity_file.exists():
            self.agent_profile.save_identity(
                f"# Agent: {agent_name}\n\n"
                f"## Agent Persona\n"
                f"- Name: {agent_name}\n"
                f"- Style: Helpful, professional\n\n"
                f"## User Profile\n"
                f"- *(Update via `/identity chat` or `/agent create`)*\n"
            )

        self.token_tracker = TokenTracker(
            audit_dir=self.config_dir / "audit_logs",
            agent_name=agent_name,
        )
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

    def switch_agent(self, agent_name: str) -> None:
        """Switch to a different agent, updating all scoped resources."""
        self._init_agent(agent_name)
        # Update workflow executor to use the new task executor
        self.workflow_executor = WorkflowExecutor(
            task_executor=self.task_executor, workflow_manager=self.workflow_manager
        )
        # Clear conversation state (different agent = fresh context)
        self.conversation_messages = []
        self.session_history = []
        self.current_session_id = None
        # Persist as the new default
        self.config_manager.set_default_agent_name(agent_name)

    @property
    def agent_name(self) -> str:
        """Current active agent name."""
        return self.agent_profile.agent_name

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

        agent = self.agent_name
        left = f" 📂 {cwd} "
        right_agent = f" 🤖 {agent} "
        right_mode = f" 🔒 {mode} "
        right_model = f" ◆ {model} "

        middle_text = ""
        if self._rss_headlines:
            headline = self._rss_headlines[self._rss_index % len(self._rss_headlines)]
            self._rss_index += 1
            middle_text = f" {headline} "

        fixed = _dw(left) + _dw(middle_text) + _dw(right_agent) + _dw(right_mode) + _dw(right_model) + 6
        padding = max(0, tw - fixed)

        if middle_text:
            pad_left = padding // 2
            pad_right = padding - pad_left
            middle = f"{' ' * pad_left}{middle_text}{' ' * pad_right}"
        else:
            middle = " " * padding

        return tw, left, middle, right_agent, right_mode, right_model

    def _get_toolbar_parts(self) -> FormattedText:
        """Build the toolbar formatted text for prompt_toolkit."""
        sb = self.session_options.get("statusbar", {})
        if not sb.get("visible", True):
            return FormattedText([])

        tw, left, middle, right_agent, right_mode, right_model = self._statusbar_content()

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
                ("class:toolbar-agent", right_agent),
                ("class:toolbar-border", "│"),
                ("class:toolbar-mode", right_mode),
                ("class:toolbar-border", "│"),
                ("class:toolbar-model", right_model),
                ("class:toolbar-border", "│"),
                ("class:toolbar-border", f"\n{bot_line}"),
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
                return f"help {args}".strip() if args else "help"
            else:
                cmd_parts = cmd.split()
                cmd_name = cmd_parts[0] if cmd_parts else ""
                if cmd_name not in self._get_commands():
                    self.console.print(f"[yellow]Unknown command: /{cmd_name}[/yellow]")
                    return None
                # Convert `/cmd --help` to `help cmd`
                if "--help" in cmd_parts:
                    cmd_parts.remove("--help")
                    return f"help {' '.join(cmd_parts)}"
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
        agent_name = self.agent_name
        print_banner(self.console, agent_name, default_model)

        # Input buffer with history and completion
        history = FileHistory(str(self.history_path))
        completer = WordCompleter(
            self.load_commands_names(self._group) if hasattr(self, "_group") else [],
            ignore_case=True,
        )

        _current_task = {"task": None}

        def on_accept(buff):
            line = buff.text.strip()
            if line:
                history.append_string(line)
            buff.reset()
            if not line:
                return
            processed = self._preprocess_line(line)
            if processed is None:
                if line.lower() in ("exit", "quit", "/exit", "/quit"):
                    # Cancel any running task before exiting
                    if _current_task["task"] and not _current_task["task"].done():
                        _current_task["task"].cancel()
                        self._cancel_requested = True
                    app.exit()
                return
            # Echo the user's input to the output area
            self.output(f"\n[bold green]❯[/bold green] {line}")
            self._cancel_requested = False
            _current_task["task"] = app.create_background_task(_run_command(processed))

        async def _run_command(line):
            loop = asyncio.get_event_loop()
            try:
                await loop.run_in_executor(None, lambda: self.call_cmd(line))
            except (SystemExit, asyncio.CancelledError):
                pass
            except Exception as e:
                self.console.print(f"[red]Error: {e}[/red]")
            finally:
                _current_task["task"] = None

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
            """Ctrl+C cancels running task, or exits if idle."""
            if _current_task["task"] and not _current_task["task"].done():
                _current_task["task"].cancel()
                self._cancel_requested = True
                self.console.print("\n[yellow]Cancelled.[/yellow]")
            else:
                event.app.exit()

        @kb.add("c-g")
        def _open_editor(event):
            """Ctrl+G opens $EDITOR for extended input."""
            event.current_buffer.open_in_editor()

        _last_esc = {"t": 0.0}

        @kb.add("escape")
        def _escape(event):
            """Double-ESC clears the input box."""
            import time

            now = time.monotonic()
            if now - _last_esc["t"] < 0.5:
                event.current_buffer.reset()
                _last_esc["t"] = 0.0
            else:
                _last_esc["t"] = now

        # Layout: spacer + top border + input line + toolbar
        def _input_top_border():
            tw = shutil.get_terminal_size().columns
            label = " Cliver "
            line = "╭─" + label + "─" * max(0, tw - len(label) - 3) + "╮"
            return FormattedText([("class:toolbar-border", line)])

        border_window = Window(
            content=FormattedTextControl(_input_top_border),
            height=1,
            dont_extend_height=True,
        )

        from prompt_toolkit.layout.margins import Margin

        class LeftBorderMargin(Margin):
            def get_width(self, get_ui_content):
                return 2

            def create_margin(self, window_render_info, width, height):
                return [("class:toolbar-border", "│ \n")] * height

        class RightBorderMargin(Margin):
            def get_width(self, get_ui_content):
                return 2

            def create_margin(self, window_render_info, width, height):
                return [("class:toolbar-border", " │\n")] * height

        input_window = Window(
            content=BufferControl(
                buffer=input_buffer,
                input_processors=[BeforeInput([("class:prompt", "❯ ")])],
            ),
            wrap_lines=True,
            dont_extend_height=True,
            left_margins=[LeftBorderMargin()],
            right_margins=[RightBorderMargin()],
        )

        toolbar_window = Window(
            content=FormattedTextControl(self._get_toolbar_parts),
            height=3,  # top border + content + bottom border
            dont_extend_height=True,
        )

        # Empty spacer to separate output from input box
        spacer_window = Window(height=3, dont_extend_height=True)

        layout = Layout(
            HSplit(
                [
                    spacer_window,
                    border_window,
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
                "toolbar-agent": "#cc88cc bold",
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
            # Wrap patched stdout with indenter for left margin on ALL output
            _patched = sys.stdout
            sys.stdout = _IndentedStdout(_patched)
            # Re-create console to pick up indented stdout
            self.console = Console(file=sys.stdout)
            try:
                app.run()
            finally:
                sys.stdout = _patched
                self.console = Console()

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
@click.option("--agent", type=str, default=None, help="Agent instance to use")
@click.pass_context
def cliver_cli(ctx: click.Context, agent: str | None):
    """
    Cliver: An application aims to make your CLI clever
    """
    cli = None
    if ctx.obj is None:
        cli = Cliver()
        if agent:
            cli._init_agent(agent)
            cli.config_manager.set_default_agent_name(agent)
        ctx.obj = cli

    if ctx.invoked_subcommand is None:
        # If no subcommand is invoked, start interactive mode
        interact(cli)


@cliver_cli.command(name="help", hidden=True)
@click.argument("command", nargs=-1)
@pass_cliver
@click.pass_context
def help_command(ctx: click.Context, cliver: Cliver, command: tuple):
    """Show help for cliver or a specific command."""
    group = cliver_cli
    # Walk down subcommand chain: cliver help model list → model list --help
    for cmd_name in command:
        sub = group.get_command(ctx, cmd_name)
        if sub is None:
            cliver.output(f"No such command '{cmd_name}'.")
            return
        if isinstance(sub, click.Group):
            group = sub
        else:
            cliver.output(sub.get_help(ctx))
            return
    cliver.output(group.get_help(ctx))


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

        # Simple text prompt (works in background thread with patch_stdout)
        while True:
            try:
                console.print("  [dim]\\[y]es / \\[n]o / \\[a]lways allow / \\[d]eny always[/dim]")
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
