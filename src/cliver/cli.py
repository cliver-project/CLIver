"""
Cliver CLI Module

The main entrance of the cliver application
"""

import shutil
import sys
from pathlib import Path
from typing import Any, Dict

import click
from langchain_core.messages.base import BaseMessage
from prompt_toolkit import Application
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.completion import Completer, Completion
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
from cliver.llm import AgentCore
from cliver.permissions import PermissionManager
from cliver.util import get_config_dir, read_piped_input, stdin_is_piped


class _IndentedStdout:
    """Wraps stdout to add left margin, while preserving fileno() for Rich width detection."""

    INDENT = "   "

    def __init__(self, real_stdout):
        self._real = real_stdout
        self._at_line_start = True

    def write(self, text):
        if not text:
            return 0
        # Lines starting with \r are spinner animations — skip indent
        # so \r can properly overwrite the current line
        if text.startswith("\r"):
            return self._real.write(text)
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


class ClickCompleter(Completer):
    """prompt_toolkit completer that walks the Click command tree.

    Provides completions for commands, subcommands, options, and
    special forms like /skill:<name>.
    """

    def __init__(self, group: click.Group):
        self._group = group

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor

        # Only complete slash commands (lines starting with /)
        if not text.startswith("/"):
            return

        # Strip the leading / for matching against Click commands
        stripped = text[1:]
        parts = stripped.split()

        # If cursor is right after a space, we're completing the next token
        at_new_token = stripped.endswith(" ") if stripped else True

        if not parts or (len(parts) == 1 and not at_new_token):
            # Completing the first word (command name) — include / in completion
            prefix = parts[0].lower() if parts else ""
            for name, cmd in sorted(self._group.commands.items()):
                if cmd.hidden:
                    continue
                if name.startswith(prefix):
                    help_text = cmd.get_short_help_str(limit=50) if cmd.help else ""
                    display_name = f"/{name}"
                    yield Completion(
                        display_name,
                        start_position=-len(text),
                        display_meta=help_text,
                    )
            # Also offer /skill:<name> completions
            if "skill".startswith(prefix) or prefix.startswith("skill:"):
                yield from self._skill_completions(prefix, text)
            return

        # skill:xxx is a complete command — no further completions
        if parts and ":" in parts[0]:
            return

        # Walk the command tree to find the current group/command
        cmd = self._group
        consumed = 0
        for i, part in enumerate(parts):
            if not isinstance(cmd, click.Group):
                break
            sub = cmd.get_command(None, part.lower())
            if sub is None:
                return  # unrecognized command — no completions
            cmd = sub
            consumed = i + 1

        # Remaining parts after resolving the command chain
        remaining = parts[consumed:]
        current_prefix = ""
        if not at_new_token and remaining:
            current_prefix = remaining[-1].lower()

        # If the resolved command is a group, suggest subcommands
        if isinstance(cmd, click.Group):
            for name, sub in sorted(cmd.commands.items()):
                if sub.hidden:
                    continue
                if name.startswith(current_prefix):
                    help_text = sub.get_short_help_str(limit=50) if sub.help else ""
                    yield Completion(
                        name,
                        start_position=-len(current_prefix),
                        display_meta=help_text,
                    )

        # Suggest options for the resolved command
        if isinstance(cmd, (click.Command, click.Group)):
            # Collect already-used options
            used = {p.lower() for p in parts[consumed:] if p.startswith("-")}
            for param in cmd.params:
                if isinstance(param, click.Option):
                    for opt in param.opts + param.secondary_opts:
                        if opt.lower() in used:
                            continue
                        if opt.startswith(current_prefix):
                            help_text = param.help or ""
                            yield Completion(
                                opt,
                                start_position=-len(current_prefix),
                                display_meta=help_text[:50],
                            )

    def _skill_completions(self, prefix: str, full_text: str):
        """Yield /skill:<name> completions."""
        try:
            from cliver.skill_manager import SkillManager

            names = SkillManager().get_skill_names()
        except Exception:
            return
        for name in sorted(names):
            full = f"/skill:{name}"
            if full[1:].startswith(prefix):  # match without /
                yield Completion(full, start_position=-len(full_text))


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
        # Pending user input state for TUI mode (permission prompts, ask_user_question)
        self._permission_pending = None  # threading.Event when waiting
        self._permission_response = None  # result string
        self._dialog_choices = None  # list of DialogChoice for validation
        self._user_input_pending = None  # threading.Event for free-form input
        self._user_input_response = None  # result string

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
        self.task_executor = AgentCore(
            llm_models=self.config_manager.list_llm_models(),
            mcp_servers=self.config_manager.list_mcp_servers_for_mcp_caller(),
            default_model=self.config_manager.get_llm_model().name if self.config_manager.get_llm_model() else None,
            user_agent=self.config_manager.config.user_agent,
            agent_name=agent_name,
            on_tool_event=create_tool_progress_handler(self.console, thinking=self.thinking),
            agent_profile=self.agent_profile,
            token_tracker=self.token_tracker,
            permission_manager=self.permission_manager,
            on_permission_prompt=_create_permission_prompt(self.console, self),
            enabled_toolsets=self.config_manager.config.enabled_toolsets,
        )
        self.task_executor.configure_rate_limits(self.config_manager.config.providers)

    def switch_agent(self, agent_name: str) -> None:
        """Switch to a different agent, updating all scoped resources."""
        self._init_agent(agent_name)
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
                    # Route piped input as a chat command through Click
                    try:
                        cliver_cli(
                            args=["chat", user_data],
                            prog_name="cliver",
                            standalone_mode=False,
                            obj=self,
                        )
                    except click.UsageError as e:
                        if e.ctx:
                            self.output(e.ctx.get_help())
                        else:
                            self.output(str(e))
        else:
            self._run_tui()

    def _tui_input(self, prompt_text: str = "") -> str:
        """Input function for TUI mode — routes through the input buffer."""
        import threading

        if prompt_text:
            self.output(prompt_text)
        event = threading.Event()
        self._user_input_response = None
        self._user_input_pending = event
        event.wait()
        self._user_input_pending = None
        return self._user_input_response or ""

    def _run_tui(self) -> None:
        """Run the persistent TUI with input+toolbar always at the bottom."""
        # Register TUI input/output functions for tools like ask_user_question
        from cliver.agent_profile import set_cli_instance, set_input_fn, set_output_fn

        set_input_fn(self._tui_input)
        set_output_fn(lambda text: self.output(text))
        set_cli_instance(self)

        # Print banner to real stdout before TUI takes over
        default_model = self.config_manager.config.default_model
        agent_name = self.agent_name
        print_banner(self.console, agent_name, default_model)

        # Input buffer with history and completion
        history = FileHistory(str(self.history_path))
        completer = ClickCompleter(self._group) if hasattr(self, "_group") else None

        from cliver.command_dispatcher import CommandDispatcher
        from cliver.commands.handlers import (
            handle_agent,
            handle_capabilities,
            handle_config,
            handle_cost,
            handle_help,
            handle_identity,
            handle_mcp,
            handle_model,
            handle_permissions,
            handle_provider,
            handle_session,
            handle_skill,
            handle_skills,
            handle_task,
            handle_workflow,
        )

        dispatcher = CommandDispatcher(self)
        dispatcher.register("help", handle_help)
        dispatcher.register("model", handle_model)
        dispatcher.register("session", handle_session)
        dispatcher.register("config", handle_config)
        dispatcher.register("permissions", handle_permissions)
        dispatcher.register("skill", handle_skill)
        dispatcher.register("identity", handle_identity)
        dispatcher.register("mcp", handle_mcp)
        dispatcher.register("agent", handle_agent)
        dispatcher.register("cost", handle_cost)
        dispatcher.register("capabilities", handle_capabilities)
        dispatcher.register("provider", handle_provider)
        dispatcher.register("task", handle_task)
        dispatcher.register("workflow", handle_workflow)
        dispatcher.register("skills", handle_skills)

        async def chat_runner(text: str):
            from cliver.commands.chat_handler import ChatOptions, async_run_chat

            opts = ChatOptions(
                text=text,
                on_pending_input=dispatcher.drain_pending,
            )
            await async_run_chat(self, opts)

        dispatcher.set_chat_runner(chat_runner)

        def on_accept(buff):
            line = buff.text.strip()
            buff.reset()
            if not line:
                return

            # Handle permission/user input prompts (unchanged)
            if self._permission_pending is not None:
                choices = self._dialog_choices or []
                for choice in choices:
                    if choice.matches(line):
                        self._permission_response = choice.value
                        self._permission_pending.set()
                        return
                if choices:
                    valid = ", ".join(c.key for c in choices)
                    self.output(f"  [yellow]Invalid choice: '{line}'. Use {valid}[/yellow]")
                    return
                self._permission_response = line
                self._permission_pending.set()
                return

            if self._user_input_pending is not None:
                self._user_input_response = line
                self._user_input_pending.set()
                return

            if line:
                history.append_string(line)

            async def _dispatch():
                result = await dispatcher.dispatch(line)
                if result == "exit":
                    app.exit()

            self._cancel_requested = False
            app.create_background_task(_dispatch())

        input_buffer = Buffer(
            accept_handler=on_accept,
            multiline=False,
            history=history,
            auto_suggest=AutoSuggestFromHistory(),
            completer=completer,
            complete_while_typing=True,
        )

        # Key bindings
        kb = KeyBindings()

        @kb.add("c-c")
        def _ctrl_c(event):
            """Ctrl+C cancels running task, pending prompts, or exits if idle."""
            if self._permission_pending is not None:
                self._permission_response = "deny"
                self._permission_pending.set()
                return
            if self._user_input_pending is not None:
                self._user_input_response = ""
                self._user_input_pending.set()
                return
            if dispatcher.is_chat_active:
                dispatcher._active_chat_task.cancel()
                self._cancel_requested = True
                return
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

        def _get_prompt():
            if self._permission_pending is not None:
                return [("class:permission-prompt", "⚠ y/n/a/d ❯ ")]
            if self._user_input_pending is not None:
                return [("class:permission-prompt", "? ❯ ")]
            return [("class:prompt", "❯ ")]

        input_window = Window(
            content=BufferControl(
                buffer=input_buffer,
                input_processors=[BeforeInput(_get_prompt)],
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

        from prompt_toolkit.layout.containers import Float, FloatContainer
        from prompt_toolkit.layout.menus import CompletionsMenu

        layout = Layout(
            FloatContainer(
                content=HSplit(
                    [
                        spacer_window,
                        border_window,
                        input_window,
                        toolbar_window,
                    ]
                ),
                floats=[
                    Float(
                        xcursor=True,
                        ycursor=True,
                        content=CompletionsMenu(max_height=10, scroll_offset=1),
                    ),
                ],
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
                "permission-prompt": "ansiyellow bold",
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
            _patched = sys.stdout
            sys.stdout = _IndentedStdout(_patched)
            indent_cols = len(_IndentedStdout.INDENT)
            effective_width = shutil.get_terminal_size().columns - indent_cols
            self.console = Console(file=sys.stdout, width=effective_width)
            try:
                app.run()
            finally:
                sys.stdout = _patched
                self.console = Console()

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
@click.option("--model", "-m", type=str, default=None, help="LLM model to use (implies --no-fallback)")
@click.option("--prompt", "-p", type=str, default=None, help="Prompt to send (shorthand for 'chat <prompt>')")
@click.option(
    "--infile",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    default=None,
    help="Read a file and use its content as the prompt",
)
@click.option(
    "--output",
    "output_format",
    type=click.Choice(["text", "json"]),
    default=None,
    help="Output format (used with -p)",
)
@click.option("--timeout", type=int, default=None, help="Wall-clock timeout in seconds (used with -p)")
@click.option(
    "--permission-mode",
    type=click.Choice(["default", "auto-edit", "yolo"]),
    default=None,
    help="Permission mode override (used with -p)",
)
@click.option(
    "--allow-tool",
    multiple=True,
    type=str,
    help="Pre-grant a tool (used with -p, repeatable)",
)
@click.option(
    "--no-fallback",
    is_flag=True,
    default=False,
    help="Disable automatic model fallback (used with -p)",
)
@click.pass_context
def cliver_cli(
    ctx: click.Context,
    agent: str | None,
    model: str | None,
    prompt: str | None,
    infile: str | None,
    output_format: str | None,
    timeout: int | None,
    permission_mode: str | None,
    allow_tool: tuple,
    no_fallback: bool = False,
):
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

    # --infile: read file content as prompt (--prompt takes precedence)
    effective_prompt = prompt
    if not effective_prompt and infile:
        from cliver.util import read_file_content

        effective_prompt = read_file_content(infile)

    if effective_prompt:
        # Route to chat command with forwarded CI flags
        from cliver.commands.chat import chat

        invoke_kwargs = {"query": (effective_prompt,)}
        if model:
            invoke_kwargs["model"] = model
            invoke_kwargs["no_fallback"] = True  # --model implies --no-fallback
        if output_format:
            invoke_kwargs["output_format"] = output_format
        if timeout is not None:
            invoke_kwargs["timeout"] = timeout
        if permission_mode:
            invoke_kwargs["permission_mode"] = permission_mode
        if allow_tool:
            invoke_kwargs["allow_tool"] = allow_tool
        if no_fallback:
            invoke_kwargs["no_fallback"] = no_fallback
        ctx.invoke(chat, **invoke_kwargs)
        return

    if ctx.invoked_subcommand is None:
        # If no subcommand is invoked, start interactive mode
        session_options = {}
        if model:
            session_options["model"] = model
        interact(cli, session_options if session_options else None)


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


def _create_permission_prompt(console: Console, cliver_inst: "Cliver" = None):
    """Create a Rich-formatted permission prompt callback."""
    from cliver.cli_dialog import PERMISSION_CHOICES, show_dialog
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

        fields = [
            ("Tool", tool_name, "bold"),
            ("Action", label, f"bold {color}"),
        ]
        if resource:
            fields.append(("Resource", resource, "cyan"))
        if event_args_summary := _format_args_summary(args, meta.resource_param):
            fields.append(("Args", event_args_summary, "dim"))

        return show_dialog(
            console=console,
            cliver_inst=cliver_inst,
            title="[bold yellow]⚠  Permission Required[/bold yellow]",
            fields=fields,
            choices=PERMISSION_CHOICES,
            border_style="yellow",
            default_on_cancel="deny",
        )

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
    # bootstrap the cliver application — use standalone_mode=False
    # so command return values propagate as exit codes
    try:
        rc = cliver_cli(standalone_mode=False)
    except click.ClickException as e:
        e.show()
        sys.exit(e.exit_code)
    except (KeyboardInterrupt, click.Abort):
        sys.exit(130)
    sys.exit(rc or 0)
