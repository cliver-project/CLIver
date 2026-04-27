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
from rich.console import Console

from cliver import __version__, commands
from cliver.agent_profile import CliverProfile
from cliver.cli_tool_progress import ThinkingIndicator, create_tool_progress_handler
from cliver.config import ConfigManager
from cliver.llm import AgentCore
from cliver.permissions import PermissionManager
from cliver.ui_bridge import CLIBridge, UIBridge
from cliver.util import get_config_dir, read_piped_input, stdin_is_piped


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

        # UIBridge — defaults to CLIBridge, replaced by TUIBridge in run_tui()
        self.ui: UIBridge = CLIBridge()

        # Create permission manager (global + local project settings)
        self.permission_manager = PermissionManager(
            global_config_dir=self.config_dir,
            local_dir=Path.cwd() / ".cliver",
        )

        # Load theme from config
        from cliver.themes import load_theme, set_theme

        theme_name = self.config_manager.config.theme
        self.theme = load_theme(theme_name)
        set_theme(self.theme)

        # Thinking spinner shown while LLM is processing
        self.thinking = ThinkingIndicator(self.console)

        # Initialize agent (may be overridden by --agent flag before run)
        self._init_agent(self.config_manager.config.default_agent_name)

        # prepare console for interaction
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.history_path = self.config_dir / "history"
        self.session = None
        self._app = None  # prompt_toolkit Application set by TUI
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
        # These are used by cli_dialog.py and will be migrated to UIBridge in Task 6.
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

    def echo_user_input(self, text: str) -> None:
        """Echo user input with a distinct background block in the conversation output."""
        from cliver.themes import get_theme

        tw = shutil.get_terminal_size().columns
        self.console.print(get_theme().user_input_markup(text, tw))

    def _init_agent(self, agent_name: str) -> None:
        """Initialize (or reinitialize) all agent-scoped components.

        Ensures the agent directory and a default identity.md exist.
        """
        from cliver.token_tracker import TokenTracker

        self.agent_profile = CliverProfile(agent_name, self.config_dir)
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
        from cliver.util import configure_timezone

        configure_timezone(self.config_manager.config.timezone)

        self.agent_core = AgentCore(
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
            skill_auto_learn=self.config_manager.config.skill_auto_learn,
            model_auto_fallback=self.config_manager.config.model_auto_fallback,
        )
        self.agent_core.configure_rate_limits(self.config_manager.config.providers)

        from cliver.cost_tracker import CostTracker

        pricing = {}
        for name, model_cfg in self.config_manager.list_llm_models().items():
            resolved = model_cfg.get_resolved_pricing()
            if resolved:
                pricing[name] = resolved

        self.cost_tracker = CostTracker(pricing=pricing)

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

    def cleanup(self):
        """Clean up resources and enforce session storage limits."""
        # Trim current session and clean up old sessions
        try:
            sm = self.get_session_manager()
            sc = self.config_manager.config.session
            if self.current_session_id:
                sm.trim_turns(self.current_session_id, keep_last=sc.max_turns_per_session)
            sm.delete_stale_sessions(max_age_days=sc.max_age_days)
            sm.delete_oldest_sessions(keep=sc.max_sessions)
        except Exception:
            pass

        self.session_options = {}
        self.session = None
        self._app = None
        self.conversation_messages = []

    # ─── Run Methods ─────────────────────────────────────────────────────────

    def run(self) -> None:
        """Run the Cliver client."""
        from cliver.tui import run_tui

        run_tui(self)


pass_cliver = click.make_pass_decorator(Cliver)


@click.group(invoke_without_command=True)
@click.version_option(version=__version__, prog_name="cliver")
@click.option("--agent", type=str, default=None, help="Agent instance to use")
@click.option("--model", "-m", type=str, default=None, help="LLM model to use")
@click.option("--prompt", "-p", type=str, default=None, help="Prompt to send to the LLM")
@click.option("--image", multiple=True, help="Image files to send with the prompt")
@click.option("--audio", multiple=True, help="Audio files to send with the prompt")
@click.option("--video", multiple=True, help="Video files to send with the prompt")
@click.option("--file", "-f", multiple=True, help="Files to upload with the prompt")
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
@click.pass_context
def cliver_cli(
    ctx: click.Context,
    agent: str | None,
    model: str | None,
    prompt: str | None,
    image: tuple,
    audio: tuple,
    video: tuple,
    file: tuple,
    output_format: str | None,
    timeout: int | None,
    permission_mode: str | None,
    allow_tool: tuple,
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
    else:
        cli = ctx.obj

    # Apply CLI permission overrides
    if permission_mode:
        from cliver.permissions import PermissionMode

        cli.permission_manager.set_mode(PermissionMode(permission_mode))
    if allow_tool:
        from cliver.permissions import PermissionAction

        for tool_pattern in allow_tool:
            cli.permission_manager.grant_session(tool_pattern, PermissionAction.ALLOW)

    # Check for piped stdin
    effective_prompt = prompt
    if not effective_prompt and stdin_is_piped():
        try:
            effective_prompt = read_piped_input()
        except Exception:
            effective_prompt = None

    if effective_prompt:
        from cliver.command_router import CommandRouter

        if model:
            cli.session_options["model"] = model

        router = CommandRouter(cli)
        router.query_sync(
            effective_prompt,
            images=list(image),
            audio_files=list(audio),
            video_files=list(video),
            files=list(file),
            output_format=output_format,
            timeout_s=timeout,
        )
        router.shutdown()
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
    # Auto-register subcommands from CommandRouter HANDLERS
    _register_handler_commands()


def _register_handler_commands():
    """Auto-register Click subcommands for each CommandRouter handler.

    This makes `cliver <handler> <args>` work from the CLI by delegating
    to the handler's dispatch() function via CommandRouter.
    """
    from cliver.command_router import HANDLERS

    for handler_name in HANDLERS:
        # Skip if already registered (e.g. by loads_commands discovering a Click command)
        if handler_name in cliver_cli.commands:
            continue
        _make_cli_command(handler_name)


def _make_cli_command(handler_name: str):
    @cliver_cli.command(
        name=handler_name,
        context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
    )
    @click.argument("args", nargs=-1, type=click.UNPROCESSED)
    @click.pass_context
    def cmd(ctx, args):
        cliver_inst = ctx.ensure_object(Cliver)
        from cliver.command_router import CommandRouter

        router = CommandRouter(cliver_inst)
        router.command_sync(handler_name, " ".join(args))
        router.shutdown()

    return cmd


def _create_permission_prompt(console: Console, cliver_inst: "Cliver" = None):
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
        from rich.panel import Panel
        from rich.text import Text

        bare = tool_name.split("#")[-1] if "#" in tool_name else tool_name
        meta = get_tool_meta(bare)
        resource = str(args.get(meta.resource_param, "")) if meta.resource_param else ""
        label, color = _ACTION_LABELS.get(meta.action_kind, ("Unknown", "white"))

        args_summary = _format_args_summary(args, meta.resource_param)

        body = Text()
        body.append("Tool:     ", style="dim")
        body.append(f"{tool_name}\n", style="bold")
        body.append("Action:   ", style="dim")
        body.append(f"{label}\n", style=f"bold {color}")
        if resource:
            body.append("Resource: ", style="dim")
            body.append(f"{resource}\n")
        if args_summary:
            body.append("Args:     ", style="dim")
            body.append(f"{args_summary}\n")

        console.print()
        console.print(
            Panel(
                body,
                title="[bold yellow]Permission Required[/bold yellow]",
                border_style="yellow",
                padding=(0, 1),
                width=min(shutil.get_terminal_size().columns, 120),
            )
        )

        return cliver_inst.ui.ask_permission(
            tool_name,
            args,
            {"action_kind": label, "resource": resource},
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
