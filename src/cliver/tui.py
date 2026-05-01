"""
Cliver TUI Module

Interactive terminal UI using prompt_toolkit. Extracted from cli.py.
Contains the run_tui() entry point and all TUI-related helpers.
"""

import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import click
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

if TYPE_CHECKING:
    from cliver.cli import Cliver


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
    skill names for ``/skills run <name>``.
    """

    def __init__(self, group: click.Group):
        self._group = group

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor

        if not text.startswith("/"):
            return

        stripped = text[1:]
        parts = stripped.split()

        at_new_token = stripped.endswith(" ") if stripped else True

        if not parts or (len(parts) == 1 and not at_new_token):
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
            return

        # Walk the command tree to find the current group/command
        cmd = self._group
        consumed = 0
        for i, part in enumerate(parts):
            if not isinstance(cmd, click.Group):
                break
            sub = cmd.get_command(None, part.lower())
            if sub is None:
                return
            cmd = sub
            consumed = i + 1

        # Remaining parts after resolving the command chain
        remaining = parts[consumed:]
        current_prefix = ""
        if not at_new_token and remaining:
            current_prefix = remaining[-1].lower()

        # /skills run <name> — offer skill name completions
        if self._is_skills_run(parts, consumed, at_new_token):
            yield from self._skill_name_completions(current_prefix)
            return

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

        if isinstance(cmd, (click.Command, click.Group)):
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

    @staticmethod
    def _is_skills_run(parts: list[str], consumed: int, at_new_token: bool) -> bool:
        """True when cursor is at the skill-name position of ``/skills run``."""
        if len(parts) < 2:
            return False
        if parts[0].lower() != "skills" or parts[1].lower() != "run":
            return False
        remaining = len(parts) - consumed
        if at_new_token and remaining == 0:
            return True
        if not at_new_token and remaining == 1:
            return True
        return False

    @staticmethod
    def _skill_name_completions(prefix: str):
        """Yield skill name completions for /skills run."""
        try:
            from cliver.skill_manager import SkillManager

            mgr = SkillManager()
            for skill in mgr.list_skills():
                if skill.name.startswith(prefix):
                    yield Completion(
                        skill.name,
                        start_position=-len(prefix),
                        display_meta=skill.description[:50] if skill.description else "",
                    )
        except Exception:
            return


# ─── TUI helper functions ───────────────────────────────────────────────────


def _echo_user_input(cliver: "Cliver", text: str) -> None:
    """Echo user input with a distinct background block in the conversation output."""
    import shutil as _shutil

    from cliver.themes import get_theme

    tw = _shutil.get_terminal_size().columns
    cliver.console.print(get_theme().user_input_markup(text, tw))


def _statusbar_content(cliver: "Cliver"):
    """Build status bar content parts."""
    try:
        from wcwidth import wcswidth

        def _dw(s):
            w = wcswidth(s)
            return w if w >= 0 else len(s)
    except ImportError:
        _dw = len

    cwd = str(Path.cwd())
    model = cliver.session_options.get("model") or cliver.config_manager.config.default_model or "—"
    mode = cliver.permission_manager._effective_mode().value
    tw = shutil.get_terminal_size().columns

    agent = cliver.agent_name
    left = f" 📂 {cwd} "
    right_agent = f" 🤖 {agent} "
    right_mode = f" 🔒 {mode} "
    right_model = f" ◆ {model} "

    middle_text = ""
    if cliver._rss_headlines:
        headline = cliver._rss_headlines[cliver._rss_index % len(cliver._rss_headlines)]
        cliver._rss_index += 1
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


def _get_toolbar_parts(cliver: "Cliver") -> FormattedText:
    """Build the toolbar formatted text for prompt_toolkit."""
    sb = cliver.session_options.get("statusbar", {})
    if not sb.get("visible", True):
        return FormattedText([])

    tw, left, middle, right_agent, right_mode, right_model = _statusbar_content(cliver)

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


# ─── Main TUI entry point ───────────────────────────────────────────────────


def run_tui(cliver: "Cliver") -> None:
    """Run the persistent TUI with input+toolbar always at the bottom."""
    from cliver.agent_profile import set_cli_instance, set_input_fn, set_output_fn
    from cliver.cli_ui import print_banner
    from cliver.command_router import CommandRouter
    from cliver.ui_bridge import CLIBridge, TUIBridge

    bridge = TUIBridge()
    cliver.ui = bridge
    router = CommandRouter(cliver)

    set_input_fn(lambda prompt="": bridge.ask_input(prompt))
    set_output_fn(lambda text: bridge.output(text))
    set_cli_instance(cliver)

    # Print banner to real stdout before TUI takes over
    default_model = cliver.config_manager.config.default_model
    agent_name = cliver.agent_name
    print_banner(cliver.console, agent_name, default_model)

    # Input buffer with history and completion
    history = FileHistory(str(cliver.history_path))
    completer = ClickCompleter(cliver._group) if hasattr(cliver, "_group") else None

    def on_accept(buff):
        line = buff.text.strip()
        buff.reset()
        if not line:
            return

        # Handle TUIBridge pending input (permission prompts, ask_user_question)
        if bridge._pending is not None:
            if bridge.try_receive(line):
                return
            # Invalid choice — let user retry (try_receive returns False)
            valid = ", ".join(bridge._valid_choices) if bridge._valid_choices else ""
            cliver.output(f"  [yellow]Invalid choice: '{line}'. Use {valid}[/yellow]")
            return

        # Legacy dialog state (cli_dialog.py still uses these until Task 6)
        if cliver._permission_pending is not None:
            choices = cliver._dialog_choices or []
            for choice in choices:
                if choice.matches(line):
                    cliver._permission_response = choice.value
                    cliver._permission_pending.set()
                    return
            if choices:
                valid = ", ".join(c.key for c in choices)
                cliver.output(f"  [yellow]Invalid choice: '{line}'. Use {valid}[/yellow]")
                return
            cliver._permission_response = line
            cliver._permission_pending.set()
            return

        if cliver._user_input_pending is not None:
            cliver._user_input_response = line
            cliver._user_input_pending.set()
            return

        if line:
            history.append_string(line)

        # Exit commands
        if line.lower() in ("exit", "quit", "/exit", "/quit"):
            app.exit()
            return

        # Echo user input
        _echo_user_input(cliver, line)

        # Slash commands
        if line.startswith("/"):
            parts = line[1:].split(None, 1)
            if not parts:
                return
            cmd_name = parts[0].lower()
            cmd_args = parts[1] if len(parts) > 1 else ""
            if router.is_busy:
                cliver.output("[yellow]A task is running. Ctrl+C to cancel.[/yellow]")
                return
            app.create_background_task(router.command(cmd_name, cmd_args))
            return

        # Plain text -> LLM query or follow-up
        if router.is_query_active:
            router.inject_input(line)
        elif router.is_busy:
            cliver.output("[yellow]A task is running. Ctrl+C to cancel.[/yellow]")
        else:
            cliver._cancel_requested = False
            app.create_background_task(router.query(line))

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
        # TUIBridge pending
        if bridge._pending is not None:
            bridge.cancel_pending()
            return
        # Legacy dialog state
        if cliver._permission_pending is not None:
            cliver._permission_response = "deny"
            cliver._permission_pending.set()
            return
        if cliver._user_input_pending is not None:
            cliver._user_input_response = ""
            cliver._user_input_pending.set()
            return
        # Cancel active router task
        if router.is_busy:
            if router._active_future:
                router._active_future.cancel()
            cliver._cancel_requested = True
            return
        event.app.exit()

    @kb.add("c-g")
    def _open_editor(event):
        """Ctrl+G opens $EDITOR for extended input."""
        event.current_buffer.open_in_editor()

    @kb.add("c-o")
    def _expand_output(event):
        """Ctrl+O expands the last truncated tool output in a pager."""
        import subprocess
        import tempfile

        from cliver.cli_tool_progress import get_last_full_output

        text, tool_name = get_last_full_output()
        if not text:
            cliver.output("[dim]No truncated output to expand.[/dim]")
            return

        header = f"── {tool_name} (full output) ──\n\n" if tool_name else ""
        full = header + text

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write(full)
            tmp_path = f.name

        try:
            import os

            pager = os.environ.get("PAGER", "less")
            subprocess.run([pager, tmp_path])
        finally:
            os.unlink(tmp_path)

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
        return FormattedText([("class:input-border", line)])

    border_window = Window(
        content=FormattedTextControl(_input_top_border),
        height=1,
        dont_extend_height=True,
        style="class:input-area",
    )

    from prompt_toolkit.layout.margins import Margin

    class LeftBorderMargin(Margin):
        def get_width(self, get_ui_content):
            return 2

        def create_margin(self, window_render_info, width, height):
            return [("class:input-border", "│ \n")] * height

    class RightBorderMargin(Margin):
        def get_width(self, get_ui_content):
            return 2

        def create_margin(self, window_render_info, width, height):
            return [("class:input-border", " │\n")] * height

    def _get_prompt():
        if bridge._pending is not None:
            if bridge._valid_choices:
                return [("class:permission-prompt", "⚠ y/n/a/d ❯ ")]
            return [("class:permission-prompt", "? ❯ ")]
        if cliver._permission_pending is not None:
            return [("class:permission-prompt", "⚠ y/n/a/d ❯ ")]
        if cliver._user_input_pending is not None:
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
        style="class:input-area",
    )

    toolbar_window = Window(
        content=FormattedTextControl(lambda: _get_toolbar_parts(cliver)),
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

    from cliver.themes import get_theme

    style = Style.from_dict(get_theme().prompt_toolkit_styles())

    app = Application(
        layout=layout,
        key_bindings=kb,
        style=style,
        full_screen=False,
    )
    cliver._app = app

    with patch_stdout(raw=True):
        _patched = sys.stdout
        sys.stdout = _IndentedStdout(_patched)
        indent_cols = len(_IndentedStdout.INDENT)
        effective_width = shutil.get_terminal_size().columns - indent_cols
        cliver.console = Console(file=sys.stdout, width=effective_width)
        try:
            app.run()
        finally:
            sys.stdout = _patched
            cliver.console = Console()

    # Cleanup
    cliver.ui = CLIBridge()
    router.shutdown()
    cliver.cleanup()
