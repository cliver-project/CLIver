"""
Reusable TUI dialog component for user interaction.

Provides a consistent UI for permission prompts, ask_user_question,
and any future dialogs that need user input within the TUI.
"""

import threading
from typing import TYPE_CHECKING, List, Optional, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

if TYPE_CHECKING:
    from cliver.cli import Cliver


class DialogChoice:
    """A predefined choice for a dialog."""

    def __init__(self, key: str, label: str, value: str, aliases: Optional[List[str]] = None):
        """
        Args:
            key: Short key shown in brackets, e.g. "y"
            label: Display label, e.g. "yes"
            value: Return value when selected
            aliases: Additional accepted inputs, e.g. ["yes"]
        """
        self.key = key
        self.label = label
        self.value = value
        self.aliases = aliases or []

    def matches(self, input_str: str) -> bool:
        lower = input_str.lower()
        return lower == self.key or lower == self.label.lower() or lower in [a.lower() for a in self.aliases]


# Common choice sets
PERMISSION_CHOICES = [
    DialogChoice("y", "yes", "allow", ["yes"]),
    DialogChoice("n", "no", "deny", ["no"]),
    DialogChoice("a", "always allow", "allow_always", ["always"]),
    DialogChoice("d", "deny always", "deny_always", ["deny"]),
]


def show_dialog(
    console: Console,
    cliver_inst: Optional["Cliver"],
    title: str,
    fields: Optional[List[Tuple[str, str, str]]] = None,
    body_text: Optional[str] = None,
    choices: Optional[List[DialogChoice]] = None,
    numbered_options: Optional[List[Tuple[str, str]]] = None,
    border_style: str = "yellow",
    width: int = 60,
    default_on_cancel: str = "",
) -> str:
    """Show a dialog panel and collect user input.

    Args:
        console: Rich Console for output.
        cliver_inst: Cliver instance (for TUI input routing). None for CLI mode.
        title: Panel title (Rich markup supported).
        fields: List of (label, value, style) tuples for key-value display.
        body_text: Free-form body text (alternative to fields).
        choices: Predefined choices with key shortcuts (e.g. y/n/a/d).
        numbered_options: Numbered options as (label, description) tuples.
            User can pick by number or type free text.
        border_style: Rich style for the panel border.
        width: Panel width in characters.
        default_on_cancel: Value returned on Ctrl+C / cancel.

    Returns:
        The selected choice value, or free-form text from the user.
    """
    body = Text()

    # Render key-value fields
    if fields:
        max_label = max(len(f[0]) for f in fields)
        for label, value, style in fields:
            body.append(f"{label + ':':<{max_label + 1}} ", style="dim")
            body.append(f"{value}\n", style=style)

    # Render free-form body text
    if body_text:
        if fields:
            body.append("\n")
        body.append(body_text)
        body.append("\n")

    # Render numbered options
    if numbered_options:
        if fields or body_text:
            body.append("\n")
        for i, (label, desc) in enumerate(numbered_options, 1):
            body.append(f"  [{i}] ", style="bold")
            body.append(f"{label}\n")
            if desc:
                body.append(f"      {desc}\n", style="dim")
        body.append("  [0] ", style="bold")
        body.append("Other (type custom response)\n")

    # Render choice hints
    if choices:
        body.append("\n")
        hints = " / ".join(f"[{c.key}]{c.label[len(c.key) :]}" for c in choices)
        body.append(hints, style="dim")

    console.print()
    console.print(Panel(body, title=title, border_style=border_style, width=width))

    # Collect input
    return _collect_input(console, cliver_inst, choices, numbered_options, default_on_cancel)


def _collect_input(
    console: Console,
    cliver_inst: Optional["Cliver"],
    choices: Optional[List[DialogChoice]],
    numbered_options: Optional[List[Tuple[str, str]]],
    default_on_cancel: str,
) -> str:
    """Collect user input via TUI buffer or stdin."""
    if cliver_inst is not None and cliver_inst._app is not None:
        return _collect_tui(cliver_inst, choices, numbered_options, default_on_cancel)
    else:
        return _collect_stdin(console, choices, numbered_options, default_on_cancel)


def _collect_tui(
    cliver_inst: "Cliver",
    choices: Optional[List[DialogChoice]],
    numbered_options: Optional[List[Tuple[str, str]]],
    default_on_cancel: str,
) -> str:
    """Collect input through the TUI input buffer."""
    if choices:
        # Use permission-style input (validated choices)
        event = threading.Event()
        cliver_inst._permission_response = None
        cliver_inst._permission_pending = event
        cliver_inst._dialog_choices = choices
        event.wait()
        cliver_inst._permission_pending = None
        cliver_inst._dialog_choices = None
        return cliver_inst._permission_response or default_on_cancel
    else:
        # Use free-form input
        event = threading.Event()
        cliver_inst._user_input_response = None
        cliver_inst._user_input_pending = event
        event.wait()
        cliver_inst._user_input_pending = None
        response = cliver_inst._user_input_response or ""

        # Handle numbered options
        if numbered_options and response:
            try:
                choice = int(response)
                if 1 <= choice <= len(numbered_options):
                    return f"User selected: {numbered_options[choice - 1][0]}"
                elif choice == 0:
                    # Ask for custom response
                    event2 = threading.Event()
                    cliver_inst._user_input_response = None
                    cliver_inst._user_input_pending = event2
                    event2.wait()
                    cliver_inst._user_input_pending = None
                    return f"User response: {cliver_inst._user_input_response or ''}"
            except ValueError:
                pass
            return f"User response: {response}"

        return response


def _collect_stdin(
    console: Console,
    choices: Optional[List[DialogChoice]],
    numbered_options: Optional[List[Tuple[str, str]]],
    default_on_cancel: str,
) -> str:
    """Collect input via stdin (non-TUI mode)."""
    while True:
        try:
            response = input("  > ").strip()
        except (EOFError, KeyboardInterrupt):
            return default_on_cancel

        if choices:
            for choice in choices:
                if choice.matches(response):
                    return choice.value
            # Invalid choice — retry
            valid = ", ".join(c.key for c in choices)
            console.print(f"  [yellow]Invalid choice. Use {valid}[/yellow]")
            continue

        # Handle numbered options
        if numbered_options and response:
            try:
                choice_num = int(response)
                if 1 <= choice_num <= len(numbered_options):
                    return f"User selected: {numbered_options[choice_num - 1][0]}"
                elif choice_num == 0:
                    try:
                        custom = input("  Your response: ").strip()
                    except (EOFError, KeyboardInterrupt):
                        return default_on_cancel
                    return f"User response: {custom}"
            except ValueError:
                pass
            return f"User response: {response}"

        return response
