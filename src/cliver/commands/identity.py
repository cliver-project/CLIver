"""
Identity management commands.

View and update the agent's identity profile through conversation.
The identity is a living markdown document describing the agent persona
and user profile (name, preferences, environment, etc.).
"""

import click

from cliver.cli import Cliver, pass_cliver
from cliver.commands import click_help, wants_help


@click.group(
    name="identity",
    help="Manage the agent's identity profile (persona, user preferences, communication style)",
    invoke_without_command=True,
)
@pass_cliver
@click.pass_context
def identity(ctx, cliver: Cliver):
    """View or update the identity profile."""
    if ctx.invoked_subcommand is None:
        _show_identity(cliver)


def _edit_identity(cliver: Cliver):
    """Open the identity.md file in the user's editor."""
    import os
    import subprocess

    editor = os.environ.get("EDITOR", os.environ.get("VISUAL", "vi"))
    identity_path = str(cliver.agent_profile.identity_file)

    cliver.agent_profile.ensure_dirs()
    if not cliver.agent_profile.identity_file.exists():
        from cliver.agent_profile import _DEFAULT_IDENTITY

        cliver.agent_profile.save_identity(_DEFAULT_IDENTITY)
        cliver.output(f"Created new identity file at {identity_path}")

    try:
        _run_editor(editor, identity_path)
    except subprocess.CalledProcessError as e:
        cliver.output(f"[red]Editor exited with error: {e}[/red]")
    except FileNotFoundError:
        cliver.output(f"[red]Editor '{editor}' not found. Set $EDITOR or $VISUAL.[/red]")


def _run_editor(editor: str, path: str) -> None:
    """Run an external editor, temporarily restoring cooked terminal mode.

    When CLIver's TUI (prompt_toolkit) is active, the terminal is in raw mode
    so it can capture individual keystrokes.  A subprocess editor inherits that
    broken state, rendering vim unusable.  We save the current terminal
    attributes, run ``stty sane`` to get a clean cooked mode for the editor,
    then restore the original attributes afterward so prompt_toolkit continues
    to work correctly.
    """
    import io
    import subprocess
    import sys

    # Resolve a real fd to the controlling terminal.  When stdin is
    # replaced by a non-file object (e.g. Click's CliRunner) there is
    # nothing to save/restore; we fall back to a plain subprocess call.
    saved = None
    tty_fd = None
    try:
        tty_fd = sys.stdin.fileno()
    except (io.UnsupportedOperation, OSError, AttributeError):
        pass

    if tty_fd is not None and sys.stdin.isatty():
        import termios

        saved = termios.tcgetattr(tty_fd)
        subprocess.run(["stty", "sane"], check=False)

    try:
        subprocess.run([editor, path], check=True)
    finally:
        if saved is not None:
            import termios

            termios.tcsetattr(tty_fd, termios.TCSANOW, saved)


def _clear_identity(cliver: Cliver):
    """Clear the identity profile."""
    cliver.agent_profile.save_identity("")
    cliver.output("Identity profile cleared.")


# ---------------------------------------------------------------------------
# Dispatch function
# ---------------------------------------------------------------------------


_SUBCOMMANDS: dict[str, click.Command] = {}


def dispatch(cliver: Cliver, args: str):
    """Manage agent identity — show, edit, clear."""
    parts = args.strip().split(None, 1) if args.strip() else []
    sub = parts[0] if parts else "show"
    rest = parts[1] if len(parts) > 1 else ""

    if sub in ("--help", "-h", "help"):
        cliver.output(click_help(identity, "/identity"))
        return

    if sub in _SUBCOMMANDS and wants_help(rest):
        cliver.output(click_help(_SUBCOMMANDS[sub], f"/identity {sub}"))
        return

    if sub == "show":
        _show_identity(cliver)
    elif sub == "edit":
        _edit_identity(cliver)
    elif sub == "clear":
        _clear_identity(cliver)
    else:
        cliver.output(f"Unknown subcommand: /identity {sub}")
        cliver.output("Run '/identity help' for usage.")


# ---------------------------------------------------------------------------
# Click commands (thin wrappers)
# ---------------------------------------------------------------------------


@identity.command(name="show", help="Display the current identity profile as markdown text")
@pass_cliver
def show_identity(cliver: Cliver):
    """Display the current identity document."""
    _show_identity(cliver)


@identity.command(name="edit", help="Open the identity profile in your editor for direct editing")
@pass_cliver
def edit_identity(cliver: Cliver):
    """Open identity.md in $EDITOR for direct editing."""
    _edit_identity(cliver)


@identity.command(name="clear", help="Delete the entire identity profile permanently (no confirmation)")
@pass_cliver
def clear_identity(cliver: Cliver):
    """Clear the identity profile."""
    _clear_identity(cliver)


# Populate subcommand map for dispatch help generation.
_SUBCOMMANDS.update(
    {
        "show": show_identity,
        "edit": edit_identity,
        "clear": clear_identity,
    }
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _show_identity(cliver: Cliver) -> None:
    """Display the current identity document."""
    content = cliver.agent_profile.load_identity()
    if not content:
        cliver.output("No identity profile set.")
        cliver.output("Run '/identity edit' to edit your profile.")
        return
    cliver.output(content)
