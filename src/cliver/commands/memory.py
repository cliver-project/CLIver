"""
Memory management commands.

View, add, and clear persistent memory entries.
Memory is stored in memory.md and injected into the system prompt.
"""

import click

from cliver.cli import Cliver, pass_cliver
from cliver.commands import click_help, wants_help


def _show_memory(cliver: Cliver):
    """Display current memory contents."""
    content = cliver.agent_profile.load_memory()
    if not content:
        cliver.output("No memories stored yet.")
        return
    cliver.output(content)


def _add_memory(cliver: Cliver, entry: str):
    """Add a memory entry."""
    cliver.agent_profile.append_memory(entry)
    cliver.output(f"Saved to memory: {entry}")


def _clear_memory(cliver: Cliver):
    """Clear all memory."""
    path = cliver.agent_profile.memory_file
    if not path.exists():
        cliver.output("Memory is already empty.")
        return
    response = cliver.ui.ask_input("Clear all memory? Type 'yes' to confirm: ")
    if response.lower() not in ("y", "yes"):
        cliver.output("Cancelled.")
        return
    path.unlink()
    cliver.output("Memory cleared.")


# TUI dispatch entry point

_SUBCOMMANDS: dict[str, click.Command] = {}


def dispatch(cliver: Cliver, args: str):
    """Manage persistent memory — show, add, clear."""
    parts = args.strip().split(None, 1) if args.strip() else []
    sub = parts[0] if parts else "show"
    rest = parts[1] if len(parts) > 1 else ""

    if sub in ("--help", "-h", "help"):
        cliver.output(click_help(memory_group, "/memory"))
        return

    if sub in _SUBCOMMANDS and wants_help(rest):
        cliver.output(click_help(_SUBCOMMANDS[sub], f"/memory {sub}"))
        return

    if sub == "show":
        _show_memory(cliver)
    elif sub == "add":
        if not rest:
            cliver.output("Usage: /memory add <text>")
            return
        _add_memory(cliver, rest)
    elif sub == "clear":
        _clear_memory(cliver)
    else:
        cliver.output(f"Unknown: /memory {sub}")
        cliver.output("Available: show, add, clear")


# Click wrappers


@click.group(
    name="memory",
    help="Manage persistent memory — show, add, clear",
    invoke_without_command=True,
)
@pass_cliver
@click.pass_context
def memory_group(ctx, cliver: Cliver):
    if ctx.invoked_subcommand is None:
        _show_memory(cliver)


@memory_group.command(name="show", help="Display current memory contents")
@pass_cliver
def show(cliver: Cliver):
    _show_memory(cliver)


@memory_group.command(name="add", help="Add a memory entry")
@click.argument("words", nargs=-1, required=True)
@pass_cliver
def add(cliver: Cliver, words: tuple):
    _add_memory(cliver, " ".join(words))


@memory_group.command(name="clear", help="Clear all memory (requires confirmation)")
@pass_cliver
def clear(cliver: Cliver):
    _clear_memory(cliver)


# Module-level alias for auto-discovery (filename stem must match)
memory = memory_group

_SUBCOMMANDS.update(
    {
        "show": show,
        "add": add,
        "clear": clear,
    }
)
