"""Clear conversation history."""

from __future__ import annotations

from typing import TYPE_CHECKING

import click

if TYPE_CHECKING:
    from cliver.cli import Cliver


def dispatch(cliver: "Cliver", args: str) -> None:
    """Clear the current conversation context."""
    router = getattr(cliver, "_command_router", None)
    if router and router.has_active_query:
        cliver.output("[yellow]Cannot clear while a query is running.[/yellow]")
        return

    cliver.conversation_messages = []
    cliver.session_history = []
    cliver.current_session_id = None
    cliver.output("Conversation cleared.")


@click.command(name="clear", help="Clear the conversation history and start fresh")
def clear_cmd():
    """Clear conversation — TUI only, this Click stub is for --help display."""
    pass
