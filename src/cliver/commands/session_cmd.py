"""
Session management commands for interactive mode.

Manages conversation sessions — list, load, create, delete.
Session recording happens automatically in chat.py after each LLM response.
"""

import click

from cliver.cli import Cliver, pass_cliver


@click.group(name="session", help="Manage conversation sessions", invoke_without_command=True)
@pass_cliver
@click.pass_context
def session_cmd(ctx, cliver: Cliver):
    """View or manage conversation sessions."""
    if ctx.invoked_subcommand is None:
        # Default: show current session info
        if not hasattr(cliver, "current_session_id") or not cliver.current_session_id:
            click.echo("No active session.")
            return
        sm = cliver.get_session_manager()
        info = sm.get_session_info(cliver.current_session_id)
        if info:
            click.echo(f"Current session: {info['id']}")
            click.echo(f"  Title: {info.get('title', '(untitled)')}")
            click.echo(f"  Turns: {info.get('turn_count', 0)}")
            click.echo(f"  Created: {info.get('created_at', '?')}")
            click.echo(f"  Updated: {info.get('updated_at', '?')}")
        else:
            click.echo(f"Session '{cliver.current_session_id}' not found in index.")


@session_cmd.command(name="list", help="List all saved sessions")
@pass_cliver
def list_sessions(cliver: Cliver):
    """List all sessions, most recent first."""
    sm = cliver.get_session_manager()
    sessions = sm.list_sessions()

    if not sessions:
        click.echo("No saved sessions.")
        return

    click.echo("Sessions (most recent first):")
    for s in sessions:
        title = s.get("title", "(untitled)")
        if len(title) > 60:
            title = title[:57] + "..."
        turns = s.get("turn_count", 0)
        updated = s.get("updated_at", "")
        active = " *" if hasattr(cliver, "current_session_id") and s["id"] == cliver.current_session_id else ""
        click.echo(f"  {s['id']}{active}  [{turns} turns]  {title}  ({updated})")


@session_cmd.command(name="load", help="Load a previous session to continue the conversation")
@click.argument("session_id")
@pass_cliver
def load_session(cliver: Cliver, session_id: str):
    """Load a session's conversation history so the chat can continue."""
    sm = cliver.get_session_manager()
    info = sm.get_session_info(session_id)
    if not info:
        click.echo(f"Session '{session_id}' not found.")
        return

    turns = sm.load_turns(session_id)
    cliver.current_session_id = session_id
    cliver.session_history = turns

    click.echo(f"Loaded session '{session_id}': {info.get('title', '(untitled)')}")
    click.echo(f"  {len(turns)} conversation turns restored.")
    click.echo("Continue chatting — the conversation history is active.")


@session_cmd.command(name="new", help="Start a new session")
@pass_cliver
def new_session(cliver: Cliver):
    """Create a new empty session and set it as current."""
    sm = cliver.get_session_manager()
    session_id = sm.create_session()
    cliver.current_session_id = session_id
    cliver.session_history = []
    click.echo(f"New session started: {session_id}")


@session_cmd.command(name="delete", help="Delete a saved session")
@click.argument("session_id")
@pass_cliver
def delete_session(cliver: Cliver, session_id: str):
    """Delete a session and its conversation history."""
    sm = cliver.get_session_manager()
    if sm.delete_session(session_id):
        click.echo(f"Session '{session_id}' deleted.")
        if hasattr(cliver, "current_session_id") and cliver.current_session_id == session_id:
            cliver.current_session_id = None
            cliver.session_history = []
            click.echo("Active session cleared.")
    else:
        click.echo(f"Session '{session_id}' not found.")


