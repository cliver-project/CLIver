"""
Session management commands for interactive mode.

Manages conversation sessions — list, load, create, delete, compress.
Session recording happens automatically in chat.py after each LLM response.
"""

import asyncio

import click
from langchain_core.messages import AIMessage, HumanMessage

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

    # Convert recorded turns to LLM-ready BaseMessage objects
    cliver.conversation_messages = _turns_to_messages(turns)

    # Compress if the loaded history exceeds context window
    _compress_loaded_session(cliver)

    click.echo(f"Loaded session '{session_id}': {info.get('title', '(untitled)')}")
    click.echo(f"  {len(turns)} conversation turns restored ({len(cliver.conversation_messages)} messages in context).")
    click.echo("Continue chatting — the conversation history is active.")


@session_cmd.command(name="new", help="Start a new session")
@pass_cliver
def new_session(cliver: Cliver):
    """Create a new empty session and set it as current."""
    sm = cliver.get_session_manager()
    session_id = sm.create_session()
    cliver.current_session_id = session_id
    cliver.session_history = []
    cliver.conversation_messages = []
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
            cliver.conversation_messages = []
            click.echo("Active session cleared.")
    else:
        click.echo(f"Session '{session_id}' not found.")


@session_cmd.command(name="compress", help="Compress current conversation history")
@pass_cliver
def compress_session(cliver: Cliver):
    """Force-compress the current conversation history to save tokens."""
    from cliver.conversation_compressor import ConversationCompressor, estimate_tokens, get_context_window

    if not cliver.conversation_messages or len(cliver.conversation_messages) < 2:
        click.echo("Not enough conversation history to compress.")
        return

    # Get model config for context window
    _session_options = cliver.session_options or {}
    model_name = _session_options.get("model", None)
    task_executor = cliver.task_executor
    model_config = task_executor._get_llm_model(model_name)

    if not model_config:
        click.echo("No model configured. Cannot compress.")
        return

    context_window = get_context_window(model_config)
    compressor = ConversationCompressor(context_window)
    llm_engine = task_executor.get_llm_engine(model_name)

    before_tokens = estimate_tokens(cliver.conversation_messages)
    before_count = len(cliver.conversation_messages)

    try:
        compressed = asyncio.run(compressor.compress(cliver.conversation_messages, llm_engine, force=True))
    except Exception as e:
        click.echo(f"Compression failed: {e}")
        return

    cliver.conversation_messages = compressed
    after_tokens = estimate_tokens(cliver.conversation_messages)

    click.echo(
        f"Compressed: {before_count} messages (~{before_tokens} tokens) "
        f"→ {len(compressed)} messages (~{after_tokens} tokens)"
    )


def _turns_to_messages(turns):
    """Convert session turn dicts to LLM-ready BaseMessage objects."""
    messages = []
    for turn in turns:
        role = turn.get("role", "")
        content = turn.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    return messages


def _compress_loaded_session(cliver):
    """Compress conversation messages if they exceed context window on session load."""
    from cliver.conversation_compressor import ConversationCompressor, estimate_tokens, get_context_window

    if not cliver.conversation_messages:
        return

    _session_options = cliver.session_options or {}
    model_name = _session_options.get("model", None)
    task_executor = cliver.task_executor
    model_config = task_executor._get_llm_model(model_name)

    if not model_config:
        return

    context_window = get_context_window(model_config)
    compressor = ConversationCompressor(context_window)

    if not compressor.needs_compression([], cliver.conversation_messages, ""):
        return

    before_tokens = estimate_tokens(cliver.conversation_messages)
    llm_engine = task_executor.get_llm_engine(model_name)

    try:
        compressed = asyncio.run(compressor.compress(cliver.conversation_messages, llm_engine))
        cliver.conversation_messages = compressed
        after_tokens = estimate_tokens(cliver.conversation_messages)
        click.echo(f"[Session compressed: ~{before_tokens} → ~{after_tokens} tokens]")
    except Exception as e:
        click.echo(f"[Warning: session compression failed: {e}]")
