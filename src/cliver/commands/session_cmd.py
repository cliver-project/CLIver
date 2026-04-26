"""
Session management commands for interactive mode.

Manages conversation sessions — list, load, create, delete, compress.
Also manages session-scoped permission grants via /session permission
and inference options via /session option.
Session recording happens automatically in chat.py after each LLM response.
"""

import asyncio

import click
from langchain_core.messages import AIMessage, HumanMessage

from cliver.cli import Cliver, pass_cliver
from cliver.config import ModelOptions
from cliver.permissions import PermissionAction, PermissionMode
from cliver.util import parse_key_value_options


@click.group(
    name="session",
    help="Manage conversation sessions (list, load, search, compress, options, permissions)",
    invoke_without_command=True,
)
@pass_cliver
@click.pass_context
def session_cmd(ctx, cliver: Cliver):
    """View or manage conversation sessions."""
    if ctx.invoked_subcommand is None:
        _show_current_session(cliver)


# ---------------------------------------------------------------------------
# Business logic functions
# ---------------------------------------------------------------------------


def _show_current_session(cliver: Cliver):
    """Show current session info."""
    if not hasattr(cliver, "current_session_id") or not cliver.current_session_id:
        cliver.output("No active session.")
        return
    sm = cliver.get_session_manager()
    info = sm.get_session_info(cliver.current_session_id)
    if info:
        cliver.output(f"Current session: {info['id']}")
        cliver.output(f"  Title: {info.get('title', '(untitled)')}")
        cliver.output(f"  Turns: {info.get('turn_count', 0)}")
        cliver.output(f"  Created: {info.get('created_at', '?')}")
        cliver.output(f"  Updated: {info.get('updated_at', '?')}")
    else:
        cliver.output(f"Session '{cliver.current_session_id}' not found in index.")


def _search_sessions(cliver: Cliver, query: str, limit: int = 10):
    """Full-text search across all past conversation sessions."""
    sm = cliver.get_session_manager()
    try:
        results = sm.search(query, limit=limit)
    except Exception as e:
        cliver.output(f"[red]Search failed: {e}[/red]")
        return

    if not results:
        cliver.output(f'No sessions found matching "{query}".')
        return

    cliver.output(f'\n[bold]🔍 {len(results)} session(s) matching "{query}"[/bold]\n')

    for r in results:
        title = r.get("title") or "(untitled)"
        sid = r["session_id"]
        created = r.get("created_at", "")
        turns = r.get("turn_count", 0)

        cliver.output("━" * 50)
        cliver.output(f"[bold]📋 {sid}[/bold] — {title}")
        cliver.output(f"   {created} · {turns} turns\n")

        for snippet in r.get("snippets", [])[:5]:
            role = snippet["role"]
            content = snippet["content"]
            cliver.output(f"   [dim]{role}:[/dim] {content}")

        cliver.output("")


def _list_sessions(cliver: Cliver):
    """List all sessions, most recent first."""
    sm = cliver.get_session_manager()
    sessions = sm.list_sessions()

    if not sessions:
        cliver.output("No saved sessions.")
        return

    cliver.output("Sessions (most recent first):")
    for s in sessions:
        title = s.get("title", "(untitled)")
        if len(title) > 60:
            title = title[:57] + "..."
        turns = s.get("turn_count", 0)
        updated = s.get("updated_at", "")
        active = " *" if hasattr(cliver, "current_session_id") and s["id"] == cliver.current_session_id else ""
        cliver.output(f"  {s['id']}{active}  [{turns} turns]  {title}  ({updated})")


def _load_session(cliver: Cliver, session_id: str):
    """Load a session's conversation history so the chat can continue."""
    sm = cliver.get_session_manager()
    info = sm.get_session_info(session_id)
    if not info:
        cliver.output(f"Session '{session_id}' not found.")
        return

    turns = sm.load_turns(session_id)
    cliver.current_session_id = session_id
    cliver.session_history = turns

    # Convert recorded turns to LLM-ready BaseMessage objects
    cliver.conversation_messages = _turns_to_messages(turns)

    # Compress if the loaded history exceeds context window
    _compress_loaded_session(cliver)

    # Restore saved session options (model, temperature, etc.)
    saved_options = sm.load_options(session_id)
    if saved_options:
        cliver.session_options.update(saved_options)

    cliver.output(f"Loaded session '{session_id}': {info.get('title', '(untitled)')}")
    msg_count = len(cliver.conversation_messages)
    cliver.output(f"  {len(turns)} conversation turns restored ({msg_count} messages in context).")
    if saved_options:
        model = saved_options.get("model")
        if model:
            cliver.output(f"  Restored session model: {model}")
    cliver.output("Continue chatting — the conversation history is active.")


def _new_session(cliver: Cliver):
    """Create a new empty session and set it as current."""
    sm = cliver.get_session_manager()
    session_id = sm.create_session()
    cliver.current_session_id = session_id
    cliver.session_history = []
    cliver.conversation_messages = []
    cliver.output(f"New session started: {session_id}")


def _delete_session(cliver: Cliver, session_id: str):
    """Delete a session and its conversation history."""
    sm = cliver.get_session_manager()
    if sm.delete_session(session_id):
        cliver.output(f"Session '{session_id}' deleted.")
        if hasattr(cliver, "current_session_id") and cliver.current_session_id == session_id:
            cliver.current_session_id = None
            cliver.session_history = []
            cliver.conversation_messages = []
            cliver.output("Active session cleared.")
    else:
        cliver.output(f"Session '{session_id}' not found.")


def _compress_session(cliver: Cliver):
    """Force-compress the current conversation history to save tokens."""
    from cliver.conversation_compressor import ConversationCompressor, estimate_tokens, get_context_window

    if not cliver.conversation_messages or len(cliver.conversation_messages) < 2:
        cliver.output("Not enough conversation history to compress.")
        return

    # Get model config for context window
    _session_options = cliver.session_options or {}
    model_name = _session_options.get("model", None)
    agent_core = cliver.agent_core
    model_config = agent_core._get_llm_model(model_name)

    if not model_config:
        cliver.output("No model configured. Cannot compress.")
        return

    context_window = get_context_window(model_config)
    compressor = ConversationCompressor(context_window)
    llm_engine = agent_core.get_llm_engine(model_name)

    before_tokens = estimate_tokens(cliver.conversation_messages)
    before_count = len(cliver.conversation_messages)

    try:
        compressed = asyncio.run(compressor.compress(cliver.conversation_messages, llm_engine, force=True))
    except Exception as e:
        cliver.output(f"Compression failed: {e}")
        return

    cliver.conversation_messages = compressed
    after_tokens = estimate_tokens(cliver.conversation_messages)

    cliver.output(
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
    agent_core = cliver.agent_core
    model_config = agent_core._get_llm_model(model_name)

    if not model_config:
        return

    context_window = get_context_window(model_config)
    compressor = ConversationCompressor(context_window)

    if not compressor.needs_compression([], cliver.conversation_messages, ""):
        return

    before_tokens = estimate_tokens(cliver.conversation_messages)
    llm_engine = agent_core.get_llm_engine(model_name)

    try:
        compressed = asyncio.run(compressor.compress(cliver.conversation_messages, llm_engine))
        cliver.conversation_messages = compressed
        after_tokens = estimate_tokens(cliver.conversation_messages)
        cliver.output(f"[Session compressed: ~{before_tokens} → ~{after_tokens} tokens]")
    except Exception as e:
        cliver.output(f"[Warning: session compression failed: {e}]")


# ---------------------------------------------------------------------------
# /session option — session-scoped inference options
# ---------------------------------------------------------------------------


def _display_options(cliver: Cliver) -> None:
    """Display current session options, showing real values for everything."""
    opts = cliver.session_options
    defaults = ModelOptions()

    # Resolve real values: session override → model config → global defaults
    default_model = cliver.config_manager.config.default_model or "(none)"
    model = opts.get("model") or default_model

    # Get per-model option overrides if a model is configured
    model_config = cliver.config_manager.get_llm_model(model if model != "(none)" else None)
    model_opts = model_config.options if model_config and model_config.options else defaults

    llm_opts = opts.get("options", {})
    temperature = llm_opts.get("temperature", model_opts.temperature)
    max_tokens = llm_opts.get("max_tokens", model_opts.max_tokens)
    top_p = llm_opts.get("top_p", model_opts.top_p)
    freq_penalty = llm_opts.get("frequency_penalty", model_opts.frequency_penalty)

    stream = opts.get("stream", True)
    save_media = opts.get("save_media", False)
    media_dir = opts.get("media_dir") or "(current directory)"

    cliver.output("Session options:")
    cliver.output(f"  model:             {model}")
    cliver.output(f"  stream:            {stream}")
    cliver.output(f"  temperature:       {temperature}")
    cliver.output(f"  max_tokens:        {max_tokens}")
    cliver.output(f"  top_p:             {top_p}")
    cliver.output(f"  frequency_penalty: {freq_penalty}")
    cliver.output(f"  save_media:        {save_media}")
    cliver.output(f"  media_dir:         {media_dir}")

    # Show any extra key=value options
    extra = {k: v for k, v in llm_opts.items() if k not in ("temperature", "max_tokens", "top_p", "frequency_penalty")}
    if extra:
        cliver.output(f"  extra options:     {extra}")

    # Show model exclusions
    excluded = cliver.agent_core.excluded_models
    if excluded:
        cliver.output(f"\n  excluded models:   {', '.join(sorted(excluded))}")


@session_cmd.group(
    name="option",
    help="Manage session-scoped inference options (model, temperature, streaming, etc.)",
    invoke_without_command=True,
)
@pass_cliver
@click.pass_context
def session_option(ctx, cliver: Cliver):
    """View or modify persistent inference options for the current session."""
    if ctx.invoked_subcommand is None:
        _display_options(cliver)


@session_option.command(name="set", help="Set one or more inference options for the current session")
@click.option("--model", "-m", type=str, help="LLM model to use for this session. Must match a name from 'model list'")
@click.option("--temperature", type=float, help="Sampling temperature (0.0 = deterministic, higher = more creative)")
@click.option("--max-tokens", type=int, help="Maximum number of tokens in the LLM response")
@click.option("--top-p", type=float, help="Top-p (nucleus) sampling parameter (0.0-1.0)")
@click.option("--frequency-penalty", type=float, help="Frequency penalty to reduce repetition (0.0-2.0)")
@click.option(
    "--stream",
    "-s",
    is_flag=True,
    default=None,
    help="Enable streaming output (tokens displayed as generated)",
)
@click.option("--no-stream", is_flag=True, default=None, help="Disable streaming (wait for full response)")
@click.option(
    "--save-media",
    "-sm",
    is_flag=True,
    default=None,
    help="Enable saving generated media (images, audio) to disk",
)
@click.option("--no-save-media", is_flag=True, default=None, help="Disable saving generated media to disk")
@click.option("--media-dir", "-md", type=str, help="Directory path for saved media files (default: current directory)")
@click.option(
    "--option",
    multiple=True,
    type=str,
    help="Additional option as key=value (repeatable, e.g. --option seed=42)",
)
@pass_cliver
def set_options(
    cliver: Cliver,
    model,
    temperature,
    max_tokens,
    top_p,
    frequency_penalty,
    stream,
    no_stream,
    save_media,
    no_save_media,
    media_dir,
    option,
):
    """Set one or more inference options for the current session."""
    options_provided = any(
        [
            model is not None,
            temperature is not None,
            max_tokens is not None,
            top_p is not None,
            frequency_penalty is not None,
            stream is not None,
            no_stream is not None,
            save_media is not None,
            no_save_media is not None,
            media_dir is not None,
            len(option) > 0,
        ]
    )

    if not options_provided:
        _display_options(cliver)
        return 0

    _llm_options = cliver.session_options.get("options", {})

    if model is not None:
        if not cliver.config_manager.get_llm_model(model):
            cliver.output(f"Unknown model: {model}, please define it first.")
            return 1
        cliver.session_options["model"] = model
        cliver.output(f"Set model to '{model}' for this session.")

    if temperature is not None:
        _llm_options["temperature"] = temperature
        cliver.output(f"Set temperature to {temperature} for this session.")

    if max_tokens is not None:
        _llm_options["max_tokens"] = max_tokens
        cliver.output(f"Set max_tokens to {max_tokens} for this session.")

    if top_p is not None:
        _llm_options["top_p"] = top_p
        cliver.output(f"Set top_p to {top_p} for this session.")

    if frequency_penalty is not None:
        _llm_options["frequency_penalty"] = frequency_penalty
        cliver.output(f"Set frequency_penalty to {frequency_penalty} for this session.")

    if stream is True:
        cliver.session_options["stream"] = True
        cliver.output("Enabled streaming for this session.")
    elif no_stream is True:
        cliver.session_options["stream"] = False
        cliver.output("Disabled streaming for this session.")

    if save_media is True:
        cliver.session_options["save_media"] = True
        cliver.output("Enabled save-media for this session.")
    elif no_save_media is True:
        cliver.session_options["save_media"] = False
        cliver.output("Disabled save-media for this session.")

    if media_dir is not None:
        cliver.session_options["media_dir"] = media_dir
        cliver.output(f"Set media_dir to '{media_dir}' for this session.")

    if option and len(option) > 0:
        opts_dict = parse_key_value_options(option)
        _llm_options.update(opts_dict)
        cliver.output(f"Updated additional options: {dict(opts_dict)}")

    # Persist options into session data so they survive load/restore
    _persist_session_options(cliver)

    return 0


@session_option.command(
    name="reset",
    help="Reset all session options to their default values and clear model exclusions",
)
@pass_cliver
def reset_options(cliver: Cliver):
    """Reset all session options to their defaults."""
    default_options = ModelOptions()
    cliver.session_options = {
        "model": cliver.config_manager.get_llm_model(),
        "temperature": default_options.temperature,
        "max_tokens": default_options.max_tokens,
        "top_p": default_options.top_p,
        "frequency_penalty": default_options.frequency_penalty,
        "options": {},
        "stream": False,
        "save_media": False,
        "media_dir": None,
    }
    cliver.agent_core.excluded_models.clear()
    cliver.output("Session options have been reset to defaults.")
    _persist_session_options(cliver)
    return 0


@session_option.command(name="exclude", help="Exclude a model from being used as a fallback target for this session")
@click.argument("model_name")
@pass_cliver
def option_model_exclude(cliver: Cliver, model_name: str):
    """Exclude a model from being used as a fallback target."""
    if model_name not in cliver.config_manager.list_llm_models():
        cliver.output(f"Unknown model: {model_name}")
        return
    cliver.agent_core.excluded_models.add(model_name)
    cliver.output(f"Excluded '{model_name}' from fallback for this session.")


@session_option.command(name="include", help="Re-include a previously excluded model for fallback")
@click.argument("model_name")
@pass_cliver
def option_model_include(cliver: Cliver, model_name: str):
    """Re-include a previously excluded model."""
    if model_name in cliver.agent_core.excluded_models:
        cliver.agent_core.excluded_models.discard(model_name)
        cliver.output(f"Re-included '{model_name}' for fallback.")
    else:
        cliver.output(f"'{model_name}' is not excluded.")


def _persist_session_options(cliver: Cliver) -> None:
    """Save current session options to the session index if a session is active."""
    if not cliver.current_session_id:
        return
    sm = cliver.get_session_manager()
    sm.save_options(cliver.current_session_id, cliver.session_options)


def _dispatch_option(cliver: Cliver, args: str):
    """Dispatch /session option subcommands."""
    parts = args.strip().split(None, 1) if args.strip() else []
    sub = parts[0] if parts else "show"
    rest = parts[1] if len(parts) > 1 else ""

    if sub in ("show", ""):
        _display_options(cliver)
    elif sub == "set":
        # Parse key=value pairs
        if not rest:
            cliver.output("[yellow]Usage: /session option set <key>=<value> ...[/yellow]")
            return
        from cliver.util import parse_key_value_options

        opts = parse_key_value_options(rest.split())
        for key, value in opts.items():
            cliver.session_options.setdefault("options", {})[key] = value
        cliver.output(f"[green]Set: {opts}[/green]")
    elif sub == "reset":
        cliver.session_options.pop("options", None)
        cliver.output("[green]Session options reset to defaults.[/green]")
    elif sub == "exclude":
        if rest:
            cliver.agent_core.excluded_models.add(rest.strip())
            cliver.output(f"Excluded model: {rest.strip()}")
        else:
            cliver.output("[yellow]Usage: /session option exclude <model>[/yellow]")
    elif sub == "include":
        if rest:
            cliver.agent_core.excluded_models.discard(rest.strip())
            cliver.output(f"Re-included model: {rest.strip()}")
        else:
            cliver.output("[yellow]Usage: /session option include <model>[/yellow]")
    elif sub in ("--help", "help"):
        cliver.output("Manage inference options for the current session only.")
        cliver.output("")
        cliver.output("Usage: /session option [show|set|reset|exclude|include] [arguments]")
        cliver.output("")
        cliver.output("Subcommands:")
        cliver.output("  show                — Display current session options (model, temperature, etc.).")
        cliver.output("  set <key>=<value>   — Set one or more options. Supported keys:")
        cliver.output("    model=STRING, temperature=FLOAT, max_tokens=INT, top_p=FLOAT,")
        cliver.output("    frequency_penalty=FLOAT, or any custom key=value.")
        cliver.output("    Example: /session option set model=qwen/qwen3-coder")
        cliver.output("    Example: /session option set temperature=0.7")
        cliver.output("  reset               — Reset all session options to defaults.")
        cliver.output("  exclude <model>     — Exclude a model from being used as fallback.")
        cliver.output("    model  STRING (required) — Model name from '/model list'.")
        cliver.output("  include <model>     — Re-include a previously excluded model.")
        cliver.output("    model  STRING (required) — Model name to re-include.")
    else:
        cliver.output(f"[yellow]Unknown: /session option {sub}[/yellow]")


def _dispatch_permission(cliver: Cliver, args: str):
    """Dispatch /session permission subcommands."""
    from cliver.permissions import PermissionAction, PermissionMode

    parts = args.strip().split(None, 1) if args.strip() else []
    sub = parts[0] if parts else "show"
    rest = parts[1] if len(parts) > 1 else ""

    pm = cliver.permission_manager

    if sub in ("show", ""):
        effective = pm._effective_mode()
        cliver.output(f"Permission mode: {pm.mode.value} (effective: {effective.value})")
        grants = pm._session_grants
        if not grants:
            cliver.output("No session grants active.")
        else:
            cliver.output(f"\nSession grants ({len(grants)}):")
            for tool, action in grants.items():
                marker = "✔" if action == PermissionAction.ALLOW else "✘"
                cliver.output(f"  {marker} {tool}: {action.value}")
    elif sub == "mode":
        if not rest or rest.strip() not in ("default", "auto-edit", "yolo"):
            cliver.output("[yellow]Usage: /session permission mode <default|auto-edit|yolo>[/yellow]")
            return
        pm.set_mode(PermissionMode(rest.strip()))
        cliver.output(f"Session permission mode set to '{rest.strip()}'.")
    elif sub == "grant":
        if not rest:
            cliver.output("[yellow]Usage: /session permission grant <tool>[/yellow]")
            return
        pm.grant_session(rest.strip(), PermissionAction.ALLOW)
        cliver.output(f"Granted: {rest.strip()} allowed for this session.")
    elif sub == "deny":
        if not rest:
            cliver.output("[yellow]Usage: /session permission deny <tool>[/yellow]")
            return
        pm.grant_session(rest.strip(), PermissionAction.DENY)
        cliver.output(f"Denied: {rest.strip()} blocked for this session.")
    elif sub == "clear":
        pm.clear_session_grants()
        cliver.output("All session permission grants cleared.")
    elif sub in ("--help", "help"):
        cliver.output("Manage session-scoped permission grants (not saved to file).")
        cliver.output("")
        cliver.output("Usage: /session permission [show|mode|grant|deny|clear] [arguments]")
        cliver.output("")
        cliver.output("Subcommands:")
        cliver.output("  show              — Show active session grants and effective permission mode.")
        cliver.output("  mode <mode>       — Override permission mode for this session only.")
        cliver.output("    mode  CHOICE(default|auto-edit|yolo) (required)")
        cliver.output("  grant <tool>      — Allow a tool for the rest of this session.")
        cliver.output("    tool  STRING (required) — Tool name or regex pattern (e.g. 'Bash', 'github#.*').")
        cliver.output("  deny <tool>       — Deny a tool for the rest of this session.")
        cliver.output("    tool  STRING (required) — Tool name or regex pattern.")
        cliver.output("  clear             — Remove all session permission grants.")
    else:
        cliver.output(f"[yellow]Unknown: /session permission {sub}[/yellow]")


# ---------------------------------------------------------------------------
# Dispatch function
# ---------------------------------------------------------------------------


def dispatch(cliver: Cliver, args: str):
    """Manage conversation sessions — list, search, load, compress, permissions."""
    parts = args.strip().split(None, 1) if args.strip() else []
    sub = parts[0] if parts else "list"
    rest = parts[1] if len(parts) > 1 else ""

    if sub == "list" or sub == "":
        _show_current_session(cliver)
    elif sub == "search":
        if not rest:
            cliver.output("[red]Missing search query[/red]")
            return
        _search_sessions(cliver, rest.strip())
    elif sub == "load":
        if not rest:
            cliver.output("[red]Missing session ID[/red]")
            return
        _load_session(cliver, rest.strip())
    elif sub == "new":
        _new_session(cliver)
    elif sub == "delete":
        if not rest:
            cliver.output("[red]Missing session ID[/red]")
            return
        _delete_session(cliver, rest.strip())
    elif sub == "compress":
        _compress_session(cliver)
    elif sub == "option":
        _dispatch_option(cliver, rest)
    elif sub == "permission":
        _dispatch_permission(cliver, rest)
    elif sub in ("--help", "help"):
        cliver.output("Manage conversation sessions. Sessions store chat history and can be")
        cliver.output("saved, loaded, searched, and compressed.")
        cliver.output("")
        cliver.output("Usage: /session [list|search|load|new|delete|compress|option|permission] [arguments]")
        cliver.output("")
        cliver.output("Subcommands:")
        cliver.output("  (no args)            — Show current session info (ID, title, turn count, dates).")
        cliver.output("  list                 — List all saved sessions (most recent first) with ID,")
        cliver.output("                         title, turn count, and last update time.")
        cliver.output("  search <query>       — Full-text search across all past sessions.")
        cliver.output("    query  STRING (required) — Search term to find in conversation content.")
        cliver.output("    Example: /session search 'authentication bug'")
        cliver.output("  load <id>            — Load a previous session's conversation history so you")
        cliver.output("                         can continue chatting. Auto-compresses if too long.")
        cliver.output("    id  STRING (required) — Session ID from '/session list'.")
        cliver.output("    Example: /session load abc123")
        cliver.output("  new                  — Start a fresh session (clears current conversation).")
        cliver.output("  delete <id>          — Delete a session and its conversation history.")
        cliver.output("    id  STRING (required) — Session ID to delete.")
        cliver.output("  compress             — Force-compress the current conversation history to")
        cliver.output("                         reduce token usage. Uses the LLM to summarize.")
        cliver.output("")
        cliver.output("Sub-groups:")
        cliver.output("  option               — Manage session-scoped inference options.")
        cliver.output("    /session option                          — show current options")
        cliver.output("    /session option set <key>=<value> ...    — set option(s)")
        cliver.output("    /session option reset                    — reset all to defaults")
        cliver.output("    /session option exclude <model>          — exclude model from fallback")
        cliver.output("    /session option include <model>          — re-include excluded model")
        cliver.output("  permission           — Manage session-scoped permission grants.")
        cliver.output("    /session permission                      — show grants and mode")
        cliver.output("    /session permission mode <default|auto-edit|yolo>  — override mode")
        cliver.output("    /session permission grant <tool>         — allow tool this session")
        cliver.output("    /session permission deny <tool>          — deny tool this session")
        cliver.output("    /session permission clear                — clear all session grants")
        cliver.output("")
        cliver.output("Default subcommand: (show current session info)")
    else:
        cliver.output(f"[yellow]Unknown subcommand: /session {sub}[/yellow]")
        cliver.output("Run '/session help' for usage.")


# ---------------------------------------------------------------------------
# /session permission — session-scoped permission grants
# ---------------------------------------------------------------------------


@session_cmd.group(
    name="permission",
    help="Manage session-scoped permission grants (not saved to file, cleared on session end)",
    invoke_without_command=True,
)
@pass_cliver
@click.pass_context
def session_permission(ctx, cliver: Cliver):
    """Show or manage session permission grants."""
    if ctx.invoked_subcommand is None:
        ctx.invoke(show_session_permissions, cliver=cliver)


@session_permission.command(name="show", help="Show session grants and effective mode", hidden=True)
@pass_cliver
def show_session_permissions(cliver: Cliver):
    """Display current session grants and effective mode."""
    pm = cliver.permission_manager
    effective = pm._effective_mode()
    cliver.output(f"Permission mode: {pm.mode.value} (effective: {effective.value})")

    grants = pm._session_grants
    if not grants:
        cliver.output("No session grants active.")
    else:
        cliver.output(f"\nSession grants ({len(grants)}):")
        for tool, action in grants.items():
            marker = "✔" if action == PermissionAction.ALLOW else "✘"
            cliver.output(f"  {marker} {tool}: {action.value}")


@session_permission.command(name="mode", help="Override permission mode for this session only (not saved to file)")
@click.argument("new_mode", type=click.Choice(["default", "auto-edit", "yolo"]))
@pass_cliver
def session_set_mode(cliver: Cliver, new_mode: str):
    """Override the permission mode for the current session only (not saved)."""
    mode = PermissionMode(new_mode)
    cliver.permission_manager.set_mode(mode)
    cliver.output(f"Session permission mode set to '{new_mode}' (not saved to file).")


@session_permission.command(name="grant", help="Allow a tool (or tool regex pattern) for the rest of this session")
@click.argument("tool")
@pass_cliver
def session_grant(cliver: Cliver, tool: str):
    """Allow a tool (or tool pattern) for the rest of this session."""
    cliver.permission_manager.grant_session(tool, PermissionAction.ALLOW)
    cliver.output(f"Granted: {tool} is allowed for this session.")


@session_permission.command(name="deny", help="Deny a tool (or tool regex pattern) for the rest of this session")
@click.argument("tool")
@pass_cliver
def session_deny(cliver: Cliver, tool: str):
    """Deny a tool (or tool pattern) for the rest of this session."""
    cliver.permission_manager.grant_session(tool, PermissionAction.DENY)
    cliver.output(f"Denied: {tool} is blocked for this session.")


@session_permission.command(name="clear", help="Remove all session permission grants, reverting to persistent rules")
@pass_cliver
def session_clear(cliver: Cliver):
    """Clear all session grants, reverting to persistent rules."""
    cliver.permission_manager.clear_session_grants()
    cliver.output("All session permission grants cleared.")


# ---------------------------------------------------------------------------
# Click command wrappers
# ---------------------------------------------------------------------------


@session_cmd.command(name="search", help="Full-text search across all past conversation sessions")
@click.argument("query")
@click.option(
    "--limit",
    "-n",
    type=int,
    default=10,
    help="Maximum number of matching sessions to display (default: 10)",
)
@pass_cliver
def search_sessions(cliver: Cliver, query: str, limit: int):
    """Full-text search across all past conversation sessions."""
    _search_sessions(cliver, query, limit)


@session_cmd.command(name="list", help="List all saved sessions (most recent first) with ID, title, turn count")
@pass_cliver
def list_sessions(cliver: Cliver):
    """List all sessions, most recent first."""
    _list_sessions(cliver)


@session_cmd.command(
    name="load",
    help="Load a previous session's history to continue the conversation (auto-compresses if needed)",
)
@click.argument("session_id")
@pass_cliver
def load_session(cliver: Cliver, session_id: str):
    """Load a session's conversation history so the chat can continue."""
    _load_session(cliver, session_id)


@session_cmd.command(name="new", help="Start a fresh session (clears current conversation history)")
@pass_cliver
def new_session(cliver: Cliver):
    """Create a new empty session and set it as current."""
    _new_session(cliver)


@session_cmd.command(name="delete", help="Delete a session and its entire conversation history permanently")
@click.argument("session_id")
@pass_cliver
def delete_session(cliver: Cliver, session_id: str):
    """Delete a session and its conversation history."""
    _delete_session(cliver, session_id)


@session_cmd.command(
    name="compress",
    help="Force-compress current conversation history via LLM summarization to save tokens",
)
@pass_cliver
def compress_session(cliver: Cliver):
    """Force-compress the current conversation history to save tokens."""
    _compress_session(cliver)
