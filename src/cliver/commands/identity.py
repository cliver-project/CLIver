"""
Identity management commands.

View and update the agent's identity profile through conversation.
The identity is a living markdown document describing the agent persona
and user profile (name, preferences, environment, etc.).
"""

import click

from cliver.cli import Cliver, pass_cliver


@click.group(
    name="identity",
    help="Manage the agent identity profile",
    invoke_without_command=True,
)
@pass_cliver
@click.pass_context
def identity(ctx, cliver: Cliver):
    """View or update the identity profile."""
    if ctx.invoked_subcommand is None:
        _show_identity(cliver)


def _chat_identity(cliver: Cliver):
    """Start a guided conversation to build or update the identity profile."""
    prompt = _build_identity_prompt(cliver)

    # Route through the chat command so it goes through the normal
    # AgentCore flow (with tools, memory, session recording, etc.)
    from cliver.cli import cliver_cli

    try:
        cliver_cli(
            args=["chat", prompt],
            prog_name="cliver",
            standalone_mode=False,
            obj=cliver,
        )
    except click.UsageError as e:
        if e.ctx:
            cliver.output(e.ctx.get_help())
        else:
            cliver.output(str(e))


def _clear_identity(cliver: Cliver):
    """Clear the identity profile."""
    cliver.agent_profile.save_identity("")
    cliver.output("Identity profile cleared.")


# ---------------------------------------------------------------------------
# Dispatch function
# ---------------------------------------------------------------------------


def dispatch(cliver: Cliver, args: str):
    """Dispatch /identity commands from string args."""
    parts = args.strip().split(None, 1) if args.strip() else []
    sub = parts[0] if parts else "show"

    if sub == "show":
        _show_identity(cliver)
    elif sub == "chat":
        _chat_identity(cliver)
    elif sub == "clear":
        _clear_identity(cliver)
    elif sub in ("--help", "help"):
        cliver.output("Usage: /identity [show|chat|clear]")
        cliver.output("  show  - Display the current identity profile")
        cliver.output("  chat  - Update identity through guided conversation")
        cliver.output("  clear - Clear the identity profile")
    else:
        cliver.output(f"[yellow]Unknown subcommand: /identity {sub}[/yellow]")
        cliver.output("Run '/identity help' for usage.")


# ---------------------------------------------------------------------------
# Click commands (thin wrappers)
# ---------------------------------------------------------------------------


@identity.command(name="show", help="Show the current identity profile")
@pass_cliver
def show_identity(cliver: Cliver):
    """Display the current identity document."""
    _show_identity(cliver)


@identity.command(name="chat", help="Update identity through a guided conversation")
@pass_cliver
def chat_identity(cliver: Cliver):
    """Start a guided conversation to build or update the identity profile.

    The LLM will interview you about your profile and update the identity
    document with what it learns. Existing information is preserved.
    """
    _chat_identity(cliver)


@identity.command(name="clear", help="Clear the identity profile")
@pass_cliver
def clear_identity(cliver: Cliver):
    """Clear the identity profile."""
    _clear_identity(cliver)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _show_identity(cliver: Cliver) -> None:
    """Display the current identity document."""
    content = cliver.agent_profile.load_identity()
    if not content:
        cliver.output("No identity profile set.")
        cliver.output("Run '/identity chat' to build your profile through conversation.")
        return
    cliver.output(content)


def _build_identity_prompt(cliver: Cliver) -> str:
    """Build a prompt that guides the LLM to interview the user."""
    current = cliver.agent_profile.load_identity()

    if current:
        return (
            "I want to update my identity profile. Here is my current profile:\n\n"
            f"{current}\n\n"
            "Please ask me a few questions to update or add to this profile. "
            "After I answer, use the Identity tool to save the updated profile. "
            "Keep all existing information and add the new details."
        )
    else:
        return (
            "I want to set up my identity profile. "
            "Please ask me a few questions to learn about me — "
            "my name, role, location, preferences for how you should respond, "
            "and anything else that would help you assist me better. "
            "After I answer, use the Identity tool to save my profile."
        )
