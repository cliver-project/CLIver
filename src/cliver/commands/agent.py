"""
Agent management commands.

Create, switch, list, rename, and delete agent instances.
Each agent has its own memory, identity, sessions, and tasks.
"""

import click

from cliver.agent_profile import CliverProfile
from cliver.cli import Cliver, pass_cliver
from cliver.commands import click_help, wants_help

# Business logic (plain functions — no Click, no async)


def _show_agents(cliver: Cliver):
    """Display all agents with the active one highlighted."""
    agents = CliverProfile.list_agents(cliver.config_dir)
    current = cliver.agent_name

    cliver.output("[bold]Agent Instances[/bold]")
    if not agents:
        cliver.output(f"  [green]* {current}[/green] (active, no data yet)")
    else:
        for name in agents:
            if name == current:
                cliver.output(f"  [green]* {name}[/green] (active)")
            else:
                cliver.output(f"    {name}")
        if current not in agents:
            cliver.output(f"  [green]* {current}[/green] (active, no data yet)")
    cliver.output()
    cliver.output("[dim]  /agent create <name>  — create new agent[/dim]")
    cliver.output("[dim]  /agent switch <name>  — switch to agent[/dim]")
    cliver.output("[dim]  /agent rename <name>  — rename current agent[/dim]")
    cliver.output("[dim]  /agent delete <name>  — delete an agent[/dim]")


def _switch_agent(cliver: Cliver, name: str):
    """Switch to an agent instance. Creates it if it doesn't exist."""
    if name == cliver.agent_name:
        cliver.output(f"Already using agent [green]{name}[/green].")
        return
    cliver.switch_agent(name)
    cliver.output(f"Switched to agent [bold green]{name}[/bold green].")


def _create_agent(cliver: Cliver, name: str):
    """Create a new agent with an identity profile and switch to it."""
    existing = CliverProfile.list_agents(cliver.config_dir)
    if name in existing:
        cliver.output(f"Agent [yellow]{name}[/yellow] already exists. Use [bold]/agent switch {name}[/bold].")
        return

    cliver.output(f"\nSetting up agent [bold green]{name}[/bold green]…\n")
    cliver.output("[dim]Press Enter to skip any question.[/dim]\n")

    # Gather identity info interactively
    identity = _ask_identity(name, cliver=cliver)

    # Create the agent and write its identity
    profile = CliverProfile(name, cliver.config_dir)
    profile.ensure_dirs()
    profile.save_identity(identity)

    cliver.switch_agent(name)
    cliver.output(f"\nCreated and switched to agent [bold green]{name}[/bold green].")
    cliver.output("[dim]Update identity anytime with /identity chat[/dim]")


def _ask_identity(agent_name: str, cliver=None) -> str:
    """Interactively build an identity.md for a new agent."""

    def _ask(prompt: str, default: str = "") -> str:
        if cliver:
            value = cliver.ui.ask_input(f"  {prompt}: ")
        else:
            try:
                value = input(f"  {prompt}: ").strip()
            except (EOFError, KeyboardInterrupt):
                value = ""
        return value if value else default

    # Agent persona
    persona_style = _ask("Agent style (e.g., professional, casual, technical)", "Helpful and professional")
    persona_focus = _ask("Agent focus area (e.g., coding, research, general)", "General-purpose assistant")

    # User profile
    user_name = _ask("Your name")
    user_role = _ask("Your role (e.g., developer, researcher, student)")
    user_languages = _ask("Preferred programming languages (comma-separated)")
    user_comm = _ask("Communication preference (e.g., concise, detailed)", "Concise and clear")

    # Build markdown
    lines = [f"# Agent: {agent_name}", ""]

    lines.append("## Agent Persona")
    lines.append(f"- Name: {agent_name}")
    lines.append(f"- Style: {persona_style}")
    lines.append(f"- Focus: {persona_focus}")
    lines.append("")

    lines.append("## User Profile")
    if user_name:
        lines.append(f"- Name: {user_name}")
    if user_role:
        lines.append(f"- Role: {user_role}")
    if user_languages:
        lines.append(f"- Languages: {user_languages}")
    lines.append(f"- Communication: {user_comm}")
    lines.append("")

    return "\n".join(lines)


def _rename_agent(cliver: Cliver, new_name: str):
    """Rename the current agent, moving all its resources."""
    old_name = cliver.agent_name
    if old_name == new_name:
        cliver.output("New name is the same as the current name.")
        return

    existing = CliverProfile.list_agents(cliver.config_dir)
    if new_name in existing:
        cliver.output(f"[red]Agent '{new_name}' already exists. Choose a different name.[/red]")
        return

    try:
        cliver.agent_profile.rename(new_name)
        cliver.switch_agent(new_name)
        cliver.output(f"Renamed agent [yellow]{old_name}[/yellow] to [bold green]{new_name}[/bold green].")
    except Exception as e:
        cliver.output(f"[red]Failed to rename: {e}[/red]")


def _delete_agent(cliver: Cliver, name: str):
    """Delete an agent instance after confirmation."""
    import shutil

    if name == cliver.agent_name:
        cliver.output(f"[red]Cannot delete the active agent '{name}'. Switch to another agent first.[/red]")
        return

    agent_dir = cliver.config_dir / "agents" / name
    if not agent_dir.exists():
        cliver.output(f"Agent '{name}' not found.")
        return

    # Confirm deletion
    cliver.output(f"[bold yellow]Delete agent '{name}' and all its data?[/bold yellow]")
    cliver.output(f"  Directory: {agent_dir}")

    response = cliver.ui.ask_input("  Type 'yes' to confirm: ")
    if response.lower() not in ("y", "yes"):
        cliver.output("Cancelled.")
        return

    shutil.rmtree(agent_dir)
    cliver.output(f"Deleted agent [red]{name}[/red].")


# TUI dispatch entry point


_SUBCOMMANDS: dict[str, click.Command] = {}


def dispatch(cliver: Cliver, args: str):
    """Manage agent instances — list, switch, create, delete."""
    parts = args.strip().split(None, 1) if args.strip() else []
    sub = parts[0] if parts else "list"  # default subcommand
    rest = parts[1] if len(parts) > 1 else ""

    if sub in ("--help", "-h", "help"):
        cliver.output(click_help(agent, "/agent"))
        return

    if sub in _SUBCOMMANDS and wants_help(rest):
        cliver.output(click_help(_SUBCOMMANDS[sub], f"/agent {sub}"))
        return

    if sub == "list":
        _show_agents(cliver)
    elif sub == "switch":
        if not rest:
            cliver.output(click_help(_SUBCOMMANDS["switch"], "/agent switch"))
            return
        _switch_agent(cliver, rest.strip())
    elif sub == "create":
        if not rest:
            cliver.output(click_help(_SUBCOMMANDS["create"], "/agent create"))
            return
        _create_agent(cliver, rest.strip())
    elif sub == "rename":
        if not rest:
            cliver.output(click_help(_SUBCOMMANDS["rename"], "/agent rename"))
            return
        _rename_agent(cliver, rest.strip())
    elif sub == "delete":
        if not rest:
            cliver.output(click_help(_SUBCOMMANDS["delete"], "/agent delete"))
            return
        _delete_agent(cliver, rest.strip())
    else:
        cliver.output(f"[yellow]Unknown: /agent {sub}[/yellow]")


# Click wrappers (thin — just call logic functions)


@click.group(
    name="agent",
    help="Manage agent instances. Each agent has its own memory, identity, sessions, and tasks",
    invoke_without_command=True,
)
@pass_cliver
@click.pass_context
def agent(ctx, cliver: Cliver):
    """View or manage agent instances."""
    if ctx.invoked_subcommand is None:
        _show_agents(cliver)


@agent.command(name="list", help="List all agent instances with the active one highlighted")
@pass_cliver
def list_agents(cliver: Cliver):
    """List all agent instances."""
    _show_agents(cliver)


@agent.command(name="switch", help="Switch to a different agent instance (creates it if it does not exist)")
@click.argument("name")
@pass_cliver
def switch_agent(cliver: Cliver, name: str):
    _switch_agent(cliver, name)


@agent.command(name="create", help="Create a new agent with interactive identity setup and switch to it")
@click.argument("name")
@pass_cliver
def create_agent(cliver: Cliver, name: str):
    _create_agent(cliver, name)


@agent.command(name="rename", help="Rename the currently active agent, moving all its data files")
@click.argument("new_name")
@pass_cliver
def rename_agent(cliver: Cliver, new_name: str):
    _rename_agent(cliver, new_name)


@agent.command(
    name="delete",
    help="Delete an agent and all its data (requires confirmation, cannot delete active agent)",
)
@click.argument("name")
@pass_cliver
def delete_agent(cliver: Cliver, name: str):
    _delete_agent(cliver, name)


# Populate subcommand map for dispatch help generation.
_SUBCOMMANDS.update(
    {
        "list": list_agents,
        "switch": switch_agent,
        "create": create_agent,
        "rename": rename_agent,
        "delete": delete_agent,
    }
)
