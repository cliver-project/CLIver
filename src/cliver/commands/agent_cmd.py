"""
Agent management commands.

List, show, create, edit, remove, and switch agents.
Agents are named personas with role, system prompt, model, and skills.
"""

import click

from cliver.cli import Cliver, pass_cliver
from cliver.commands import click_help, wants_help


def _list_agents(cliver: Cliver):
    cfg = cliver.config_manager.config
    cfg.ensure_default_agent()
    if not cfg.agents:
        cliver.output("No agents configured.")
        return
    cliver.output("Configured agents:\n")
    for name, ac in cfg.agents.items():
        default = " (default)" if name == cfg.default_agent else ""
        desc = ac.description or ""
        model = ac.model or "(default model)"
        skills = ", ".join(ac.skills) if ac.skills else "none"
        cliver.output(f"  **{name}**{default}")
        if desc:
            cliver.output(f"    {desc}")
        cliver.output(f"    Model: {model} | Skills: {skills}")
        cliver.output("")


def _show_agent(cliver: Cliver, name: str):
    cfg = cliver.config_manager.config
    ac = cfg.agents.get(name)
    if not ac:
        cliver.output(f"Agent '{name}' not found.")
        return
    default = " (default)" if name == cfg.default_agent else ""
    cliver.output(f"**{name}**{default}\n")
    if ac.description:
        cliver.output(f"Description: {ac.description}")
    if ac.role:
        cliver.output(f"Role: {ac.role}")
    if ac.system_prompt:
        cliver.output(f"System Prompt:\n{ac.system_prompt}")
    cliver.output(f"Model: {ac.model or '(default)'}")
    cliver.output(f"Skills: {', '.join(ac.skills) if ac.skills else 'none'}")


def _use_agent(cliver: Cliver, name: str):
    cfg = cliver.config_manager.config
    ac = cfg.agents.get(name)
    if not ac:
        cliver.output(f"Agent '{name}' not found.")
        return
    cfg.default_agent = name
    cliver.config_manager._save_config()
    cliver.output(f"Switched to agent **{name}**.")


def _add_agent(cliver: Cliver, name: str, description: str, role: str):
    from cliver.config import AgentConfig

    cfg = cliver.config_manager.config
    if name in cfg.agents:
        cliver.output(f"Agent '{name}' already exists.")
        return
    cfg.agents[name] = AgentConfig(
        description=description or None,
        role=role or None,
    )
    cliver.config_manager._save_config()
    cliver.output(f"Agent **{name}** created.")


def _remove_agent(cliver: Cliver, name: str):
    cfg = cliver.config_manager.config
    if name not in cfg.agents:
        cliver.output(f"Agent '{name}' not found.")
        return
    if name == cfg.default_agent:
        cliver.output("Cannot remove the default agent.")
        return
    del cfg.agents[name]
    if name == cfg.default_agent:
        cfg.default_agent = next(iter(cfg.agents))
    cliver.config_manager._save_config()
    cliver.output(f"Agent '{name}' removed.")


# TUI dispatch entry point

_SUBCOMMANDS: dict[str, click.Command] = {}


def dispatch(cliver: Cliver, args: str):
    """Manage agents — list, show, use, add, remove."""
    parts = args.strip().split(None, 1) if args.strip() else []
    sub = parts[0] if parts else "list"
    rest = parts[1].strip() if len(parts) > 1 else ""

    if sub in ("--help", "-h", "help"):
        cliver.output(click_help(agent_group, "/agent"))
        return

    if sub in _SUBCOMMANDS and wants_help(rest):
        cliver.output(click_help(_SUBCOMMANDS[sub], f"/agent {sub}"))
        return

    if sub == "list":
        _list_agents(cliver)
    elif sub == "show":
        if not rest:
            cliver.output("Usage: /agent show <name>")
            return
        _show_agent(cliver, rest)
    elif sub == "use":
        if not rest:
            cliver.output("Usage: /agent use <name>")
            return
        _use_agent(cliver, rest)
    elif sub == "add":
        tokens = rest.split(None, 1)
        name = tokens[0] if tokens else ""
        if not name:
            cliver.output("Usage: /agent add <name> [description]")
            return
        desc = tokens[1] if len(tokens) > 1 else ""
        _add_agent(cliver, name, desc, "")
    elif sub == "remove":
        if not rest:
            cliver.output("Usage: /agent remove <name>")
            return
        _remove_agent(cliver, rest)
    else:
        cliver.output(f"Unknown: /agent {sub}")
        cliver.output("Available: list, show, use, add, remove")


# Click wrappers


@click.group(
    name="agent",
    help="Manage agents — list, show, use, add, remove",
    invoke_without_command=True,
)
@pass_cliver
@click.pass_context
def agent_group(ctx, cliver: Cliver):
    if ctx.invoked_subcommand is None:
        _list_agents(cliver)


@agent_group.command(name="list", help="List all configured agents")
@pass_cliver
def list_cmd(cliver: Cliver):
    _list_agents(cliver)


@agent_group.command(name="show", help="Show agent details")
@click.argument("name")
@pass_cliver
def show(cliver: Cliver, name: str):
    _show_agent(cliver, name)


@agent_group.command(name="use", help="Switch the default agent")
@click.argument("name")
@pass_cliver
def use(cliver: Cliver, name: str):
    _use_agent(cliver, name)


@agent_group.command(name="add", help="Create a new agent")
@click.argument("name")
@click.option("--description", "-d", default="", help="Agent description")
@click.option("--role", "-r", default="", help="Agent role text")
@pass_cliver
def add(cliver: Cliver, name: str, description: str, role: str):
    _add_agent(cliver, name, description, role)


@agent_group.command(name="remove", help="Remove an agent")
@click.argument("name")
@pass_cliver
def remove(cliver: Cliver, name: str):
    _remove_agent(cliver, name)


agent_cmd = agent_group

_SUBCOMMANDS.update(
    {
        "list": list_cmd,
        "show": show,
        "use": use,
        "add": add,
        "remove": remove,
    }
)
