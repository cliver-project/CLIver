"""
Scenario management commands.

Install, list, and remove community scenario templates.
"""

import click

from cliver.cli import Cliver, pass_cliver
from cliver.commands import click_help, wants_help


def _get_installer(cliver: Cliver):
    from pathlib import Path

    from cliver.project.scenario_registry import ScenarioRegistry
    from cliver.scenario_installer import ScenarioInstaller

    user_dir = cliver.agent_profile.config_dir / "scenarios"
    builtin_dir = Path(__file__).parent.parent / "scenarios"
    registry = ScenarioRegistry([d for d in [builtin_dir, user_dir] if d.exists() or d == user_dir])
    registry.set_builtin_dir(builtin_dir)
    return ScenarioInstaller(user_dir, registry), registry


def _list_scenarios(cliver: Cliver):
    _, registry = _get_installer(cliver)
    scenarios = registry.list_scenarios()
    if not scenarios:
        cliver.output("No scenarios installed.")
        return
    lines = []
    for s in scenarios:
        source_badge = f"[{s.source}]"
        tags = ", ".join(s.tags) if s.tags else ""
        lines.append(f"  {s.id:24s} {s.display_name:24s} {source_badge:10s} {tags}")
    cliver.output("Installed scenarios:\n" + "\n".join(lines))


def _install_scenario(cliver: Cliver, source: str):
    installer, _ = _get_installer(cliver)
    try:
        display_name = installer.install_from_github(source)
        cliver.output(f"Installed '{display_name}' successfully.")
    except (ValueError, RuntimeError) as e:
        cliver.output(f"Install failed: {e}")


def _remove_scenario(cliver: Cliver, name: str):
    installer, _ = _get_installer(cliver)
    try:
        if installer.remove(name):
            cliver.output(f"Scenario '{name}' removed.")
        else:
            cliver.output(f"Scenario '{name}' not found in installed scenarios.")
    except ValueError as e:
        cliver.output(str(e))


def _info_scenario(cliver: Cliver, name: str):
    _, registry = _get_installer(cliver)
    s = registry.get_scenario(name)
    if not s:
        cliver.output(f"Scenario '{name}' not found.")
        return
    template = registry.get_template(name)
    cell_count = len(template.get("cells", [])) if template else 0
    lines = [
        f"Name:         {s.display_name}",
        f"ID:           {s.id}",
        f"Source:        {s.source}",
        f"Description:  {s.description}",
        f"Tags:         {', '.join(s.tags) if s.tags else 'none'}",
        f"Agents:       {', '.join(s.agent_requirements) if s.agent_requirements else 'any'}",
        f"Cells:        {cell_count}",
    ]
    cliver.output("\n".join(lines))


_SUBCOMMANDS: dict[str, click.Command] = {}


def dispatch(cliver: Cliver, args: str):
    """Manage scenario templates — list, install, remove, info."""
    parts = args.strip().split(None, 1) if args.strip() else []
    sub = parts[0] if parts else "list"
    rest = parts[1].strip() if len(parts) > 1 else ""

    if sub in ("--help", "-h", "help"):
        cliver.output(click_help(scenario_group, "/scenario"))
        return

    if sub in _SUBCOMMANDS and wants_help(rest):
        cliver.output(click_help(_SUBCOMMANDS[sub], f"/scenario {sub}"))
        return

    if sub == "list":
        _list_scenarios(cliver)
    elif sub == "install":
        if not rest:
            cliver.output("Usage: /scenario install github:user/repo")
            return
        _install_scenario(cliver, rest)
    elif sub == "remove":
        if not rest:
            cliver.output("Usage: /scenario remove <name>")
            return
        _remove_scenario(cliver, rest)
    elif sub == "info":
        if not rest:
            cliver.output("Usage: /scenario info <name>")
            return
        _info_scenario(cliver, rest)
    else:
        cliver.output(f"Unknown: /scenario {sub}")
        cliver.output("Available: list, install, remove, info")


@click.group(
    name="scenario",
    help="Manage scenario templates — list, install, remove, info",
    invoke_without_command=True,
)
@pass_cliver
@click.pass_context
def scenario_group(ctx, cliver: Cliver):
    if ctx.invoked_subcommand is None:
        _list_scenarios(cliver)


@scenario_group.command(name="list", help="List all installed scenarios")
@pass_cliver
def list_cmd(cliver: Cliver):
    _list_scenarios(cliver)


@scenario_group.command(name="install", help="Install a scenario from GitHub")
@click.argument("source")
@pass_cliver
def install_cmd(cliver: Cliver, source: str):
    _install_scenario(cliver, source)


@scenario_group.command(name="remove", help="Remove an installed scenario")
@click.argument("name")
@pass_cliver
def remove_cmd(cliver: Cliver, name: str):
    _remove_scenario(cliver, name)


@scenario_group.command(name="info", help="Show scenario details")
@click.argument("name")
@pass_cliver
def info_cmd(cliver: Cliver, name: str):
    _info_scenario(cliver, name)


scenario = scenario_group

_SUBCOMMANDS.update(
    {
        "list": list_cmd,
        "install": install_cmd,
        "remove": remove_cmd,
        "info": info_cmd,
    }
)
