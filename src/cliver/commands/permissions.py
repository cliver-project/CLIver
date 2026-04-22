"""
Persistent permission rule management commands.

Manages rules saved to cliver-settings.yaml files (global and local).
Session-scoped grants are managed via /session permission instead.
"""

import click
from rich import box
from rich.table import Table

from cliver.cli import Cliver, pass_cliver
from cliver.permissions import PermissionAction, PermissionMode, PermissionRule

# Business logic (plain functions — no Click, no async)


def _show_rules(cliver: Cliver):
    """Display all permission rules with their source file."""
    pm = cliver.permission_manager
    if not pm.rules:
        cliver.output("[dim]No permission rules configured.[/dim]")
        _show_mode_info(cliver)
        return

    table = Table(title="Permission Rules", box=box.SQUARE)
    table.add_column("#", style="dim", max_width=3)
    table.add_column("Tool", style="green")
    table.add_column("Resource", style="cyan")
    table.add_column("Action", style="bold")
    table.add_column("Source", style="dim")

    for i, rule in enumerate(pm.rules):
        source = pm._rule_sources[i] if i < len(pm._rule_sources) else "?"
        # Shorten source path for display
        source_short = _shorten_path(source)
        action_style = "green" if rule.action == PermissionAction.ALLOW else "red"
        table.add_row(
            str(i),
            rule.tool,
            rule.resource or "[dim]—[/dim]",
            f"[{action_style}]{rule.action.value}[/{action_style}]",
            source_short,
        )

    cliver.output(table)
    _show_mode_info(cliver)


def _set_mode(cliver: Cliver, new_mode: str = None):
    """Show current mode or set a new one (saves to file)."""
    pm = cliver.permission_manager
    if new_mode is None:
        _show_mode_info(cliver)
        return

    mode = PermissionMode(new_mode)
    target = _prompt_save_target(cliver)
    if target is None:
        return
    pm.save_mode(mode, target)
    cliver.output(f"[green]Permission mode set to '{new_mode}' ({target})[/green]")


def _add_rule(cliver: Cliver):
    """Interactive permission builder."""
    pm = cliver.permission_manager
    console = cliver.console

    console.print("[bold]Permission Builder[/bold]")
    console.print("[dim]─────────────────[/dim]")

    # 1. Tool pattern
    console.print("\n[bold]1. Tool pattern[/bold]")
    console.print("   [dim]Examples: Read, github#.*, Bash, .*[/dim]")
    tool = cliver.ui.ask_input("   > ")
    if not tool:
        console.print("[yellow]Cancelled.[/yellow]")
        return

    # 2. Resource constraint
    console.print("\n[bold]2. Resource constraint[/bold] [dim](optional, leave empty for all)[/dim]")
    console.print("   [dim]Examples: /data/**, git *, https://api.github.com/**[/dim]")
    resource = cliver.ui.ask_input("   > ") or None

    # 3. Action
    console.print("\n[bold]3. Action[/bold]")
    action_input = cliver.ui.ask_input("   [a]llow or [d]eny? > ", choices=["a", "allow", "d", "deny"])
    if not action_input:
        console.print("[yellow]Cancelled.[/yellow]")
        return
    if action_input.lower() in ("a", "allow"):
        action = PermissionAction.ALLOW
    else:
        action = PermissionAction.DENY

    # 4. Save target
    target = _prompt_save_target(cliver)
    if target is None:
        return

    rule = PermissionRule(tool=tool, resource=resource, action=action)
    pm.save_rule(rule, target)

    resource_str = f" on [cyan]{resource}[/cyan]" if resource else ""
    action_style = "green" if action == PermissionAction.ALLOW else "red"
    console.print(
        f"\n[green]✔[/green] Rule added: [{action_style}]{action.value}[/{action_style}] "
        f"[bold]{tool}[/bold]{resource_str}"
    )
    console.print(f"  Saved to {target} settings.")


def _remove_rule(cliver: Cliver, index: int):
    """Remove a rule by its index (shown in /permissions rules)."""
    pm = cliver.permission_manager
    if index < 0 or index >= len(pm.rules):
        cliver.output(f"[red]Invalid index {index}. Use /permissions rules to see available rules.[/red]")
        return

    rule = pm.rules[index]
    source = pm._rule_sources[index] if index < len(pm._rule_sources) else "?"
    resource_str = f" on {rule.resource}" if rule.resource else ""
    cliver.output(f"Removing rule #{index}: {rule.action.value} {rule.tool}{resource_str}")
    cliver.output(f"  Source: {source}")

    response = cliver.ui.ask_input("  Confirm? [y/n] > ", choices=["y", "yes", "n", "no"])
    if response.lower() not in ("y", "yes"):
        cliver.output("[yellow]Cancelled.[/yellow]")
        return

    pm.remove_rule(index)
    cliver.output("[green]Rule removed.[/green]")


# TUI dispatch entry point


def dispatch(cliver: Cliver, args: str):
    """Manage tool permission rules — rules, mode, add, remove."""
    parts = args.strip().split(None, 1) if args.strip() else []
    sub = parts[0] if parts else "rules"  # default subcommand
    rest = parts[1] if len(parts) > 1 else ""

    if sub == "rules":
        _show_rules(cliver)
    elif sub == "mode":
        if rest and rest.strip() in ("default", "auto-edit", "yolo"):
            _set_mode(cliver, rest.strip())
        else:
            _set_mode(cliver)
    elif sub == "add":
        _add_rule(cliver)
    elif sub == "remove":
        if not rest:
            cliver.output("[yellow]Usage: /permissions remove <index>[/yellow]")
            return
        try:
            index = int(rest.strip())
            _remove_rule(cliver, index)
        except ValueError:
            cliver.output("[red]Index must be a number[/red]")
    elif sub in ("--help", "help"):
        cliver.output("Usage: /permissions [rules|mode|add|remove] ...")
        cliver.output("  rules              — show all permission rules")
        cliver.output("  mode [mode]        — show or set mode (default|auto-edit|yolo)")
        cliver.output("  add                — add a permission rule interactively")
        cliver.output("  remove <index>     — remove a rule by index")
    else:
        cliver.output(f"[yellow]Unknown: /permissions {sub}[/yellow]")


# --- Helpers ---


def _show_mode_info(cliver: Cliver):
    pm = cliver.permission_manager
    source = pm._mode_source or "default"
    cliver.output(f"\nPermission mode: [bold]{pm.mode.value}[/bold] [dim]({source})[/dim]")


def _prompt_save_target(cliver: Cliver) -> str | None:
    """Prompt user to choose global or local save target."""
    pm = cliver.permission_manager
    console = cliver.console

    global_path = pm._global_settings_path or "N/A"
    local_path = pm._local_settings_path or "N/A"

    console.print("\n[bold]Save to:[/bold]")
    console.print(f"  [g]lobal ({global_path})")
    console.print(f"  [l]ocal  ({local_path})")

    choice = cliver.ui.ask_input("  > ", choices=["g", "global", "l", "local"])
    if not choice:
        console.print("[yellow]Cancelled.[/yellow]")
        return None

    if choice.lower() in ("g", "global"):
        return "global"
    else:
        return "local"


def _shorten_path(path: str) -> str:
    """Shorten a file path for display."""
    if "/.config/cliver/" in path:
        return "global"
    elif "/.cliver/" in path:
        return "local"
    return path


# Click wrappers (thin — just call logic functions)


@click.group(name="permissions", help="Manage persistent permission rules", invoke_without_command=True)
@pass_cliver
@click.pass_context
def permissions(ctx, cliver: Cliver):
    """View or manage persistent permission rules."""
    if ctx.invoked_subcommand is None:
        # Default: show rules
        _show_rules(cliver)


@permissions.command(name="rules", help="Show all loaded permission rules")
@pass_cliver
def show_rules(cliver: Cliver):
    _show_rules(cliver)


@permissions.command(name="mode", help="Show or set permission mode")
@click.argument("new_mode", required=False, type=click.Choice(["default", "auto-edit", "yolo"]))
@pass_cliver
def set_mode(cliver: Cliver, new_mode: str):
    _set_mode(cliver, new_mode)


@permissions.command(name="add", help="Add a permission rule interactively")
@pass_cliver
def add_rule(cliver: Cliver):
    _add_rule(cliver)


@permissions.command(name="remove", help="Remove a permission rule by index")
@click.argument("index", type=int)
@pass_cliver
def remove_rule(cliver: Cliver, index: int):
    _remove_rule(cliver, index)
