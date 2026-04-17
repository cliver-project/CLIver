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


@click.group(name="permissions", help="Manage persistent permission rules", invoke_without_command=True)
@pass_cliver
@click.pass_context
def permissions(ctx, cliver: Cliver):
    """View or manage persistent permission rules."""
    if ctx.invoked_subcommand is None:
        # Default: show rules
        ctx.invoke(show_rules)


@permissions.command(name="rules", help="Show all loaded permission rules")
@pass_cliver
def show_rules(cliver: Cliver):
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


@permissions.command(name="mode", help="Show or set permission mode")
@click.argument("new_mode", required=False, type=click.Choice(["default", "auto-edit", "yolo"]))
@pass_cliver
def set_mode(cliver: Cliver, new_mode: str):
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


@permissions.command(name="add", help="Add a permission rule interactively")
@pass_cliver
def add_rule(cliver: Cliver):
    """Interactive permission builder."""
    pm = cliver.permission_manager
    console = cliver.console

    console.print("[bold]Permission Builder[/bold]")
    console.print("[dim]─────────────────[/dim]")

    # 1. Tool pattern
    console.print("\n[bold]1. Tool pattern[/bold]")
    console.print("   [dim]Examples: Read, github#.*, Bash, .*[/dim]")
    try:
        tool = input("   > ").strip()
    except (EOFError, KeyboardInterrupt):
        console.print("[yellow]Cancelled.[/yellow]")
        return
    if not tool:
        console.print("[yellow]Cancelled.[/yellow]")
        return

    # 2. Resource constraint
    console.print("\n[bold]2. Resource constraint[/bold] [dim](optional, leave empty for all)[/dim]")
    console.print("   [dim]Examples: /data/**, git *, https://api.github.com/**[/dim]")
    try:
        resource = input("   > ").strip() or None
    except (EOFError, KeyboardInterrupt):
        console.print("[yellow]Cancelled.[/yellow]")
        return

    # 3. Action
    console.print("\n[bold]3. Action[/bold]")
    try:
        action_input = input("   [a]llow or [d]eny? > ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        console.print("[yellow]Cancelled.[/yellow]")
        return
    if action_input in ("a", "allow"):
        action = PermissionAction.ALLOW
    elif action_input in ("d", "deny"):
        action = PermissionAction.DENY
    else:
        console.print("[yellow]Invalid action. Cancelled.[/yellow]")
        return

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


@permissions.command(name="remove", help="Remove a permission rule by index")
@click.argument("index", type=int)
@pass_cliver
def remove_rule(cliver: Cliver, index: int):
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

    try:
        confirm = input("  Confirm? [y/n] > ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        cliver.output("[yellow]Cancelled.[/yellow]")
        return
    if confirm not in ("y", "yes"):
        cliver.output("[yellow]Cancelled.[/yellow]")
        return

    pm.remove_rule(index)
    cliver.output("[green]Rule removed.[/green]")


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

    try:
        choice = input("  > ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        console.print("[yellow]Cancelled.[/yellow]")
        return None

    if choice in ("g", "global"):
        return "global"
    elif choice in ("l", "local"):
        return "local"
    else:
        console.print("[yellow]Invalid choice. Cancelled.[/yellow]")
        return None


def _shorten_path(path: str) -> str:
    """Shorten a file path for display."""
    if "/.config/cliver/" in path:
        return "global"
    elif "/.cliver/" in path:
        return "local"
    return path
