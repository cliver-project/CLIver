"""
Status bar configuration commands.

Configure what's displayed in the bottom toolbar during interactive sessions.
"""

import click

from cliver.cli import Cliver, pass_cliver


@click.group(
    name="statusbar",
    help="Configure the status bar",
    invoke_without_command=True,
)
@pass_cliver
@click.pass_context
def statusbar(ctx, cliver: Cliver):
    """View or configure the bottom status bar."""
    if ctx.invoked_subcommand is None:
        _show_config(cliver)


@statusbar.command(name="show", help="Show current status bar configuration")
@pass_cliver
def statusbar_show(cliver: Cliver):
    _show_config(cliver)


@statusbar.command(name="set", help="Configure status bar options")
@click.option("--rss", type=str, default=None, help="RSS feed URL for scrolling news")
@click.option("--visible/--hidden", default=None, help="Show or hide the status bar")
@pass_cliver
def statusbar_set(cliver: Cliver, rss: str | None, visible: bool | None):
    """Set status bar options for the current session."""
    sb = cliver.session_options.setdefault("statusbar", {})

    if visible is not None:
        sb["visible"] = visible
        state = "visible" if visible else "hidden"
        cliver.console.print(f"  Status bar is now [bold]{state}[/bold].")

    if rss is not None:
        sb["rss_url"] = rss if rss else None
        if rss:
            cliver.console.print(f"  RSS feed set to: [cyan]{rss}[/cyan]")
            cliver.console.print("  [dim]Feed will scroll in the status bar middle section.[/dim]")
        else:
            cliver.console.print("  RSS feed cleared.")

    # Refresh the prompt session toolbar if available
    if cliver.session:
        cliver.session.app.invalidate()


@statusbar.command(name="reset", help="Reset status bar to defaults")
@pass_cliver
def statusbar_reset(cliver: Cliver):
    """Reset status bar configuration to defaults."""
    cliver.session_options.pop("statusbar", None)
    cliver.console.print("  Status bar reset to defaults.")


def _show_config(cliver: Cliver):
    """Display the current status bar configuration."""
    sb = cliver.session_options.get("statusbar", {})
    visible = sb.get("visible", True)
    rss_url = sb.get("rss_url")

    cliver.console.print("[bold]Status Bar Configuration[/bold]")
    cliver.console.print(f"  Visible:  [bold]{'yes' if visible else 'no'}[/bold]")
    cliver.console.print(f"  RSS Feed: [cyan]{rss_url or '—'}[/cyan]")
    cliver.console.print()
    cliver.console.print("[dim]  Sections: [cwd] │ [permission mode] │ [model][/dim]")
    if rss_url:
        cliver.console.print("[dim]  RSS headlines scroll in the middle section.[/dim]")
