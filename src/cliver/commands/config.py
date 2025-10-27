import click

from cliver.cli import Cliver, pass_cliver


@click.group(name="config", help="Manage configuration settings.")
@click.pass_context
def config(ctx: click.Context):
    """
    Configuration command group.
    This group contains commands to manage configuration settings.
    """
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())
        ctx.exit()


def post_group():
    pass


# noinspection PyUnresolvedReferences
@config.command(name="validate", help="Validate configuration")
@pass_cliver
def validate_config(cliver: Cliver):
    """Validate the current configuration."""
    try:
        # Check if config is valid by attempting to load it
        config_manager = cliver.config_manager
        if config_manager.config:
            cliver.console.print("[green]✓ Configuration is valid[/green]")
        else:
            cliver.console.print("[red]✗ Configuration is not valid[/red]")
    except Exception as e:
        cliver.console.print(f"[red]✗ Configuration validation error: {e}[/red]")


# noinspection PyUnresolvedReferences
@config.command(name="show", help="Show current configuration")
@pass_cliver
def show_config(cliver: Cliver):
    """Show the current configuration."""
    try:
        config_data = cliver.config_manager.config
        if config_data:
            cliver.console.print_json(data=config_data)
        else:
            cliver.console.print("No configuration found.")
    except Exception as e:
        cliver.console.print(f"[red]Error showing configuration: {e}[/red]")


# noinspection PyUnresolvedReferences
@config.command(name="path", help="Show configuration file path")
@pass_cliver
def show_config_path(cliver: Cliver):
    """Show the path to the configuration file."""
    config_path = cliver.config_manager.config_file
    cliver.console.print(f"Configuration file path: {config_path}")
