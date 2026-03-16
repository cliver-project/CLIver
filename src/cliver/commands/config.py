import click
import yaml

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
    """Show the current configuration with sensitive values masked."""
    try:
        config_data = cliver.config_manager.config
        if config_data:
            data = config_data.model_dump()
            _mask_secrets(data)
            output = yaml.dump(
                data,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            )
            cliver.console.print(output)
        else:
            cliver.console.print("No configuration found.")
    except Exception as e:
        cliver.console.print(f"[red]Error showing configuration: {e}[/red]")


def _mask_secrets(data, keys_to_mask=("api_key",)):
    """Recursively mask sensitive values in a config dict.

    Keyring references (keyring:service:key) are shown as-is since
    they don't contain the actual secret. Plain text secrets are masked.
    """
    if isinstance(data, dict):
        for key, value in data.items():
            if key in keys_to_mask and isinstance(value, str):
                if value.startswith("keyring:"):
                    pass  # keyring references are safe to display
                else:
                    # Mask plain text: show first 3 and last 3 chars
                    if len(value) > 8:
                        data[key] = value[:3] + "***" + value[-3:]
                    else:
                        data[key] = "***"
            elif isinstance(value, (dict, list)):
                _mask_secrets(value, keys_to_mask)
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, (dict, list)):
                _mask_secrets(item, keys_to_mask)


# noinspection PyUnresolvedReferences
@config.command(name="path", help="Show configuration file path")
@pass_cliver
def show_config_path(cliver: Cliver):
    """Show the path to the configuration file."""
    config_path = cliver.config_manager.config_file
    cliver.console.print(f"Configuration file path: {config_path}")
