import click
from rich import box
from rich.table import Table

from cliver.cli import Cliver, pass_cliver
from cliver.config import RateLimitConfig
from cliver.model_capabilities import ProviderEnum


def _parse_rate_limit(value: str) -> RateLimitConfig:
    parts = value.split("/", 1)
    if len(parts) != 2:
        raise click.BadParameter(f"Expected 'requests/period' (e.g. '5000/5h'), got: {value}")
    try:
        requests = int(parts[0].strip())
    except ValueError as e:
        raise click.BadParameter(f"Requests must be an integer, got: {parts[0].strip()}") from e
    period = parts[1].strip()
    if not period:
        raise click.BadParameter("Period cannot be empty")
    return RateLimitConfig(requests=requests, period=period)


@click.group(name="provider", help="Manage LLM providers")
@click.pass_context
def provider(ctx: click.Context):
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())
        ctx.exit()


@provider.command(name="list", help="List configured providers")
@pass_cliver
def list_providers(cliver: Cliver):
    providers = cliver.config_manager.list_providers()
    if not providers:
        cliver.output("No providers configured.")
        return
    table = Table(title="Configured Providers", box=box.SQUARE)
    table.add_column("Name", style="green")
    table.add_column("Type", style="cyan")
    table.add_column("API URL", style="dim")
    table.add_column("API Key", style="dim")
    table.add_column("Rate Limit", style="yellow")
    table.add_column("Image URL", style="dim")

    for name, prov in providers.items():
        api_key_display = "***" if prov.api_key else "-"
        rl = f"{prov.rate_limit.requests}/{prov.rate_limit.period}" if prov.rate_limit else "-"
        table.add_row(name, prov.type, prov.api_url, api_key_display, rl, prov.image_url or "-")
    cliver.output(table)


@provider.command(name="add", help="Add a provider")
@click.option("--name", "-n", type=str, required=True, help="Provider name")
@click.option(
    "--type",
    "-t",
    "ptype",
    type=click.Choice([p.value for p in ProviderEnum]),
    required=True,
    help="Provider type",
)
@click.option("--api-url", "-u", type=str, required=True, help="API base URL")
@click.option("--api-key", "-k", type=str, default=None, help="API key")
@click.option(
    "--rate-limit",
    "-r",
    type=str,
    default=None,
    help="Rate limit as 'requests/period' (e.g. '5000/5h')",
)
@click.option("--image-url", type=str, default=None, help="Image generation endpoint URL")
@click.option("--audio-url", type=str, default=None, help="Audio generation endpoint URL")
@pass_cliver
def add_provider(
    cliver: Cliver, name: str, ptype: str, api_url: str, api_key: str, rate_limit: str, image_url: str, audio_url: str
):
    if name in cliver.config_manager.list_providers():
        cliver.output(f"Provider '{name}' already exists. Use 'provider set' to update.")
        return
    rl = _parse_rate_limit(rate_limit) if rate_limit else None
    cliver.config_manager.add_or_update_provider(name, ptype, api_url, api_key, rl, image_url, audio_url)
    cliver.output(f"Added provider: {name}")


@provider.command(name="set", help="Update a provider")
@click.option("--name", "-n", type=str, required=True, help="Provider name")
@click.option(
    "--type",
    "-t",
    "ptype",
    type=click.Choice([p.value for p in ProviderEnum]),
    default=None,
    help="Provider type",
)
@click.option("--api-url", "-u", type=str, default=None, help="API base URL")
@click.option("--api-key", "-k", type=str, default=None, help="API key")
@click.option(
    "--rate-limit",
    "-r",
    type=str,
    default=None,
    help="Rate limit as 'requests/period' (e.g. '5000/5h')",
)
@click.option("--image-url", type=str, default=None, help="Image generation endpoint URL")
@click.option("--audio-url", type=str, default=None, help="Audio generation endpoint URL")
@pass_cliver
def set_provider(
    cliver: Cliver, name: str, ptype: str, api_url: str, api_key: str, rate_limit: str, image_url: str, audio_url: str
):
    existing = cliver.config_manager.list_providers().get(name)
    if not existing:
        cliver.output(f"Provider '{name}' not found.")
        return
    rl = _parse_rate_limit(rate_limit) if rate_limit else existing.rate_limit
    cliver.config_manager.add_or_update_provider(
        name,
        ptype or existing.type,
        api_url or existing.api_url,
        api_key if api_key is not None else existing.api_key,
        rl,
        image_url,
        audio_url,
    )
    cliver.output(f"Updated provider: {name}")


@provider.command(name="remove", help="Remove a provider")
@click.option("--name", "-n", type=str, required=True, help="Provider name")
@pass_cliver
def remove_provider(cliver: Cliver, name: str):
    try:
        if cliver.config_manager.remove_provider(name):
            cliver.output(f"Removed provider: {name}")
        else:
            cliver.output(f"Provider '{name}' not found.")
    except ValueError as e:
        cliver.output(f"[red]{e}[/red]")
