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


@click.group(name="provider", help="Manage LLM provider endpoints (API URLs, keys, rate limits)")
@click.pass_context
def provider(ctx: click.Context):
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())
        ctx.exit()


# ---------------------------------------------------------------------------
# Business logic functions
# ---------------------------------------------------------------------------


def _list_providers(cliver: Cliver):
    """List all configured providers."""
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


def _add_provider(
    cliver: Cliver, name: str, ptype: str, api_url: str, api_key: str, rate_limit: str, image_url: str, audio_url: str
):
    """Add a new provider."""
    if name in cliver.config_manager.list_providers():
        cliver.output(f"Provider '{name}' already exists. Use 'provider set' to update.")
        return
    rl = _parse_rate_limit(rate_limit) if rate_limit else None
    cliver.config_manager.add_or_update_provider(name, ptype, api_url, api_key, rl, image_url, audio_url)
    cliver.output(f"Added provider: {name}")


def _set_provider(
    cliver: Cliver, name: str, ptype: str, api_url: str, api_key: str, rate_limit: str, image_url: str, audio_url: str
):
    """Update an existing provider."""
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


def _remove_provider(cliver: Cliver, name: str):
    """Remove a provider."""
    try:
        if cliver.config_manager.remove_provider(name):
            cliver.output(f"Removed provider: {name}")
        else:
            cliver.output(f"Provider '{name}' not found.")
    except ValueError as e:
        cliver.output(f"[red]{e}[/red]")


# ---------------------------------------------------------------------------
# Dispatch function
# ---------------------------------------------------------------------------


def dispatch(cliver: Cliver, args: str):
    """Manage LLM providers — list, add, set, remove."""
    parts = args.strip().split(None, 1) if args.strip() else []
    sub = parts[0] if parts else "list"

    if sub == "list":
        _list_providers(cliver)
    elif sub in ("--help", "help"):
        cliver.output("Manage LLM providers (API endpoints). Providers host one or more models.")
        cliver.output("")
        cliver.output("Usage: /provider [list|add|set|remove] [options]")
        cliver.output("")
        cliver.output("Subcommands:")
        cliver.output("  list    — List all configured providers with type, URL, and rate limit.")
        cliver.output("            No parameters.")
        cliver.output("")
        cliver.output("  add     — Add a new provider endpoint.")
        cliver.output("    --name, -n       STRING (required) — Unique provider name (e.g. 'deepseek').")
        cliver.output("    --type, -t       CHOICE (required) — Provider type (e.g. openai, ollama, vllm).")
        cliver.output("    --api-url, -u    STRING (required) — API base URL (e.g. 'https://api.deepseek.com').")
        cliver.output("    --api-key, -k    STRING (optional) — API key. Supports Jinja2 templates:")
        cliver.output("                       {{ env.DEEPSEEK_API_KEY }} or {{ keyring('cliver','key') }}.")
        cliver.output("    --rate-limit, -r STRING (optional) — Rate limit as 'requests/period'")
        cliver.output("                       (e.g. '5000/5h', '100/1m').")
        cliver.output("    --image-url      STRING (optional) — Image generation endpoint URL.")
        cliver.output("    --audio-url      STRING (optional) — Audio generation endpoint URL.")
        cliver.output("    Example: /provider add --name deepseek --type openai --api-url https://api.deepseek.com")
        cliver.output("")
        cliver.output("  set     — Update an existing provider (only provided values are changed).")
        cliver.output("    --name, -n  STRING (required) — Provider name to update. Must exist.")
        cliver.output("    (same options as 'add', all optional)")
        cliver.output("    Example: /provider set --name deepseek --rate-limit 5000/5h")
        cliver.output("")
        cliver.output("  remove  — Remove a provider. Fails if models still reference it.")
        cliver.output("    --name, -n  STRING (required) — Provider name to remove.")
        cliver.output("    Example: /provider remove --name old-provider")
        cliver.output("")
        cliver.output("Default subcommand: list (when /provider is called with no arguments)")
    else:
        cliver.output(f"[yellow]Unknown subcommand: /provider {sub}[/yellow]")
        cliver.output("Run '/provider help' for usage.")


# ---------------------------------------------------------------------------
# Click commands (thin wrappers)
# ---------------------------------------------------------------------------


@provider.command(name="list", help="List all configured providers with type, API URL, and rate limit")
@pass_cliver
def list_providers(cliver: Cliver):
    _list_providers(cliver)


@provider.command(name="add", help="Add a new LLM provider endpoint with API URL and credentials")
@click.option(
    "--name",
    "-n",
    type=str,
    required=True,
    help="Unique provider name (e.g. 'deepseek', 'ollama'). Must not exist",
)
@click.option(
    "--type",
    "-t",
    "ptype",
    type=click.Choice([p.value for p in ProviderEnum]),
    required=True,
    help="Provider type (determines API protocol)",
)
@click.option(
    "--api-url",
    "-u",
    type=str,
    required=True,
    help="API base URL (e.g. 'https://api.deepseek.com')",
)
@click.option(
    "--api-key",
    "-k",
    type=str,
    default=None,
    help="API key. Supports templates: {{ env.VAR }} or {{ keyring() }}",
)
@click.option(
    "--rate-limit",
    "-r",
    type=str,
    default=None,
    help="Rate limit as 'requests/period' (e.g. '5000/5h', '100/1m')",
)
@click.option("--image-url", type=str, default=None, help="Separate image generation endpoint URL (optional)")
@click.option("--audio-url", type=str, default=None, help="Separate audio generation endpoint URL (optional)")
@pass_cliver
def add_provider(
    cliver: Cliver, name: str, ptype: str, api_url: str, api_key: str, rate_limit: str, image_url: str, audio_url: str
):
    _add_provider(cliver, name, ptype, api_url, api_key, rate_limit, image_url, audio_url)


@provider.command(name="set", help="Update an existing provider (only provided values are changed)")
@click.option(
    "--name",
    "-n",
    type=str,
    required=True,
    help="Provider name to update. Must match an existing provider",
)
@click.option(
    "--type",
    "-t",
    "ptype",
    type=click.Choice([p.value for p in ProviderEnum]),
    default=None,
    help="New provider type (changes API protocol)",
)
@click.option("--api-url", "-u", type=str, default=None, help="New API base URL")
@click.option(
    "--api-key",
    "-k",
    type=str,
    default=None,
    help="New API key. Supports templates: {{ env.VAR }} or {{ keyring() }}",
)
@click.option(
    "--rate-limit",
    "-r",
    type=str,
    default=None,
    help="New rate limit as 'requests/period' (e.g. '5000/5h')",
)
@click.option("--image-url", type=str, default=None, help="New image generation endpoint URL")
@click.option("--audio-url", type=str, default=None, help="New audio generation endpoint URL")
@pass_cliver
def set_provider(
    cliver: Cliver, name: str, ptype: str, api_url: str, api_key: str, rate_limit: str, image_url: str, audio_url: str
):
    _set_provider(cliver, name, ptype, api_url, api_key, rate_limit, image_url, audio_url)


@provider.command(name="remove", help="Remove a provider (fails if models still reference it)")
@click.option(
    "--name",
    "-n",
    type=str,
    required=True,
    help="Provider name to remove. Must not have models referencing it",
)
@pass_cliver
def remove_provider(cliver: Cliver, name: str):
    _remove_provider(cliver, name)
