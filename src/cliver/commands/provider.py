from enum import Enum

import click
from rich import box
from rich.panel import Panel
from rich.table import Table

from cliver.cli import Cliver, pass_cliver
from cliver.commands import click_help, wants_help
from cliver.config import RateLimitConfig


class ProviderEnum(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


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
# Helpers
# ---------------------------------------------------------------------------


def _parse_provider_flags(rest: str) -> dict:
    """Parse --flag value pairs from a rest string for provider commands."""
    from shlex import split as shlex_split

    try:
        tokens = shlex_split(rest)
    except ValueError:
        tokens = rest.split()

    opts: dict = {}
    i = 0
    flag_map = {
        "--name": "name",
        "-n": "name",
        "--type": "type",
        "-t": "type",
        "--api-url": "api_url",
        "-u": "api_url",
        "--api-key": "api_key",
        "-k": "api_key",
        "--rate-limit": "rate_limit",
        "-r": "rate_limit",
    }
    while i < len(tokens):
        if tokens[i] in flag_map and i + 1 < len(tokens):
            opts[flag_map[tokens[i]]] = tokens[i + 1]
            i += 2
        else:
            i += 1
    return opts


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
    for name, prov in providers.items():
        api_key_display = "***" if prov.api_key else "-"
        rl = f"{prov.rate_limit.requests}/{prov.rate_limit.period}" if prov.rate_limit else "-"
        table.add_row(
            name,
            prov.type,
            prov.api_url or "-",
            api_key_display,
            rl,
        )
    cliver.output(table)


def _add_provider(
    cliver: Cliver,
    name: str,
    ptype: str,
    api_url: str,
    api_key: str,
    rate_limit: str,
):
    """Add a new provider."""
    if name in cliver.config_manager.list_providers():
        cliver.output(f"Provider '{name}' already exists. Use '/provider set' to update.")
        return
    rl = _parse_rate_limit(rate_limit) if rate_limit else None
    cliver.config_manager.add_or_update_provider(
        name,
        ptype,
        api_url,
        api_key,
        rl,
    )
    cliver.output(f"Added provider: {name}")


def _set_provider(
    cliver: Cliver,
    name: str,
    ptype: str,
    api_url: str,
    api_key: str,
    rate_limit: str,
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
        api_key if api_key else existing.api_key,
        rl,
    )
    cliver.output(f"Updated provider: {name}")


def _show_provider(cliver: Cliver, name: str):
    """Show detailed information about a specific provider."""
    providers = cliver.config_manager.list_providers()
    prov = providers.get(name)
    if not prov:
        cliver.output(f"Provider '{name}' not found.")
        return

    t = Table(box=None, show_header=False, padding=(0, 2))
    t.add_column("Key", style="dim", min_width=14)
    t.add_column("Value")

    t.add_row("Name", name)
    t.add_row("Type", prov.type)
    t.add_row("API URL", prov.api_url or "-")
    t.add_row("API Key", "***" if prov.api_key else "-")
    if prov.rate_limit:
        t.add_row("Rate Limit", f"{prov.rate_limit.requests}/{prov.rate_limit.period}")

    # List models under this provider
    models = cliver.config_manager.list_llm_models()
    attached = [n for n, m in models.items() if m.provider == name]
    t.add_row("Models", ", ".join(attached) if attached else "-")

    cliver.output(Panel(t, title=f"Provider: {name}", border_style="green", padding=(0, 1)))


def _remove_provider(cliver: Cliver, name: str):
    """Remove a provider."""
    try:
        if cliver.config_manager.remove_provider(name):
            cliver.output(f"Removed provider: {name}")
        else:
            cliver.output(f"Provider '{name}' not found.")
    except ValueError as e:
        cliver.output(f"{e}")


# ---------------------------------------------------------------------------
# Dispatch function
# ---------------------------------------------------------------------------


def _dispatch_add(cliver: Cliver, rest: str) -> None:
    """Handle /provider add from TUI dispatch."""
    opts = _parse_provider_flags(rest)
    name = opts.get("name")
    ptype = opts.get("type")
    api_url = opts.get("api_url")
    if not name or not ptype or not api_url:
        cliver.output("Usage: /provider add -n NAME -t TYPE -u API_URL [-k API_KEY] [-r RATE_LIMIT]")
        return
    _add_provider(
        cliver,
        name=name,
        ptype=ptype,
        api_url=api_url,
        api_key=opts.get("api_key", ""),
        rate_limit=opts.get("rate_limit", ""),
    )


def _dispatch_set(cliver: Cliver, rest: str) -> None:
    """Handle /provider set from TUI dispatch."""
    opts = _parse_provider_flags(rest)
    name = opts.get("name")
    if not name:
        cliver.output("Usage: /provider set -n NAME [-t TYPE] [-u API_URL] [-k API_KEY] [-r RATE_LIMIT]")
        return
    _set_provider(
        cliver,
        name=name,
        ptype=opts.get("type", ""),
        api_url=opts.get("api_url", ""),
        api_key=opts.get("api_key", ""),
        rate_limit=opts.get("rate_limit", ""),
    )


def dispatch(cliver: Cliver, args: str):
    """Manage LLM providers — list, add, set, remove."""
    parts = args.strip().split(None, 1) if args.strip() else []
    sub = parts[0] if parts else "list"
    rest = parts[1] if len(parts) > 1 else ""

    if sub in ("--help", "-h", "help"):
        cliver.output(click_help(provider, "/provider"))
        return

    if sub in _SUBCOMMANDS and wants_help(rest):
        cliver.output(click_help(_SUBCOMMANDS[sub], f"/provider {sub}"))
        return

    if sub == "list":
        _list_providers(cliver)
    elif sub == "show":
        name = rest.strip()
        if not name:
            cliver.output(click_help(_SUBCOMMANDS["show"], "/provider show"))
            return
        _show_provider(cliver, name)
    elif sub == "add":
        _dispatch_add(cliver, rest)
    elif sub == "set":
        _dispatch_set(cliver, rest)
    elif sub == "remove":
        name = rest.strip()
        if not name:
            cliver.output(click_help(_SUBCOMMANDS["remove"], "/provider remove"))
            return
        _remove_provider(cliver, name)
    else:
        cliver.output(f"Unknown subcommand: /provider {sub}")
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
    help="API key. Use a key name from /keys, an ENV_VAR name, or a literal value.",
)
@click.option(
    "--rate-limit",
    "-r",
    type=str,
    default=None,
    help="Rate limit as 'requests/period' (e.g. '5000/5h', '100/1m')",
)
@pass_cliver
def add_provider(
    cliver: Cliver,
    name: str,
    ptype: str,
    api_url: str,
    api_key: str,
    rate_limit: str,
):
    _add_provider(cliver, name, ptype, api_url, api_key, rate_limit)


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
    help="New API key. Use a key name from /keys, an ENV_VAR name, or a literal value.",
)
@click.option(
    "--rate-limit",
    "-r",
    type=str,
    default=None,
    help="New rate limit as 'requests/period' (e.g. '5000/5h')",
)
@pass_cliver
def set_provider(
    cliver: Cliver,
    name: str,
    ptype: str,
    api_url: str,
    api_key: str,
    rate_limit: str,
):
    _set_provider(cliver, name, ptype, api_url, api_key, rate_limit)


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


@provider.command(name="show", help="Show detailed information about a specific provider (including attached models)")
@click.option("--name", "-n", type=str, required=True, help="Provider name")
@pass_cliver
def show_provider(cliver: Cliver, name: str):
    _show_provider(cliver, name)


# Subcommand lookup for dispatch help
_SUBCOMMANDS = {
    "list": list_providers,
    "show": show_provider,
    "add": add_provider,
    "set": set_provider,
    "remove": remove_provider,
}
