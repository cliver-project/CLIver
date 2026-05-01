import click
from rich import box
from rich.columns import Columns
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from cliver.cli import Cliver, pass_cliver
from cliver.commands import click_help, wants_help

# ── Logic Functions ──


def _validate_config(cliver: Cliver):
    """Validate the current configuration."""
    try:
        # Check if config is valid by attempting to load it
        config_manager = cliver.config_manager
        if config_manager.config:
            cliver.output("[green]✓ Configuration is valid[/green]")
        else:
            cliver.output("[red]✗ Configuration is not valid[/red]")
    except Exception as e:
        cliver.output(f"[red]✗ Configuration validation error: {e}[/red]")


def _set_config(cliver: Cliver, user_agent: str):
    """Update general configuration settings."""
    if not user_agent:
        cliver.output("[dim]Usage: config set --user-agent VALUE[/dim]")
        cliver.output("[dim]Use /agent to manage agent instances.[/dim]")
        return

    if user_agent:
        cliver.config_manager.set_user_agent(user_agent)
        cliver.output(f"User-Agent set to: [green]{user_agent}[/green]")


def _show_config(cliver: Cliver):
    """Show the current configuration with sensitive values masked."""
    try:
        cfg = cliver.config_manager.config
        if not cfg:
            cliver.output("No configuration found.")
            return

        console = cliver.console

        # ── General Settings ──
        general_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
        general_table.add_column("Key", style="cyan", min_width=16)
        general_table.add_column("Value", style="white")

        general_table.add_row("Default Agent", cfg.default_agent_name)
        general_table.add_row("Active Agent", f"[bold green]{cliver.agent_name}[/bold green]")
        if cfg.default_model:
            general_table.add_row("Default Model", f"[green]{cfg.default_model}[/green]")
        if cfg.user_agent:
            general_table.add_row("User-Agent", cfg.user_agent)

        console.print(
            Panel(
                general_table,
                title="[bold cyan]General Settings[/bold cyan]",
                border_style="cyan",
                padding=(0, 1),
            )
        )

        # ── Providers ──
        if cfg.providers:
            prov_table = Table(box=box.SIMPLE_HEAVY, padding=(0, 1))
            prov_table.add_column("Name", style="green")
            prov_table.add_column("Type", style="magenta")
            prov_table.add_column("API URL", style="blue")
            prov_table.add_column("API Key", style="dim")
            prov_table.add_column("Rate Limit", style="yellow")
            prov_table.add_column("Image URL", style="dim")

            for name, prov in cfg.providers.items():
                api_key_display = _mask_value(prov.api_key) if prov.api_key else "-"
                rl = f"{prov.rate_limit.requests}/{prov.rate_limit.period}" if prov.rate_limit else "-"
                prov_table.add_row(name, prov.type, prov.api_url, api_key_display, rl, prov.image_url or "-")

            console.print(
                Panel(
                    prov_table,
                    title="[bold blue]Providers[/bold blue]",
                    border_style="blue",
                    padding=(0, 1),
                )
            )

        # ── Models ──
        if cfg.models:
            model_panels = []
            for name, model in cfg.models.items():
                is_default = name == cfg.default_model
                title_label = (
                    f"[bold green]{name}[/bold green] [dim](default)[/dim]"
                    if is_default
                    else f"[bold yellow]{name}[/bold yellow]"
                )

                t = Table(box=None, show_header=False, padding=(0, 2))
                t.add_column("Key", style="dim", min_width=14)
                t.add_column("Value")

                t.add_row("Provider", f"[magenta]{model.provider}[/magenta]")
                url = model.get_resolved_url()
                if url:
                    t.add_row("URL", f"[blue]{url}[/blue]")
                if model.think_mode is not None:
                    t.add_row("Think Mode", "[green]on[/green]" if model.think_mode else "[red]off[/red]")
                if model.context_window:
                    t.add_row("Context Window", f"{model.context_window:,} tokens")

                # Capabilities
                caps = model.get_capabilities()
                if caps:
                    cap_strs = sorted(c.value for c in caps)
                    t.add_row("Capabilities", ", ".join(cap_strs))

                # Options
                if model.options:
                    opts = model.options.model_dump()
                    opt_parts = [f"{k}={v}" for k, v in opts.items() if v is not None]
                    if opt_parts:
                        t.add_row("Options", ", ".join(opt_parts))

                model_panels.append(Panel(t, title=title_label, border_style="yellow", padding=(0, 1)))

            console.print(
                Panel(
                    Columns(model_panels, equal=True, expand=True) if len(model_panels) > 1 else model_panels[0],
                    title="[bold yellow]Models[/bold yellow]",
                    border_style="yellow",
                    padding=(0, 1),
                )
            )

        # ── MCP Servers ──
        if cfg.mcpServers:
            server_table = Table(box=box.SIMPLE_HEAVY, padding=(0, 1))
            server_table.add_column("Name", style="green")
            server_table.add_column("Transport", style="magenta")
            server_table.add_column("Command / URL", style="blue")
            server_table.add_column("Details", style="dim")

            for name, server in cfg.mcpServers.items():
                data = server.model_dump()
                transport = data.get("transport", "")

                if transport == "stdio":
                    target = data.get("command", "")
                    args = data.get("args")
                    details_parts = []
                    if args:
                        details_parts.append(f"args: {' '.join(args)}")
                    env = data.get("env")
                    if env:
                        details_parts.append(f"env: {len(env)} vars")
                    details = "; ".join(details_parts) if details_parts else ""
                else:
                    target = data.get("url", "")
                    headers = data.get("headers")
                    details = f"{len(headers)} headers" if headers else ""

                server_table.add_row(name, transport, target, details)

            console.print(
                Panel(
                    server_table,
                    title="[bold magenta]MCP Servers[/bold magenta]",
                    border_style="magenta",
                    padding=(0, 1),
                )
            )

        # ── Config path footer ──
        config_path = cliver.config_manager.config_file
        console.print(Text(f"  Config file: {config_path}", style="dim"))

    except Exception as e:
        cliver.output(f"[red]Error showing configuration: {e}[/red]")


def _config_rate_limit(cliver: Cliver, provider_name: str, limit: str | None):
    """Set or show rate limit for a provider."""
    from cliver.commands.provider import _parse_rate_limit

    providers = cliver.config_manager.list_providers()
    if provider_name not in providers:
        cliver.output(f"Provider '{provider_name}' not found.")
        available = ", ".join(providers.keys()) if providers else "(none)"
        cliver.output(f"Available providers: {available}")
        return

    if limit is None:
        prov = providers[provider_name]
        if prov.rate_limit:
            cliver.output(
                f"Rate limit for '{provider_name}': "
                f"{prov.rate_limit.requests}/{prov.rate_limit.period} "
                f"(margin: {prov.rate_limit.margin:.0%})"
            )
        else:
            cliver.output(f"No rate limit set for '{provider_name}'.")
        return

    rl = _parse_rate_limit(limit)
    if cliver.config_manager.set_provider_rate_limit(provider_name, rl):
        cliver.output(f"Rate limit for '{provider_name}' set to {rl.requests}/{rl.period}.")
    else:
        cliver.output(f"Failed to update rate limit for '{provider_name}'.")


def _show_config_path(cliver: Cliver):
    """Show the path to the configuration file."""
    config_path = cliver.config_manager.config_file
    cliver.output(f"Configuration file path: {config_path}")


def _set_theme(cliver: Cliver, name: str | None):
    """Show or set the UI theme. Available: dark, light, dracula."""
    from cliver.themes import get_theme, list_themes, load_theme, set_theme

    if not name:
        current = get_theme()
        available = list_themes()
        cliver.output(f"Current theme: [bold]{current.name}[/bold]")
        cliver.output(f"Available: {', '.join(available)}")
        return

    available = list_themes()
    if name not in available:
        cliver.output(f"[red]Unknown theme '{name}'. Available: {', '.join(available)}[/red]")
        return

    theme = load_theme(name)
    set_theme(theme)
    cliver.theme = theme

    # Update prompt_toolkit app style if running in TUI
    if hasattr(cliver, "_app") and cliver._app is not None:
        from prompt_toolkit.styles import Style

        cliver._app.style = Style.from_dict(theme.prompt_toolkit_styles())
        cliver._app.invalidate()

    # Persist to config
    cliver.config_manager.config.theme = name
    cliver.config_manager._save_config()

    cliver.output(f"Theme set to [bold]{name}[/bold].")


# ── Dispatch ──

_SUBCOMMANDS: dict[str, click.Command] = {}


def dispatch(cliver: Cliver, args: str):
    """Manage configuration — show, set, validate, theme."""
    parts = args.strip().split(None, 1) if args.strip() else []
    sub = parts[0] if parts else "show"
    rest = parts[1] if len(parts) > 1 else ""

    if sub in ("--help", "-h", "help"):
        cliver.output(click_help(config, "/config"))
        return

    if sub in _SUBCOMMANDS and wants_help(rest):
        cliver.output(click_help(_SUBCOMMANDS[sub], f"/config {sub}"))
        return

    if sub == "show":
        _show_config(cliver)
    elif sub == "validate":
        _validate_config(cliver)
    elif sub == "set":
        # Parse --user-agent flag
        if "--user-agent" in rest or "-u" in rest:
            parts = rest.split()
            user_agent = None
            for i, p in enumerate(parts):
                if p in ("--user-agent", "-u") and i + 1 < len(parts):
                    user_agent = parts[i + 1]
                    break
            _set_config(cliver, user_agent or "")
        else:
            cliver.output(click_help(_SUBCOMMANDS["set"], "/config set"))
    elif sub == "rate-limit":
        # Parse provider and optional limit
        rate_parts = rest.split(None, 1)
        if rate_parts:
            provider_name = rate_parts[0]
            limit = rate_parts[1] if len(rate_parts) > 1 else None
            _config_rate_limit(cliver, provider_name, limit)
        else:
            cliver.output(click_help(_SUBCOMMANDS["rate-limit"], "/config rate-limit"))
    elif sub == "path":
        _show_config_path(cliver)
    elif sub == "theme":
        theme_name = rest.strip() if rest.strip() else None
        _set_theme(cliver, theme_name)
    else:
        cliver.output(f"[yellow]Unknown: /config {sub}[/yellow]")


# ── Click Group ──


@click.group(name="config", help="Manage CLIver configuration settings (show, validate, set, theme)")
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
@config.command(name="validate", help="Check if the configuration file is valid and parseable")
@pass_cliver
def validate_config(cliver: Cliver):
    """Validate the current configuration."""
    _validate_config(cliver)


# noinspection PyUnresolvedReferences
@config.command(name="set", help="Update general configuration settings (currently supports user-agent)")
@click.option("--user-agent", "-u", type=str, help="Set the User-Agent header sent with all LLM provider HTTP requests")
@pass_cliver
def set_config(cliver: Cliver, user_agent: str):
    """Update general configuration settings."""
    _set_config(cliver, user_agent or "")


# noinspection PyUnresolvedReferences
@config.command(
    name="show",
    help="Display the full configuration: providers, models, MCP servers (sensitive values masked)",
)
@pass_cliver
def show_config(cliver: Cliver):
    """Show the current configuration with sensitive values masked."""
    _show_config(cliver)


def _mask_value(value: str) -> str:
    """Mask a single secret value for display."""
    if value.startswith("keyring:"):
        return f"[dim]{value}[/dim]"
    if len(value) > 8:
        return f"{value[:3]}***{value[-3:]}"
    return "***"


def _mask_secrets(data, keys_to_mask=("api_key",)):
    """Recursively mask sensitive values in a config dict.

    Jinja2 template expressions (containing {{ }}) are shown as-is since
    they don't contain the actual secret. Plain text secrets are masked.
    """
    if isinstance(data, dict):
        for key, value in data.items():
            if key in keys_to_mask and isinstance(value, str):
                if "{{" in value and "}}" in value:
                    pass  # template expressions are safe to display
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
@config.command(name="rate-limit", help="Show or set the rate limit for a specific provider")
@click.argument("provider_name")
@click.argument("limit", required=False)
@pass_cliver
def config_rate_limit(cliver: Cliver, provider_name: str, limit: str):
    """Set or show rate limit for a provider.

    Usage:
      cliver config rate-limit minimax 5000/5h   # set limit
      cliver config rate-limit minimax            # show current limit
    """
    _config_rate_limit(cliver, provider_name, limit)


# noinspection PyUnresolvedReferences
@config.command(name="path", help="Display the absolute file path to the config.yaml file")
@pass_cliver
def show_config_path(cliver: Cliver):
    """Show the path to the configuration file."""
    _show_config_path(cliver)


@config.command(name="theme", help="Show or set the UI color theme (dark, light, or dracula)")
@click.argument("name", required=False)
@pass_cliver
def set_theme_cmd(cliver: Cliver, name: str = None):
    """Show or set the UI theme. Available: dark, light, dracula."""
    _set_theme(cliver, name)


# Populate subcommand map for dispatch help generation.
_SUBCOMMANDS.update(
    {
        "show": show_config,
        "validate": validate_config,
        "set": set_config,
        "rate-limit": config_rate_limit,
        "path": show_config_path,
        "theme": set_theme_cmd,
    }
)
