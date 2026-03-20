import click
from rich import box
from rich.columns import Columns
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

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
@config.command(name="set", help="Update general configuration settings")
@click.option("--agent-name", "-a", type=str, help="Set the agent display name")
@click.option("--user-agent", "-u", type=str, help="Set the User-Agent header for LLM provider requests")
@pass_cliver
def set_config(cliver: Cliver, agent_name: str, user_agent: str):
    """Update general configuration settings."""
    if not agent_name and not user_agent:
        cliver.console.print("[dim]Usage: config set --agent-name NAME | --user-agent VALUE[/dim]")
        return

    if agent_name:
        cliver.config_manager.set_agent_name(agent_name)
        cliver.console.print(f"Agent name set to: [green]{agent_name}[/green]")

    if user_agent:
        cliver.config_manager.set_user_agent(user_agent)
        cliver.console.print(f"User-Agent set to: [green]{user_agent}[/green]")


# noinspection PyUnresolvedReferences
@config.command(name="show", help="Show current configuration")
@pass_cliver
def show_config(cliver: Cliver):
    """Show the current configuration with sensitive values masked."""
    try:
        cfg = cliver.config_manager.config
        if not cfg:
            cliver.console.print("No configuration found.")
            return

        console = cliver.console

        # ── General Settings ──
        general_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
        general_table.add_column("Key", style="cyan", min_width=16)
        general_table.add_column("Value", style="white")

        general_table.add_row("Agent Name", cfg.agent_name)
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
                t.add_row("URL", f"[blue]{model.url}[/blue]")
                if model.name_in_provider:
                    t.add_row("Provider Name", model.name_in_provider)
                if model.api_key:
                    t.add_row("API Key", _mask_value(model.api_key))
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

        # ── Workflow ──
        if cfg.workflow:
            wf = cfg.workflow
            wf_table = Table(box=None, show_header=False, padding=(0, 2))
            wf_table.add_column("Key", style="dim", min_width=16)
            wf_table.add_column("Value")

            if wf.workflow_dirs:
                wf_table.add_row("Workflow Dirs", "\n".join(wf.workflow_dirs))
            if wf.cache_dir:
                wf_table.add_row("Cache Dir", wf.cache_dir)

            # Only show if there's content
            if wf.workflow_dirs or wf.cache_dir:
                console.print(
                    Panel(
                        wf_table,
                        title="[bold blue]Workflow[/bold blue]",
                        border_style="blue",
                        padding=(0, 1),
                    )
                )

        # ── Config path footer ──
        config_path = cliver.config_manager.config_file
        console.print(Text(f"  Config file: {config_path}", style="dim"))

    except Exception as e:
        cliver.console.print(f"[red]Error showing configuration: {e}[/red]")


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
@config.command(name="path", help="Show configuration file path")
@pass_cliver
def show_config_path(cliver: Cliver):
    """Show the path to the configuration file."""
    config_path = cliver.config_manager.config_file
    cliver.console.print(f"Configuration file path: {config_path}")
