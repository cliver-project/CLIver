import click
from rich import box
from rich.table import Table

from cliver.cli import Cliver, pass_cliver
from cliver.model_capabilities import ModelCapability
from cliver.util import parse_key_value_options


@click.group(name="model", help="Manage LLM model configurations (list, add, update, set default, remove)")
@click.pass_context
def model(ctx: click.Context):
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())
        ctx.exit()


# ---------------------------------------------------------------------------
# Business logic functions
# ---------------------------------------------------------------------------


def _list_models(cliver: Cliver):
    """List all configured LLM models grouped by provider."""
    models = cliver.config_manager.list_llm_models()
    if not models:
        cliver.output("No LLM Models configured.")
        return

    table = Table(title="Configured LLM Models", box=box.SQUARE)
    table.add_column("", min_width=1, max_width=1, no_wrap=True)
    table.add_column("Model", style="green")
    table.add_column("Provider", style="cyan")
    table.add_column("Modalities", style="blue")

    global_default = cliver.config_manager.config.default_model
    session_model = (cliver.session_options or {}).get("model")
    active_model = session_model or global_default

    for _, model_cfg in models.items():
        capabilities = model_cfg.get_capabilities()
        modalities = _format_modalities(capabilities) if capabilities else "-"

        is_active = model_cfg.name == active_model
        if not is_active and active_model:
            is_active = model_cfg.name.endswith(f"/{active_model}")
        marker = "[bold green]✔[/bold green]" if is_active else ""

        table.add_row(marker, model_cfg.name, model_cfg.provider, modalities)

    cliver.output(table)


def _show_model_detail(cliver: Cliver, name: str):
    """Show detailed information about a specific model."""
    from rich.panel import Panel

    mc = cliver.config_manager.get_llm_model(name)
    if not mc:
        cliver.output(f"[red]Model '{name}' not found.[/red]")
        models = cliver.config_manager.list_llm_models()
        if models:
            cliver.output(f"Available: {', '.join(models.keys())}")
        return

    is_default = mc.name == cliver.config_manager.config.default_model
    capabilities = mc.get_capabilities()

    t = Table(box=None, show_header=False, padding=(0, 2))
    t.add_column("Key", style="dim", min_width=18)
    t.add_column("Value")

    t.add_row("Name", f"[bold green]{mc.name}[/bold green]")
    t.add_row("API Model Name", mc.api_model_name)
    if is_default:
        t.add_row("Status", "[green]● default[/green]")
    t.add_row("Provider", f"[magenta]{mc.provider}[/magenta]")

    url = mc.get_resolved_url()
    if url:
        t.add_row("URL", f"[blue]{url}[/blue]")

    if mc._provider_config and mc._provider_config.api_key:
        t.add_row("API Key", f"[dim](from provider)[/dim] {_mask_api_key(mc._provider_config.api_key)}")

    if mc.context_window:
        t.add_row("Context Window", f"{mc.context_window:,} tokens")

    if mc.think_mode is not None:
        t.add_row("Think Mode", "[green]enabled[/green]" if mc.think_mode else "[red]disabled[/red]")

    modalities = _format_modalities(capabilities)
    features = _format_features(capabilities)
    t.add_row("Modalities", modalities)
    t.add_row("Features", features)
    cap_names = sorted(c.value for c in capabilities)
    t.add_row("All Capabilities", ", ".join(cap_names) if cap_names else "-")

    if mc.options:
        opts = mc.options.model_dump(exclude_unset=True)
        if opts:
            for k, v in opts.items():
                t.add_row(f"  {k}", str(v))

    pricing = mc.get_resolved_pricing()
    if pricing:
        inp, out, cached, currency = pricing
        t.add_row("Pricing (per 1M)", f"in: {inp} / out: {out} / cached: {cached} {currency}")

    cliver.output(Panel(t, title=f"[bold]Model: {mc.name}[/bold]", border_style="green", padding=(0, 1)))


def _mask_api_key(key: str) -> str:
    """Mask API key — show template expressions, mask plain text."""
    if "{{" in key and "}}" in key:
        return f"[dim]{key}[/dim]"
    if len(key) <= 8:
        return "****"
    return f"{key[:4]}{'*' * (len(key) - 8)}{key[-4:]}"


def _format_modalities(capabilities: set) -> str:
    """Format capabilities as short modality labels (text, image, audio, video)."""
    _MODALITY_MAP = {
        ModelCapability.TEXT_TO_TEXT: "text",
        ModelCapability.IMAGE_TO_TEXT: "image",
        ModelCapability.TEXT_TO_IMAGE: "image",
        ModelCapability.AUDIO_TO_TEXT: "audio",
        ModelCapability.TEXT_TO_AUDIO: "audio",
        ModelCapability.VIDEO_TO_TEXT: "video",
        ModelCapability.TEXT_TO_VIDEO: "video",
    }
    modalities = []
    for cap in _MODALITY_MAP:
        if cap in capabilities:
            label = _MODALITY_MAP[cap]
            if label not in modalities:
                modalities.append(label)
    return ", ".join(modalities) if modalities else "-"


def _format_features(capabilities: set) -> str:
    """Format non-modality capabilities as feature tags."""
    features = []
    if ModelCapability.TOOL_CALLING in capabilities:
        features.append("[green]tools[/green]")
    if ModelCapability.FUNCTION_CALLING in capabilities:
        features.append("[green]fn-call[/green]")
    if ModelCapability.THINK_MODE in capabilities:
        features.append("[magenta]think[/magenta]")
    if ModelCapability.FILE_UPLOAD in capabilities:
        features.append("upload")
    return " ".join(features) if features else "-"


class ModelNameType(click.ParamType):
    """Click parameter type that provides shell completion for model names."""

    name = "model_name"

    def shell_complete(self, ctx, param, incomplete):
        try:
            cliver_obj = ctx.find_object(Cliver)
            if cliver_obj:
                models = cliver_obj.config_manager.list_llm_models()
                return [click.shell_completion.CompletionItem(name) for name in models if name.startswith(incomplete)]
        except Exception:
            pass
        return []


def _set_default_model(cliver: Cliver, name: str = None):
    """Set or show the default LLM model."""
    if not name:
        current = cliver.config_manager.config.default_model
        if current:
            cliver.output(f"Current default model: [green]{current}[/green]")
        else:
            cliver.output("No default model set.")
        models = cliver.config_manager.list_llm_models()
        if models:
            cliver.output(f"Available models: {', '.join(models.keys())}")
        return

    if cliver.config_manager.set_default_model(name):
        mc = cliver.config_manager.get_llm_model(name)
        cliver.agent_core.default_model = mc.name
        cliver.output(f"Default model set to: [green]{mc.name}[/green]")
    else:
        cliver.output(f"[red]Model '{name}' not found.[/red]")
        models = cliver.config_manager.list_llm_models()
        if models:
            cliver.output(f"Available models: {', '.join(models.keys())}")


def _remove_model(cliver: Cliver, name: str):
    """Remove a LLM model."""
    mc = cliver.config_manager.get_llm_model(name)
    if not mc:
        cliver.output(f"No LLM Model found with name: {name}")
        return
    cliver.config_manager.remove_llm_model(mc.name)
    cliver.output(f"Removed LLM Model: {mc.name}")


def _add_model(
    cliver: Cliver,
    provider: str,
    name: str,
    option: tuple = None,
    capabilities: str = None,
):
    """Add a new LLM model to a provider."""
    if provider not in cliver.config_manager.list_providers():
        cliver.output(f"[red]Provider '{provider}' not found.[/red]")
        return

    key = f"{provider}/{name}"
    existing = cliver.config_manager.get_llm_model(key)
    if existing:
        cliver.output(f"Model '{key}' already exists.")
        return

    options_dict = {}
    if option:
        options_dict = parse_key_value_options(option, cliver.console)

    cliver.config_manager.add_or_update_llm_model(provider, name, options_dict or None, capabilities)
    cliver.output(f"Added LLM Model: {key}")


def _update_model(
    cliver: Cliver,
    name: str,
    option: tuple = None,
    capabilities: str = None,
):
    """Update an existing LLM model."""
    mc = cliver.config_manager.get_llm_model(name)
    if not mc:
        cliver.output(f"LLM Model '{name}' not found.")
        return

    options_dict = {}
    if option:
        options_dict = parse_key_value_options(option, cliver.console)

    cliver.config_manager.add_or_update_llm_model(mc.provider, mc.api_model_name, options_dict or None, capabilities)
    cliver.output(f"LLM Model: {mc.name} updated")


# ---------------------------------------------------------------------------
# Dispatch function
# ---------------------------------------------------------------------------


def dispatch(cliver: Cliver, args: str):
    """Manage LLM models — list, add, set, remove, default."""
    parts = args.strip().split(None, 1) if args.strip() else []
    sub = parts[0] if parts else "list"
    rest = parts[1] if len(parts) > 1 else ""

    if sub == "list":
        _list_models(cliver)
    elif sub == "default":
        model_name = rest.strip() if rest else None
        _set_default_model(cliver, model_name)
    elif sub == "add":
        _dispatch_add(cliver, rest)
    elif sub == "set":
        _dispatch_set(cliver, rest)
    elif sub == "remove":
        name = rest.strip()
        if not name:
            cliver.output("[yellow]Usage: /model remove <name>[/yellow]")
            return
        _remove_model(cliver, name)
    elif sub in ("--help", "help"):
        cliver.output("Manage LLM model configurations. Models are grouped under providers.")
        cliver.output("")
        cliver.output("Usage: /model [list|<name>|default|add|set|remove] [options]")
        cliver.output("")
        cliver.output("Subcommands:")
        cliver.output("  list              — List all configured models with provider, modalities,")
        cliver.output("                      and active status. No parameters.")
        cliver.output("  <name>            — Show detailed info for a specific model (capabilities,")
        cliver.output("                      pricing, options, context window).")
        cliver.output("    name  STRING — Model name, either canonical (provider/model) or short-form.")
        cliver.output("    Example: /model deepseek/deepseek-chat")
        cliver.output("  default [name]    — Show or set the default model for new sessions.")
        cliver.output("    name  STRING (optional) — Model name to set as default.")
        cliver.output("      If omitted, displays the current default. Must match a name from '/model list'.")
        cliver.output("    Example: /model default qwen/qwen3-coder")
        cliver.output("  add               — Register a new model under an existing provider.")
        cliver.output("    --provider, -p    STRING (required) — Provider name. Must exist in '/provider list'.")
        cliver.output("    --name, -n        STRING (required) — Model API name (e.g. 'deepseek-chat').")
        cliver.output("    --option, -o      STRING (optional, repeatable) — Model option as key=value")
        cliver.output("                        (e.g. -o temperature=0.7 -o max_tokens=4096).")
        cliver.output("    --capabilities, -c STRING (optional) — Comma-separated capability list")
        cliver.output("                        (e.g. 'text_to_text,tool_calling,think_mode').")
        cliver.output("    Example: /model add --provider deepseek --name deepseek-chat -o temperature=0.7")
        cliver.output("  set               — Update options or capabilities of an existing model.")
        cliver.output("    --name, -n        STRING (required) — Model name (canonical or short-form).")
        cliver.output("    --option, -o      STRING (optional, repeatable) — Option as key=value to update.")
        cliver.output("    --capabilities, -c STRING (optional) — New comma-separated capability list.")
        cliver.output("    Example: /model set --name qwen/qwen3-coder -o temperature=0.5")
        cliver.output("  remove <name>     — Remove a model configuration permanently.")
        cliver.output("    name  STRING (required) — Model name to remove.")
        cliver.output("    Example: /model remove deepseek/old-model")
        cliver.output("")
        cliver.output("Default subcommand: list (when /model is called with no arguments)")
    else:
        mc = cliver.config_manager.get_llm_model(sub)
        if mc:
            _show_model_detail(cliver, sub)
        else:
            cliver.output(f"[yellow]Unknown subcommand or model: /model {sub}[/yellow]")
            cliver.output("Run '/model help' for usage.")


def _parse_model_flags(rest: str) -> dict:
    """Parse --flag value pairs from a rest string."""
    from shlex import split as shlex_split

    try:
        tokens = shlex_split(rest)
    except ValueError:
        tokens = rest.split()

    opts: dict = {}
    options = []
    i = 0
    flag_map = {
        "--name": "name",
        "-n": "name",
        "--provider": "provider",
        "-p": "provider",
        "--capabilities": "capabilities",
        "-c": "capabilities",
    }
    while i < len(tokens):
        if tokens[i] in flag_map and i + 1 < len(tokens):
            opts[flag_map[tokens[i]]] = tokens[i + 1]
            i += 2
        elif tokens[i] in ("--option", "-o") and i + 1 < len(tokens):
            options.append(tokens[i + 1])
            i += 2
        else:
            i += 1
    if options:
        opts["option"] = tuple(options)
    return opts


def _dispatch_add(cliver: Cliver, rest: str) -> None:
    """Handle /model add from TUI dispatch."""
    opts = _parse_model_flags(rest)
    name = opts.get("name")
    provider = opts.get("provider")
    if not name or not provider:
        cliver.output("[yellow]Usage: /model add --provider PROVIDER --name NAME [--option K=V][/yellow]")
        return
    _add_model(cliver, provider, name, option=opts.get("option"), capabilities=opts.get("capabilities"))


def _dispatch_set(cliver: Cliver, rest: str) -> None:
    """Handle /model set from TUI dispatch."""
    opts = _parse_model_flags(rest)
    name = opts.get("name")
    if not name:
        cliver.output("[yellow]Usage: /model set --name NAME [--option K=V][/yellow]")
        return
    _update_model(cliver, name, option=opts.get("option"), capabilities=opts.get("capabilities"))


# ---------------------------------------------------------------------------
# Click commands (thin wrappers)
# ---------------------------------------------------------------------------


# noinspection PyUnresolvedReferences
@model.command(name="list", help="List all configured models with provider, modalities, and active status")
@pass_cliver
def list_llm_models(cliver: Cliver):
    _list_models(cliver)


# noinspection PyUnresolvedReferences
@model.command(name="default", help="Show or set the default LLM model for new sessions")
@click.argument("name", type=ModelNameType(), required=False)
@pass_cliver
def set_default_model(cliver: Cliver, name: str):
    """Set or show the default LLM model."""
    _set_default_model(cliver, name)


# noinspection PyUnresolvedReferences
@model.command(name="remove", help="Remove a model configuration permanently by name")
@click.argument("name", type=str)
@pass_cliver
def remove_llm_model(cliver: Cliver, name: str):
    _remove_model(cliver, name)


# noinspection PyUnresolvedReferences
@model.command(name="add", help="Register a new model under an existing provider")
@click.option(
    "--name",
    "-n",
    type=str,
    required=True,
    help="Model API name as used by the provider (e.g. 'deepseek-chat')",
)
@click.option(
    "--provider",
    "-p",
    type=str,
    required=True,
    help="Provider name. Must exist in 'provider list' (e.g. 'deepseek')",
)
@click.option(
    "--option",
    "-o",
    multiple=True,
    type=str,
    help="Model option as key=value (repeatable, e.g. -o temperature=0.7)",
)
@click.option(
    "--capabilities",
    "-c",
    type=str,
    help="Comma-separated capabilities (e.g. 'text_to_text,tool_calling')",
)
@pass_cliver
def add_llm_model(cliver: Cliver, name: str, provider: str, option: tuple, capabilities: str):
    _add_model(cliver, provider, name, option, capabilities)


# noinspection PyUnresolvedReferences
@model.command(name="set", help="Update options or capabilities of an existing model")
@click.option(
    "--name",
    "-n",
    type=str,
    required=True,
    help="Model name in canonical (provider/model) or short form",
)
@click.option(
    "--option",
    "-o",
    multiple=True,
    type=str,
    help="Option as key=value to update (repeatable, e.g. -o temperature=0.5)",
)
@click.option("--capabilities", "-c", type=str, help="New comma-separated capability list (replaces existing)")
@pass_cliver
def update_llm_model(cliver: Cliver, name: str, option: tuple, capabilities: str):
    _update_model(cliver, name, option, capabilities)
