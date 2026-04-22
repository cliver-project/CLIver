import click
from rich import box
from rich.table import Table

from cliver.cli import Cliver, pass_cliver
from cliver.model_capabilities import ModelCapability
from cliver.util import parse_key_value_options


@click.group(name="model", help="Manage LLM Models")
@click.pass_context
def model(ctx: click.Context):
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())
        ctx.exit()


# ---------------------------------------------------------------------------
# Business logic functions
# ---------------------------------------------------------------------------


def _list_models(cliver: Cliver):
    """List all configured LLM models with capabilities."""
    models = cliver.config_manager.list_llm_models()
    if not models:
        cliver.output("No LLM Models configured.")
        return

    table = Table(title="Configured LLM Models", box=box.SQUARE)
    table.add_column("", min_width=1, max_width=1, no_wrap=True)
    table.add_column("Name", style="green")
    table.add_column("Provider", style="cyan")
    table.add_column("Modalities", style="blue")
    table.add_column("Features", style="yellow")
    table.add_column("Context", style="dim", justify="right")

    global_default = cliver.config_manager.config.default_model
    session_model = (cliver.session_options or {}).get("model")
    active_model = session_model or global_default

    for _, model in models.items():
        capabilities = model.get_capabilities()
        modalities = _format_modalities(capabilities) if capabilities else "-"
        features = _format_features(capabilities) if capabilities else "-"
        ctx_window = f"{model.context_window:,}" if model.context_window else "-"
        marker = "[bold green]✔[/bold green]" if model.name == active_model else ""

        table.add_row(marker, model.name, model.provider, modalities, features, ctx_window)

    cliver.output(table)


def _show_model_detail(cliver: Cliver, name: str):
    """Show detailed information about a specific model."""
    from rich.panel import Panel

    models = cliver.config_manager.list_llm_models()
    if name not in models:
        cliver.output(f"[red]Model '{name}' not found.[/red]")
        cliver.output(f"Available: {', '.join(models.keys())}")
        return

    model = models[name]
    is_default = name == cliver.config_manager.config.default_model
    capabilities = model.get_capabilities()

    t = Table(box=None, show_header=False, padding=(0, 2))
    t.add_column("Key", style="dim", min_width=18)
    t.add_column("Value")

    t.add_row("Name", f"[bold green]{name}[/bold green]")
    if is_default:
        t.add_row("Status", "[green]● default[/green]")
    t.add_row("Provider", f"[magenta]{model.provider}[/magenta]")
    if model.name_in_provider:
        t.add_row("Name in Provider", model.name_in_provider)

    # URL — resolved from provider if model-level not set
    url = model.get_resolved_url()
    if url:
        t.add_row("URL", f"[blue]{url}[/blue]")

    # API key — mask if plain text, show template expressions as-is
    if model.api_key:
        t.add_row("API Key", _mask_api_key(model.api_key))
    elif model._provider_config and model._provider_config.api_key:
        t.add_row("API Key", f"[dim](from provider)[/dim] {_mask_api_key(model._provider_config.api_key)}")

    # Context window
    if model.context_window:
        t.add_row("Context Window", f"{model.context_window:,} tokens")

    # Think mode
    if model.think_mode is not None:
        t.add_row("Think Mode", "[green]enabled[/green]" if model.think_mode else "[red]disabled[/red]")

    # Capabilities
    modalities = _format_modalities(capabilities)
    features = _format_features(capabilities)
    t.add_row("Modalities", modalities)
    t.add_row("Features", features)
    cap_names = sorted(c.value for c in capabilities)
    t.add_row("All Capabilities", ", ".join(cap_names) if cap_names else "-")

    # Options
    if model.options:
        opts = model.options.model_dump(exclude_unset=True)
        if opts:
            for k, v in opts.items():
                t.add_row(f"  {k}", str(v))

    # Pricing
    pricing = model.get_resolved_pricing()
    if pricing:
        inp, out, cached, currency = pricing
        t.add_row("Pricing (per 1M)", f"in: {inp} / out: {out} / cached: {cached} {currency}")

    cliver.output(Panel(t, title=f"[bold]Model: {name}[/bold]", border_style="green", padding=(0, 1)))


def _mask_api_key(key: str) -> str:
    """Mask API key — show template expressions, mask plain text."""
    if "{{" in key and "}}" in key:
        return f"[dim]{key}[/dim]"
    if len(key) <= 8:
        return "****"
    return f"{key[:4]}{'*' * (len(key) - 8)}{key[-4:]}"


def _format_modalities(capabilities: set) -> str:
    """Format capabilities as short modality labels (text, image, audio, video)."""
    # Map capability enum values to short modality labels
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
        # Show available models
        models = cliver.config_manager.list_llm_models()
        if models:
            cliver.output(f"Available models: {', '.join(models.keys())}")
        return

    if cliver.config_manager.set_default_model(name):
        # Sync to the running AgentCore
        cliver.task_executor.default_model = name
        cliver.output(f"Default model set to: [green]{name}[/green]")
    else:
        cliver.output(f"[red]Model '{name}' not found.[/red]")
        models = cliver.config_manager.list_llm_models()
        if models:
            cliver.output(f"Available models: {', '.join(models.keys())}")


def _remove_model(cliver: Cliver, name: str):
    """Remove a LLM model."""
    model = cliver.config_manager.get_llm_model(name)
    if not model:
        cliver.output(f"No LLM Model found with name: {name}")
        return
    cliver.config_manager.remove_llm_model(name)
    cliver.output(f"Removed LLM Model: {name}")


def _add_model(
    cliver: Cliver,
    name: str,
    provider: str,
    api_key: str = None,
    url: str = None,
    option: tuple = None,
    name_in_provider: str = None,
    capabilities: str = None,
):
    """Add a new LLM model."""
    model = cliver.config_manager.get_llm_model(name)
    if model:
        cliver.output(f"LLM Model found with name: {name} already exists.")
        return

    # Convert key=value options to JSON string
    options_dict = {}
    if option:
        options_dict = parse_key_value_options(option, cliver.console)

    cliver.config_manager.add_or_update_llm_model(
        name, provider, api_key, url, options_dict, name_in_provider, capabilities
    )
    cliver.output(f"Added LLM Model: {name}")


def _update_model(
    cliver: Cliver,
    name: str,
    provider: str = None,
    api_key: str = None,
    url: str = None,
    option: tuple = None,
    name_in_provider: str = None,
    capabilities: str = None,
):
    """Update an existing LLM model."""
    model = cliver.config_manager.get_llm_model(name)
    if not model:
        cliver.output(f"LLM Model with name: {name} was not found.")
        return

    # Convert key=value options to JSON string
    options_dict = {}
    if option:
        options_dict = parse_key_value_options(option, cliver.console)

    cliver.config_manager.add_or_update_llm_model(
        name, provider, api_key, url, options_dict, name_in_provider, capabilities
    )
    cliver.output(f"LLM Model: {name} updated")


# ---------------------------------------------------------------------------
# Dispatch function
# ---------------------------------------------------------------------------


def dispatch(cliver: Cliver, args: str):
    """Dispatch /model commands from string args."""
    parts = args.strip().split(None, 1) if args.strip() else []
    sub = parts[0] if parts else "list"
    rest = parts[1] if len(parts) > 1 else ""

    if sub == "list":
        _list_models(cliver)
    elif sub == "default":
        model_name = rest.strip() if rest else None
        _set_default_model(cliver, model_name)
    elif sub in ("--help", "help"):
        cliver.output("Usage: /model [list|<name>|default|add|set|remove]")
        cliver.output("  list                - List all configured models")
        cliver.output("  <name>              - Show detailed info for a model")
        cliver.output("  default [name]      - Show or set default model")
        cliver.output("  add --name NAME ... - Add a new model")
        cliver.output("  set --name NAME ... - Update a model")
        cliver.output("  remove --name NAME  - Remove a model")
    else:
        # Check if sub is a model name → show detail
        models = cliver.config_manager.list_llm_models()
        if sub in models:
            _show_model_detail(cliver, sub)
        else:
            cliver.output(f"[yellow]Unknown subcommand or model: /model {sub}[/yellow]")
            cliver.output("Run '/model help' for usage.")


# ---------------------------------------------------------------------------
# Click commands (thin wrappers)
# ---------------------------------------------------------------------------


# noinspection PyUnresolvedReferences
@model.command(name="list", help="List LLM Models")
@pass_cliver
def list_llm_models(cliver: Cliver):
    _list_models(cliver)


# noinspection PyUnresolvedReferences
@model.command(name="default", help="Set the default LLM model")
@click.argument("name", type=ModelNameType(), required=False)
@pass_cliver
def set_default_model(cliver: Cliver, name: str):
    """Set or show the default LLM model."""
    _set_default_model(cliver, name)


# noinspection PyUnresolvedReferences
@model.command(name="remove", help="Remove a LLM Model")
@click.option(
    "--name",
    "-n",
    type=str,
    required=True,
    help="Name of the LLM Model",
)
@pass_cliver
def remove_llm_model(cliver: Cliver, name: str):
    _remove_model(cliver, name)


# noinspection PyUnresolvedReferences
@model.command(name="add", help="Add a LLM Model")
@click.option(
    "--name",
    "-n",
    type=str,
    required=True,
    help="Name of the LLM Model",
)
@click.option(
    "--provider",
    "-p",
    type=str,
    required=True,
    help="The provider of the LLM Model (provider name or type: ollama, openai, anthropic, vllm)",
)
@click.option(
    "--api-key",
    "-k",
    type=str,
    help="The api_key of the LLM Model",
)
@click.option(
    "--url",
    "-u",
    type=str,
    required=False,
    help="The url of the LLM Provider service",
)
@click.option(
    "--option",
    "-o",
    multiple=True,
    type=str,
    help="Model options in key=value format (can be specified multiple times)",
)
@click.option(
    "--name-in-provider",
    "-N",
    type=str,
    help="The name of the LLM within the Provider",
)
@click.option(
    "--capabilities",
    "-c",
    type=str,
    help="Comma-separated list of model capabilities (e.g., text_to_text,image_to_text,tool_calling)",
)
@pass_cliver
def add_llm_model(
    cliver: Cliver,
    name: str,
    provider: str,
    api_key: str,
    url: str,
    option: tuple,
    name_in_provider: str,
    capabilities: str,
):
    _add_model(cliver, name, provider, api_key, url, option, name_in_provider, capabilities)


# noinspection PyUnresolvedReferences
@model.command(name="set", help="Update a LLM Model")
@click.option(
    "--name",
    "-n",
    type=str,
    required=True,
    help="Name of the LLM Model",
)
@click.option(
    "--provider",
    "-p",
    type=str,
    help="The provider of the LLM Model (provider name or type: ollama, openai, anthropic, vllm)",
)
@click.option(
    "--api-key",
    "-k",
    type=str,
    help="The api_key of the LLM Model",
)
@click.option(
    "--url",
    "-u",
    type=str,
    help="The url of the LLM Provider service",
)
@click.option(
    "--option",
    "-o",
    multiple=True,
    type=str,
    help="Model options in key=value format (can be specified multiple times)",
)
@click.option(
    "--name-in-provider",
    "-N",
    type=str,
    help="The name of the LLM within the Provider",
)
@click.option(
    "--capabilities",
    "-c",
    type=str,
    help="Comma-separated list of model capabilities (e.g., text_to_text,image_to_text,tool_calling)",
)
@pass_cliver
def update_llm_model(
    cliver: Cliver,
    name: str,
    provider: str,
    api_key: str,
    url: str,
    option: tuple,
    name_in_provider: str,
    capabilities: str,
):
    _update_model(cliver, name, provider, api_key, url, option, name_in_provider, capabilities)
