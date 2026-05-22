import click
from rich import box
from rich.panel import Panel
from rich.table import Table

from cliver.cli import Cliver, pass_cliver
from cliver.commands import click_help, wants_help
from cliver.model.store import ModelStore
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


def _get_store(cliver: Cliver):
    """Get a ModelStore for the current config directory."""
    return ModelStore.from_config_dir(cliver.config_dir)


def _resolve_model(store, name: str):
    """Resolve a model by canonical name (provider/model) or short name.

    Returns (Model, Provider) or (None, None).
    """
    models = store.list_models()
    providers = {p.id: p for p in store.list_providers()}

    for m in models:
        p = providers.get(m.provider_id)
        if not p:
            continue
        canonical = f"{p.name}/{m.name}"
        if canonical == name:
            return m, p
        if m.name == name:
            return m, p
    return None, None


def _list_models(cliver: Cliver):
    """List all configured LLM models grouped by provider."""
    store = _get_store(cliver)
    models = store.list_models()
    if not models:
        cliver.output("No LLM Models configured.")
        return

    providers_by_id = {p.id: p for p in store.list_providers()}

    table = Table(title="Configured LLM Models", box=box.SQUARE)
    table.add_column("", min_width=1, max_width=1, no_wrap=True)
    table.add_column("Model", style="green")
    table.add_column("Provider", style="cyan")
    table.add_column("Modalities", style="blue")

    try:
        default_model = store.get_default_model()
        default_id = default_model.id if default_model else None
    except Exception:
        default_id = None

    for m in models:
        provider = providers_by_id.get(m.provider_id)
        provider_name = provider.name if provider else m.provider_id
        canonical = f"{provider_name}/{m.name}"

        modalities = []
        caps = m.capabilities or []
        if any(c in caps for c in ["text_to_text"]):
            modalities.append("text")
        if any(c in caps for c in ["image_to_text", "text_to_image"]):
            modalities.append("image")
        if any(c in caps for c in ["audio_to_text", "text_to_audio"]):
            modalities.append("audio")
        if any(c in caps for c in ["video_to_text", "text_to_video"]):
            modalities.append("video")

        marker = "✔" if m.id == default_id else ""
        table.add_row(marker, canonical, provider_name, ", ".join(modalities) or "-")

    cliver.output(table)


def _show_model_detail(cliver: Cliver, name: str):
    """Show detailed information about a specific model."""
    store = _get_store(cliver)
    model_obj, provider = _resolve_model(store, name)
    if not model_obj:
        cliver.output(f"Model '{name}' not found.")
        models = store.list_models()
        if models:
            providers = {p.id: p for p in store.list_providers()}
            names = []
            for m in models:
                p = providers.get(m.provider_id)
                names.append(f"{p.name}/{m.name}" if p else m.name)
            cliver.output(f"Available: {', '.join(names)}")
        return

    endpoint = store.get_endpoint(model_obj.endpoint_id) if model_obj.endpoint_id else None
    provider_name = provider.name if provider else model_obj.provider_id

    is_default = model_obj.is_default == 1

    t = Table(box=None, show_header=False, padding=(0, 2))
    t.add_column("Key", style="dim", min_width=18)
    t.add_column("Value")

    canonical = f"{provider_name}/{model_obj.name}"
    t.add_row("Name", canonical)
    t.add_row("API Model Name", model_obj.name)
    if is_default:
        t.add_row("Status", "default")
    t.add_row("Provider", provider_name)

    if endpoint:
        t.add_row("URL", endpoint.base_url)

    if provider and provider.api_key:
        t.add_row("API Key", f"(from provider) {_mask_api_key(provider.api_key)}")

    if model_obj.context_window:
        t.add_row("Context Window", f"{model_obj.context_window:,} tokens")

    if model_obj.think_mode is not None:
        t.add_row("Think Mode", "enabled" if model_obj.think_mode else "disabled")

    caps_set = _string_caps_to_set(model_obj.capabilities or [])
    modalities = _format_modalities(caps_set)
    features = _format_features(caps_set)
    t.add_row("Modalities", modalities)
    t.add_row("Features", features)
    cap_names = sorted(c.value for c in caps_set) if caps_set else []
    t.add_row("All Capabilities", ", ".join(cap_names) if cap_names else "-")

    if model_obj.options:
        opts = model_obj.options
        if opts:
            for k, v in opts.items():
                t.add_row(f"  {k}", str(v))

    if model_obj.pricing:
        inp = model_obj.pricing.get("input")
        out = model_obj.pricing.get("output")
        cached = model_obj.pricing.get("cached_input", inp)
        currency = model_obj.pricing.get("currency", "USD")
        if inp is not None and out is not None:
            t.add_row("Pricing (per 1M)", f"in: {inp} / out: {out} / cached: {cached} {currency}")

    cliver.output(Panel(t, title=f"Model: {canonical}", border_style="green", padding=(0, 1)))


def _string_caps_to_set(cap_list):
    """Convert a list of capability strings to a set of ModelCapability enums."""
    result = set()
    for c in cap_list:
        try:
            result.add(ModelCapability(c))
        except ValueError:
            pass
    return result


def _mask_api_key(key: str) -> str:
    """Mask API key — show template expressions, mask plain text."""
    if "{{" in key and "}}" in key:
        return f"{key}"
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
        features.append("tools")
    if ModelCapability.FUNCTION_CALLING in capabilities:
        features.append("fn-call")
    if ModelCapability.THINK_MODE in capabilities:
        features.append("think")
    return " ".join(features) if features else "-"


class ModelNameType(click.ParamType):
    """Click parameter type that provides shell completion for model names."""

    name = "model_name"

    def shell_complete(self, ctx, param, incomplete):
        try:
            cliver_obj = ctx.find_object(Cliver)
            if cliver_obj:
                store = _get_store(cliver_obj)
                models = store.list_models()
                providers = {p.id: p for p in store.list_providers()}

                names = []
                for m in models:
                    p = providers.get(m.provider_id)
                    canonical = f"{p.name}/{m.name}" if p else m.name
                    if canonical not in names:
                        names.append(canonical)

                return [click.shell_completion.CompletionItem(n) for n in names if n.startswith(incomplete)]
        except Exception:
            pass
        return []


def _set_default_model(cliver: Cliver, name: str = None):
    """Set or show the default LLM model."""
    store = _get_store(cliver)
    if not name:
        try:
            default = store.get_default_model()
            if default:
                provider = store.get_provider(default.provider_id)
                canonical = f"{provider.name}/{default.name}" if provider else default.name
                cliver.output(f"Current default model: {canonical}")
            else:
                cliver.output("No default model set.")
        except Exception:
            cliver.output("No default model set.")

        models = store.list_models()
        if models:
            providers = {p.id: p for p in store.list_providers()}
            names = []
            for m in models:
                p = providers.get(m.provider_id)
                names.append(f"{p.name}/{m.name}" if p else m.name)
            cliver.output(f"Available models: {', '.join(names)}")
        return

    model_obj, provider = _resolve_model(store, name)
    if not model_obj:
        cliver.output(f"Model '{name}' not found.")
        models = store.list_models()
        if models:
            providers = {p.id: p for p in store.list_providers()}
            names = []
            for m in models:
                p = providers.get(m.provider_id)
                names.append(f"{p.name}/{m.name}" if p else m.name)
            cliver.output(f"Available models: {', '.join(names)}")
        return

    store.set_default_model(model_obj.id)
    canonical = f"{provider.name}/{model_obj.name}"
    cliver.output(f"Default model set to: {canonical}")


def _remove_model(cliver: Cliver, name: str):
    """Remove a LLM model."""
    store = _get_store(cliver)
    model_obj, provider = _resolve_model(store, name)
    if not model_obj:
        cliver.output(f"No LLM Model found with name: {name}")
        return
    canonical = f"{provider.name}/{model_obj.name}"
    store.delete_model(model_obj.id)
    cliver.output(f"Removed LLM Model: {canonical}")


def _add_model(
    cliver: Cliver,
    provider_id: str,
    endpoint_id: str,
    name: str,
    option: tuple = None,
    capabilities: str = None,
):
    """Add a new LLM model to a provider."""
    store = _get_store(cliver)
    provider = store.get_provider(provider_id)
    if not provider:
        cliver.output(f"Provider '{provider_id}' not found.")
        return

    canonical = f"{provider.name}/{name}"
    existing_models = store.list_models()
    for m in existing_models:
        p = store.get_provider(m.provider_id)
        if p and f"{p.name}/{m.name}" == canonical:
            cliver.output(f"Model '{canonical}' already exists.")
            return

    options_dict = {}
    if option:
        options_dict = parse_key_value_options(option, cliver.console)

    caps_list = None
    if capabilities:
        caps_list = [c.strip() for c in capabilities.split(",") if c.strip()]

    store.create_model(
        provider_id=provider_id,
        endpoint_id=endpoint_id,
        name=name,
        capabilities=caps_list,
        options=options_dict or None,
    )
    cliver.output(f"Added LLM Model: {canonical}")


def _update_model(
    cliver: Cliver,
    name: str,
    option: tuple = None,
    capabilities: str = None,
):
    """Update an existing LLM model."""
    store = _get_store(cliver)
    model_obj, provider = _resolve_model(store, name)
    if not model_obj:
        cliver.output(f"LLM Model '{name}' not found.")
        return

    kwargs = {}
    if option:
        options_dict = parse_key_value_options(option, cliver.console)
        if options_dict:
            kwargs["options"] = options_dict
    if capabilities:
        caps_list = [c.strip() for c in capabilities.split(",") if c.strip()]
        kwargs["capabilities"] = caps_list

    store.update_model(model_obj.id, **kwargs)
    canonical = f"{provider.name}/{model_obj.name}"
    cliver.output(f"LLM Model: {canonical} updated")


# ---------------------------------------------------------------------------
# Dispatch function
# ---------------------------------------------------------------------------

_SUBCOMMANDS: dict[str, click.Command] = {}


def dispatch(cliver: Cliver, args: str):
    """Manage LLM models — list, add, set, remove, default."""
    parts = args.strip().split(None, 1) if args.strip() else []
    sub = parts[0] if parts else "list"
    rest = parts[1] if len(parts) > 1 else ""

    if sub in ("--help", "-h", "help"):
        cliver.output(click_help(model, "/model"))
        return

    if sub in _SUBCOMMANDS and wants_help(rest):
        cliver.output(click_help(_SUBCOMMANDS[sub], f"/model {sub}"))
        return

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
            cliver.output(click_help(_SUBCOMMANDS["remove"], "/model remove"))
            return
        _remove_model(cliver, name)
    else:
        _show_model_detail(cliver, sub)


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
        "--endpoint": "endpoint",
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
    provider_id = opts.get("provider")
    endpoint_id = opts.get("endpoint")
    if not name or not provider_id or not endpoint_id:
        cliver.output(
            "Usage: /model add --provider PROVIDER_ID --endpoint ENDPOINT_ID --name NAME "
            "[--capabilities CAPS] [--option K=V]"
        )
        return
    _add_model(cliver, provider_id, endpoint_id, name, option=opts.get("option"), capabilities=opts.get("capabilities"))


def _dispatch_set(cliver: Cliver, rest: str) -> None:
    """Handle /model set from TUI dispatch."""
    opts = _parse_model_flags(rest)
    name = opts.get("name")
    if not name:
        cliver.output("Usage: /model set --name NAME [--option K=V] [--capabilities CAPS]")
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
    help="Provider ID. Must exist in 'provider list' (e.g. 'deepseek')",
)
@click.option(
    "--endpoint",
    type=str,
    required=True,
    help="Endpoint ID. Must exist in 'endpoint list' for the provider",
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
def add_llm_model(cliver: Cliver, name: str, provider: str, endpoint: str, option: tuple, capabilities: str):
    _add_model(cliver, provider, endpoint, name, option, capabilities)


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


# Populate subcommand map for dispatch help generation.
_SUBCOMMANDS.update(
    {
        "list": list_llm_models,
        "default": set_default_model,
        "add": add_llm_model,
        "set": update_llm_model,
        "remove": remove_llm_model,
    }
)
