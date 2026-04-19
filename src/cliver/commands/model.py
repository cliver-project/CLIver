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


# noinspection PyUnresolvedReferences
@model.command(name="list", help="List LLM Models")
@pass_cliver
def list_llm_models(cliver: Cliver):
    models = cliver.config_manager.list_llm_models()
    if models:
        table = Table(title="Configured LLM Models", box=box.SQUARE)
        table.add_column("", min_width=1, max_width=1, no_wrap=True)
        table.add_column("Name", style="green")
        table.add_column("Name In Provider", style="green")
        table.add_column("Provider", style="cyan")
        table.add_column("URL", style="dim")
        table.add_column("Capabilities", style="blue")

        global_default = cliver.config_manager.config.default_model
        session_model = (cliver.session_options or {}).get("model")
        active_model = session_model or global_default

        for _, model in models.items():
            # Get model capabilities and show only modality keywords
            capabilities = model.get_capabilities()
            capabilities_str = _format_modalities(capabilities) if capabilities else "N/A"

            # Mark the active model with a green checkmark
            marker = "[bold green]✔[/bold green]" if model.name == active_model else ""

            table.add_row(
                marker,
                model.name,
                model.name_in_provider,
                model.provider,
                model.url,
                capabilities_str,
            )
        cliver.output(table)
    else:
        cliver.output("No LLM Models configured.")


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
    return ", ".join(modalities) if modalities else "N/A"


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


# noinspection PyUnresolvedReferences
@model.command(name="default", help="Set the default LLM model")
@click.argument("name", type=ModelNameType(), required=False)
@pass_cliver
def set_default_model(cliver: Cliver, name: str):
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
    model = cliver.config_manager.get_llm_model(name)
    if not model:
        cliver.output(f"No LLM Model found with name: {name}")
        return
    cliver.config_manager.remove_llm_model(name)
    cliver.output(f"Removed LLM Model: {name}")


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
