import click
from rich import box
from rich.table import Table

from cliver.cli import Cliver, pass_cliver
from cliver.util import parse_key_value_options
from cliver.model_capabilities import ProviderEnum, ModelCapability


@click.group(name="llm", help="Manage LLM Models")
@click.pass_context
def llm(ctx: click.Context):
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())
        ctx.exit()


# noinspection PyUnresolvedReferences
@llm.command(name="list", help="List LLM Models")
@pass_cliver
def list_llm_models(cliver: Cliver):
    models = cliver.config_manager.list_llm_models()
    if models:
        table = Table(title="Configured LLM Models", box=box.SQUARE)
        table.add_column("Name", style="green")
        table.add_column("Name In Provider", style="green")
        table.add_column("Provider")
        table.add_column("URL")
        table.add_column("Capabilities", style="blue")
        table.add_column("File Upload", style="yellow")
        for _, model in models.items():
            # Get model capabilities
            capabilities = model.get_capabilities()

            # Format capabilities as a comma-separated string
            capabilities_str = (
                ", ".join([cap.value for cap in capabilities])
                if capabilities
                else "N/A"
            )

            # Check if file upload is supported
            file_upload_supported = ModelCapability.FILE_UPLOAD in capabilities

            table.add_row(
                model.name,
                model.name_in_provider,
                model.provider,
                model.url,
                capabilities_str,
                "Yes" if file_upload_supported else "No",
            )
        cliver.console.print(table)
    else:
        cliver.console.print("No LLM Models configured.")


# noinspection PyUnresolvedReferences
@llm.command(name="remove", help="Remove a LLM Model")
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
        cliver.console.print(f"No LLM Model found with name: {name}")
        return
    cliver.config_manager.remove_llm_model(name)
    cliver.console.print(f"Removed LLM Model: {name}")


# noinspection PyUnresolvedReferences
@llm.command(name="add", help="Add a LLM Model")
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
    type=click.Choice([p.value for p in ProviderEnum]),
    required=True,
    help="The provider of the LLM Model",
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
    required=True,
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
        cliver.console.print(f"LLM Model found with name: {name} already exists.")
        return

    # Convert key=value options to JSON string
    options_dict = {}
    if option:
        options_dict = parse_key_value_options(option, cliver.console)

    cliver.config_manager.add_or_update_llm_model(
        name, provider, api_key, url, options_dict, name_in_provider, capabilities
    )
    cliver.console.print(f"Added LLM Model: {name}")


# noinspection PyUnresolvedReferences
@llm.command(name="set", help="Update a LLM Model")
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
    type=click.Choice([p.value for p in ProviderEnum]),
    help="The provider of the LLM Model",
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
        cliver.console.print(f"LLM Model with name: {name} was not found.")
        return

    # Convert key=value options to JSON string
    options_dict = {}
    if option:
        options_dict = parse_key_value_options(option, cliver.console)

    cliver.config_manager.add_or_update_llm_model(
        name, provider, api_key, url, options_dict, name_in_provider, capabilities
    )
    cliver.console.print(f"LLM Model: {name} updated")