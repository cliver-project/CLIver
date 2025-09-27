"""
Model capabilities command for Cliver client.
"""

import click
from rich.table import Table

from cliver.cli import Cliver, pass_cliver


@click.command()
@click.option("--model", "-m", help="Model name to check capabilities for")
@click.option(
    "--detailed", "-d", is_flag=True, help="Show detailed modality capabilities"
)
@pass_cliver
def capabilities(cli: Cliver, model: str = None, detailed: bool = False):
    """Display model capabilities."""
    models = cli.config_manager.list_llm_models()

    if not models:
        cli.console.print("[yellow]No models configured.[/yellow]")
        return

    if model:
        if model not in models:
            cli.console.print(f"[red]Model '{model}' not found.[/red]")
            return
        models = {model: models[model]}

    if detailed:
        table = Table(title="Detailed Model Capabilities")
        table.add_column("Model", style="cyan")
        table.add_column("Provider", style="magenta")
        table.add_column("Text", style="green")
        table.add_column("Image In", style="blue")
        table.add_column("Image Out", style="blue")
        table.add_column("Audio In", style="yellow")
        table.add_column("Audio Out", style="yellow")
        table.add_column("Video In", style="purple")
        table.add_column("Video Out", style="purple")
        table.add_column("Tools", style="red")

        for model_name, model_config in models.items():
            capabilities = model_config.get_capabilities()
            modality_caps = capabilities.get_modality_capabilities()
            table.add_row(
                model_name,
                model_config.provider,
                "✓" if modality_caps["text"] else "✗",
                "✓" if modality_caps["image_input"] else "✗",
                "✓" if modality_caps["image_output"] else "✗",
                "✓" if modality_caps["audio_input"] else "✗",
                "✓" if modality_caps["audio_output"] else "✗",
                "✓" if modality_caps["video_input"] else "✗",
                "✓" if modality_caps["video_output"] else "✗",
                "✓" if modality_caps["tool_calling"] else "✗",
            )
    else:
        table = Table(title="Model Capabilities")
        table.add_column("Model", style="cyan")
        table.add_column("Provider", style="magenta")
        table.add_column("Capabilities", style="green")

        for model_name, model_config in models.items():
            capabilities = model_config.get_capabilities()
            cap_names = [cap.value for cap in capabilities]
            cap_str = ", ".join(cap_names) if cap_names else "None"
            table.add_row(model_name, model_config.provider, cap_str)

    cli.console.print(table)
