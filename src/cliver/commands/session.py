"""
Session option management commands for interactive chat sessions.

Renamed from 'session' to 'session-option' to make room for future
session management (persistence, history, resume).
"""

import click

from cliver.cli import Cliver, pass_cliver
from cliver.config import ModelOptions
from cliver.util import parse_key_value_options


def _display_options(cliver: Cliver) -> None:
    """Display current session options."""
    click.echo("Current session options:")
    for key, value in cliver.session_options.items():
        if value:
            if key == "options":
                click.echo(f"  {key}: {dict(value)}")
            else:
                click.echo(f"  {key}: {value}")


@click.group(
    name="session-option",
    help="Manage inference options for the interactive session",
    invoke_without_command=True,
)
@pass_cliver
@click.pass_context
def session_option(ctx, cliver: Cliver):
    """View or modify persistent inference options for the current session."""
    if not hasattr(cliver, "session_options"):
        click.echo("Session options not available in this context.")
        ctx.exit(1)
        return

    # If no subcommand given, display current options
    if ctx.invoked_subcommand is None:
        _display_options(cliver)


@session_option.command(name="set", help="Set inference options for this session")
@click.option("--model", "-m", type=str, help="LLM model to use")
@click.option("--temperature", type=float, help="Temperature parameter")
@click.option("--max-tokens", type=int, help="Maximum number of tokens")
@click.option("--top-p", type=float, help="Top-p parameter")
@click.option("--frequency-penalty", type=float, help="Frequency penalty")
@click.option("--template", "-t", type=str, help="Template for prompts")
@click.option("--stream", "-s", is_flag=True, default=None, help="Enable streaming")
@click.option("--no-stream", is_flag=True, default=None, help="Disable streaming")
@click.option("--save-media", "-sm", is_flag=True, default=None, help="Enable media saving")
@click.option("--no-save-media", is_flag=True, default=None, help="Disable media saving")
@click.option("--media-dir", "-md", type=str, help="Directory for media files")
@click.option("--included-tools", type=str, help="Tool filter pattern")
@click.option("--option", multiple=True, type=str, help="Additional options (key=value)")
@pass_cliver
def set_options(
    cliver: Cliver,
    model,
    temperature,
    max_tokens,
    top_p,
    frequency_penalty,
    template,
    stream,
    no_stream,
    save_media,
    no_save_media,
    media_dir,
    included_tools,
    option,
):
    """Set one or more inference options for the current session."""
    options_provided = any(
        [
            model is not None,
            temperature is not None,
            max_tokens is not None,
            top_p is not None,
            frequency_penalty is not None,
            template is not None,
            stream is not None,
            no_stream is not None,
            save_media is not None,
            no_save_media is not None,
            media_dir is not None,
            included_tools is not None,
            len(option) > 0,
        ]
    )

    if not options_provided:
        _display_options(cliver)
        return 0

    _llm_options = cliver.session_options.get("options", {})

    if model is not None:
        if not cliver.config_manager.get_llm_model(model):
            click.echo(f"Unknown model: {model}, please define it first.")
            return 1
        cliver.session_options["model"] = model
        click.echo(f"Set model to '{model}' for this session.")

    if temperature is not None:
        _llm_options["temperature"] = temperature
        click.echo(f"Set temperature to {temperature} for this session.")

    if max_tokens is not None:
        _llm_options["max_tokens"] = max_tokens
        click.echo(f"Set max_tokens to {max_tokens} for this session.")

    if top_p is not None:
        _llm_options["top_p"] = top_p
        click.echo(f"Set top_p to {top_p} for this session.")

    if frequency_penalty is not None:
        _llm_options["frequency_penalty"] = frequency_penalty
        click.echo(f"Set frequency_penalty to {frequency_penalty} for this session.")

    if template is not None:
        cliver.session_options["template"] = template
        click.echo(f"Set template to '{template}' for this session.")

    if stream is True:
        cliver.session_options["stream"] = True
        click.echo("Enabled streaming for this session.")
    elif no_stream is True:
        cliver.session_options["stream"] = False
        click.echo("Disabled streaming for this session.")

    if save_media is True:
        cliver.session_options["save_media"] = True
        click.echo("Enabled save-media for this session.")
    elif no_save_media is True:
        cliver.session_options["save_media"] = False
        click.echo("Disabled save-media for this session.")

    if media_dir is not None:
        cliver.session_options["media_dir"] = media_dir
        click.echo(f"Set media_dir to '{media_dir}' for this session.")

    if included_tools is not None:
        cliver.session_options["included_tools"] = included_tools
        click.echo(f"Set included_tools to '{included_tools}' for this session.")

    if option and len(option) > 0:
        opts_dict = parse_key_value_options(option)
        _llm_options.update(opts_dict)
        click.echo(f"Updated additional options: {dict(opts_dict)}")

    return 0


@session_option.command(name="reset", help="Reset all session options to defaults")
@pass_cliver
def reset_options(cliver: Cliver):
    """Reset all session options to their defaults."""
    default_options = ModelOptions()
    cliver.session_options = {
        "model": cliver.config_manager.get_llm_model(),
        "temperature": default_options.temperature,
        "max_tokens": default_options.max_tokens,
        "top_p": default_options.top_p,
        "frequency_penalty": default_options.frequency_penalty,
        "options": {},
        "template": None,
        "stream": False,
        "save_media": False,
        "media_dir": None,
        "included_tools": None,
    }
    click.echo("Session options have been reset to defaults.")
    return 0


# Alias for auto-discovery (commands/__init__.py looks for module.{filename})
session = session_option
