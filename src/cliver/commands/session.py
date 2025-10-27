"""
Session management commands for interactive chat sessions.
"""

import click

from cliver.cli import Cliver, pass_cliver
from cliver.config import ModelOptions
from cliver.util import parse_key_value_options

LLM_OPTIONS_KEYS: list[str] = [
    "temperature",
    "max-tokens",
    "top-p",
    "frequency-penalty",
]


@click.command(name="session", help="Manage persistent options for the interactive session")
@click.option(
    "--model",
    "-m",
    type=str,
    required=False,
    help="Set the LLM model to use for this session",
)
@click.option(
    "--temperature",
    type=float,
    help="Set temperature parameter for the LLM",
)
@click.option(
    "--max-tokens",
    type=int,
    help="Set maximum number of tokens for the LLM",
)
@click.option(
    "--top-p",
    type=float,
    help="Set top-p parameter for the LLM",
)
@click.option(
    "--frequency-penalty",
    type=float,
    help="Set frequency penalty for the LLM",
)
@click.option(
    "--skill-set",
    "-ss",
    multiple=True,
    help="Set skill sets to apply for this session",
)
@click.option(
    "--template",
    "-t",
    type=str,
    help="Set template to use for prompts in this session",
)
@click.option(
    "--stream",
    "-s",
    is_flag=True,
    default=None,
    help="Enable streaming response",
)
@click.option(
    "--no-stream",
    is_flag=True,
    default=None,
    help="Disable streaming response",
)
@click.option(
    "--save-media",
    "-sm",
    is_flag=True,
    default=None,
    help="Enable automatic saving of media content",
)
@click.option(
    "--no-save-media",
    is_flag=True,
    default=None,
    help="Disable automatic saving of media content",
)
@click.option(
    "--media-dir",
    "-md",
    type=str,
    help="Set directory to save media files",
)
@click.option(
    "--included-tools",
    type=str,
    help="Set pattern to filter which tools to include",
)
@click.option(
    "--option",
    multiple=True,
    type=str,
    help="Set additional inference options in key=value format",
)
@click.option(
    "--reset",
    is_flag=True,
    default=False,
    help="Reset all session options to defaults",
)
@click.option(
    "--list",
    "list_options",
    is_flag=True,
    default=False,
    help="List current session options",
)
@pass_cliver
def session(
    cliver: Cliver,
    model,
    temperature,
    max_tokens,
    top_p,
    frequency_penalty,
    skill_set,
    template,
    stream,
    no_stream,
    save_media,
    no_save_media,
    media_dir,
    included_tools,
    option,
    reset,
    list_options,
):
    """
    Manage session options for the interactive chat session.

    This command allows you to view and modify persistent options that will be used
    for all subsequent chat commands during the current interactive session.
    """
    if not hasattr(cliver, "session_options"):
        click.echo("Session options not available in this context.")
        return 1

    if reset:
        # Reset all session options to defaults
        default_options = ModelOptions()
        cliver.session_options = {
            "model": cliver.config_manager.get_llm_model(),
            "temperature": default_options.temperature,
            "max_tokens": default_options.max_tokens,
            "top_p": default_options.top_p,
            "frequency_penalty": default_options.frequency_penalty,
            "options": {},
            "skill_sets": [],
            "template": None,
            "stream": False,
            "save_media": False,
            "media_dir": None,
            "included_tools": None,
        }
        click.echo("Session options have been reset to defaults.")
        return 0

    if list_options:
        # List all current session options
        click.echo("Current session options:")
        for key, value in cliver.session_options.items():
            if value:
                if key == "options":
                    if value:
                        click.echo(f"  {key}: {dict(value)}")
                    else:
                        click.echo(f"  {key}: (empty)")
                else:
                    click.echo(f"  {key}: {value}")
        return 0

    # Check if any options were provided to set
    options_provided = any(
        [
            model is not None,
            temperature is not None,
            max_tokens is not None,
            top_p is not None,
            frequency_penalty is not None,
            len(skill_set) > 0,
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
        # If no options provided but no reset/list, show current options
        click.echo("Current session options:")
        for key, value in cliver.session_options.items():
            if value:
                if key == "options":
                    if value:
                        click.echo(f"  {key}: {dict(value)}")
                    else:
                        click.echo(f"  {key}: (empty)")
                else:
                    click.echo(f"  {key}: {value}")
        return 0

    _llm_options = cliver.session_options.get("options", {})
    # Update session options if provided
    if model is not None:
        if not cliver.config_manager.get_llm_model(model):
            click.echo(f"Unknown of model: {model}, please define it first.")
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

    if len(skill_set) > 0:
        cliver.session_options["skill_sets"] = list(skill_set)
        click.echo(f"Set skill_sets to {list(skill_set)} for this session.")

    if template is not None:
        cliver.session_options["template"] = template
        click.echo(f"Set template to '{template}' for this session.")

    # Handle stream options
    if stream is True:
        cliver.session_options["stream"] = True
        click.echo("Enabled streaming for this session.")
    elif no_stream is True:
        cliver.session_options["stream"] = False
        click.echo("Disabled streaming for this session.")

    # Handle save-media options
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

    # Process additional options
    if option and len(option) > 0:
        opts_dict = parse_key_value_options(option)
        _llm_options.update(opts_dict)
        click.echo(f"Updated additional options: {dict(opts_dict)}")

    return 0
