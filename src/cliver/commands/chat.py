import logging
from typing import Any, List, Optional

import click

from cliver.cli import Cliver, interact, pass_cliver
from cliver.llm.errors import TaskTimeoutError
from cliver.util import parse_key_value_options, read_file_content

logger = logging.getLogger(__name__)


@click.command(name="chat", help="Chat with LLM models")
@click.option(
    "--model",
    "-m",
    type=str,
    required=False,
    help="Which LLM model to use",
)
@click.option(
    "--stream/--no-stream",
    "-s/-S",
    default=True,
    help="Stream the response (default: enabled)",
)
@click.option(
    "--image",
    "-img",
    multiple=True,
    help="Image files to send with the message",
)
@click.option(
    "--audio",
    "-aud",
    multiple=True,
    help="Audio files to send with the message",
)
@click.option(
    "--video",
    "-vid",
    multiple=True,
    help="Video files to send with the message",
)
@click.option(
    "--file",
    "-f",
    multiple=True,
    help="General files to upload for tools like code interpreter",
)
@click.option(
    "--save-media",
    "-sm",
    is_flag=True,
    default=False,
    help="Automatically save media content from responses to files",
)
@click.option(
    "--media-dir",
    "-md",
    type=str,
    default=None,
    help="Directory to save media files (default: current directory)",
)
@click.option(
    "--template",
    "-t",
    type=str,
    help="Template to use for the prompt",
)
@click.option(
    "--param",
    "-p",
    multiple=True,
    type=str,
    help="Parameters for skill sets and templates (key=value)",
)
@click.option(
    "--temperature",
    type=float,
    help="Temperature parameter for the LLM (Inference Options)",
)
@click.option(
    "--max-tokens",
    type=int,
    help="Maximum number of tokens for the LLM (Inference Options)",
)
@click.option(
    "--top-p",
    type=float,
    help="Top-p parameter for the LLM (Inference Options)",
)
@click.option(
    "--frequency-penalty",
    type=float,
    help="Frequency penalty for the LLM (Inference Options)",
)
@click.option(
    "--option",
    multiple=True,
    type=str,
    help="Additional inference options in key=value format (Inference Options)",
)
@click.option(
    "--system-message",
    type=str,
    help="System message to append to the conversation",
)
@click.option(
    "--system-message-file",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    help="File containing system message to append to the conversation",
)
@click.option(
    "--timeout",
    type=int,
    default=None,
    help="Wall-clock timeout in seconds for the entire agent run",
)
@click.option(
    "--output",
    "output_format",
    type=click.Choice(["text", "json"]),
    default="text",
    help="Output format: text (default) or json for machine-readable output",
)
@click.option(
    "--permission-mode",
    type=click.Choice(["default", "auto-edit", "yolo"]),
    default=None,
    help="Permission mode for this run (overrides config)",
)
@click.option(
    "--allow-tool",
    multiple=True,
    type=str,
    help="Pre-grant a tool for this run (repeatable, supports regex patterns)",
)
@click.option(
    "--debug",
    is_flag=True,
    default=False,
    help="Enable debug logging to show detailed logs in console",
)
@click.option(
    "--no-fallback",
    is_flag=True,
    default=False,
    help="Disable automatic model fallback on failure",
)
@click.option(
    "--included-tools",
    type=str,
    help="Pattern to filter which tools to include (supports wildcards like * and ?)",
)
@click.argument("query", nargs=-1)
@pass_cliver
def chat(
    cliver: Cliver,
    model: Optional[str],
    stream: bool,
    image: List[str],
    audio: List[str],
    video: List[str],
    file: List[str],
    template: Optional[str],
    param: Optional[tuple],
    save_media: bool,
    media_dir: Optional[str],
    query: tuple,
    temperature: Optional[float],
    max_tokens: Optional[int],
    top_p: Optional[float],
    frequency_penalty: Optional[float],
    option: Optional[tuple],
    system_message: Optional[str] = None,
    system_message_file: Optional[str] = None,
    timeout: Optional[int] = None,
    output_format: str = "text",
    permission_mode: Optional[str] = None,
    allow_tool: tuple = (),
    no_fallback: bool = False,
    debug: bool = False,
    included_tools: Optional[str] = None,
):
    """
    Chat with LLM models.
    """
    if debug:
        # Enable debug logging
        logging.basicConfig(level=logging.DEBUG)
        logger.debug("Debug logging enabled for chat command")

    # Apply CLI permission overrides
    if permission_mode:
        from cliver.permissions import PermissionMode

        cliver.permission_manager.set_mode(PermissionMode(permission_mode))
    if allow_tool:
        from cliver.permissions import PermissionAction

        for tool_pattern in allow_tool:
            cliver.permission_manager.grant_session(tool_pattern, PermissionAction.ALLOW)

    # Collect all the LLM configuration options into a dictionary
    options = _chat_options(frequency_penalty, max_tokens, temperature, top_p)

    # Process additional options provided via --option flag (key=value format)
    if option and len(option) > 0:
        opts_dict = parse_key_value_options(option)
        options.update(opts_dict)

    # Process system message - prioritize system_message_file over system_message
    system_message_content = None
    if system_message_file:
        try:
            system_message_content = read_file_content(system_message_file)
        except Exception as e:
            cliver.output(f"Error reading system message file {system_message_file}: {e}")
            return 1
    elif system_message:
        system_message_content = system_message

    # Join the query tuple into a single string to check if it's empty
    sentence = " ".join(query) if query else ""
    if len(sentence.strip()) == 0:
        # no query, let's start an interactive mode
        # in interactive mode, we can cache the settings from the argument above
        session_options: dict[str, Any] = {}
        if model:
            session_options["model"] = model
        if stream:
            session_options["stream"] = stream
        if included_tools:
            session_options["included_tools"] = included_tools
        if options and len(options) > 0:
            session_options["options"] = options

        # Add system message to session options if provided
        if system_message_content:
            session_options["system_message_content"] = system_message_content
        interact(cliver, session_options)
        return 0

    # Delegate to shared run_chat implementation
    import time as _time

    from cliver.commands.chat_handler import ChatOptions, run_chat

    start_time = _time.monotonic()
    _real_stdout = None

    # Setup JSON output mode if needed
    if output_format == "json":
        import io
        import sys

        from rich.console import Console

        _real_stdout = sys.stdout
        cliver.console = Console(file=io.StringIO(), quiet=True)
        if hasattr(cliver, "thinking") and cliver.thinking:
            cliver.thinking = None

    # Build ChatOptions
    chat_opts = ChatOptions(
        text=sentence,
        model=model,
        stream=stream,
        images=list(image),
        audio_files=list(audio),
        video_files=list(video),
        files=list(file),
        template=template,
        params=parse_key_value_options(param, cliver.console),
        options=options,
        save_media=save_media,
        media_dir=media_dir,
        included_tools=included_tools,
        system_message_content=system_message_content,
        timeout_s=timeout,
        auto_fallback=not no_fallback,
    )

    try:
        result = run_chat(cliver, chat_opts)
    except TaskTimeoutError as e:
        if output_format == "json":
            import sys as _sys

            duration = _time.monotonic() - start_time
            _emit_json_result(
                _real_stdout or _sys.stdout,
                False,
                e.partial_result or "",
                model,
                cliver.token_tracker,
                duration,
                error=str(e),
                timeout=True,
            )
            return 2
        cliver.output(f"[yellow]Timeout: {e}[/yellow]")
        if e.partial_result:
            cliver.output(f"\nPartial result:\n{e.partial_result}")
        return 2
    except Exception as e:
        if output_format == "json":
            import sys as _sys

            duration = _time.monotonic() - start_time
            _emit_json_result(
                _real_stdout or _sys.stdout,
                False,
                "",
                model,
                cliver.token_tracker,
                duration,
                error=str(e),
            )
            return 1
        cliver.output(f"[red]Error: {e}[/red]")
        return 1

    if output_format == "json":
        import sys as _sys

        duration = _time.monotonic() - start_time
        _emit_json_result(
            _real_stdout or _sys.stdout,
            result.success,
            result.text or "",
            model,
            cliver.token_tracker,
            duration,
            error=result.error,
        )
    return 0 if result.success else 1


def _chat_options(
    frequency_penalty: float | None,
    max_tokens: int | None,
    temperature: float | None,
    top_p: float | None,
) -> dict[Any, Any]:
    options = {}
    if temperature is not None:
        options["temperature"] = temperature
    if max_tokens is not None:
        options["max_tokens"] = max_tokens
    if top_p is not None:
        options["top_p"] = top_p
    if frequency_penalty is not None:
        options["frequency_penalty"] = frequency_penalty
    return options


def _emit_json_result(
    stdout,
    success: bool,
    output: str,
    model: str | None,
    token_tracker,
    duration: float,
    error: str | None = None,
    timeout: bool = False,
):
    """Emit a single JSON result object to stdout."""
    import json

    tokens = None
    if token_tracker and hasattr(token_tracker, "last_usage") and token_tracker.last_usage:
        last = token_tracker.last_usage
        tokens = {
            "input": last.input_tokens,
            "output": last.output_tokens,
            "total": last.total_tokens,
        }

    data = {
        "success": success,
        "output": output,
        "error": error or "",
        "model": model,
        "tokens": tokens,
        "duration_s": round(duration, 2),
    }
    if timeout:
        data["timeout"] = True

    stdout.write(json.dumps(data) + "\n")
    stdout.flush()
