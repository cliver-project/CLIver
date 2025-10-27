import asyncio
import logging
import sys
import time
from typing import Any, Callable, Dict, List, Optional

import click

from cliver.cli import Cliver, interact, pass_cliver
from cliver.llm import TaskExecutor
from cliver.media_handler import MultimediaResponseHandler
from cliver.model_capabilities import ModelCapability
from cliver.util import create_tools_filter, parse_key_value_options, read_file_content

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
    "--stream",
    "-s",
    is_flag=True,
    default=False,
    help="Stream the response",
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
    "--skill-set",
    "-ss",
    multiple=True,
    help="Skill sets to apply to the chat session",
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
    "--debug",
    is_flag=True,
    default=False,
    help="Enable debug logging to show detailed logs in console",
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
    skill_set: List[str],
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
            click.echo(f"Error reading system message file {system_message_file}: {e}")
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

    # now we can get from the session_options if any
    try:
        _session_options = cliver.session_options or {}
        use_model = _session_options.get("model", model)
        task_executor = cliver.task_executor
        llm_engine = task_executor.get_llm_engine(use_model)
        if not llm_engine:
            # print message that model is not found
            click.echo(f"Model '{use_model}' not found.")
            return 1

        # Check capabilities before making calls
        if len(file) > 0 and not llm_engine.supports_capability(ModelCapability.FILE_UPLOAD):
            logger.debug(
                "Model '%s' does not support file uploads. Will embed file contents in the prompt.",
                use_model,
            )

        if len(image) > 0 and not llm_engine.supports_capability(ModelCapability.IMAGE_TO_TEXT):
            click.echo(f"Model '{use_model}' does not support image processing.")
            return 1

        if len(audio) > 0 and not llm_engine.supports_capability(ModelCapability.AUDIO_TO_TEXT):
            click.echo(f"Model '{use_model}' does not support audio processing.")
            return 1

        if len(video) > 0 and not llm_engine.supports_capability(ModelCapability.VIDEO_TO_TEXT):
            click.echo(f"Model '{use_model}' does not support video processing.")
            return 1

        # Convert param tuples to dictionary
        params = parse_key_value_options(param, cliver.console)
        use_stream = _session_options.get("stream", stream)
        use_included_tools = _session_options.get("included_tools", included_tools)
        _llm_options: dict = _session_options.get("options", {})
        _llm_options.update(options)
        tools_filter = None
        if use_included_tools and len(use_included_tools.strip()) > 0:
            tools_filter = create_tools_filter(use_included_tools)

        system_message_appender = None
        # Get system message appender from session options if not already set
        if not system_message_content or len(system_message_content.strip()) == 0:
            system_message_content = _session_options.get("system_message_content")
        if system_message_content and len(system_message_content.strip()) > 0:

            def system_message_appender():
                return system_message_content

        return _async_chat(
            task_executor,
            sentence,
            use_model,
            use_stream,
            image,
            audio,
            video,
            file,
            skill_set,
            template,
            params,
            save_media,
            media_dir,
            _llm_options,
            tools_filter,
            system_message_appender,
        )
    except Exception as e:
        click.echo(f"Error: {e}")
        return 1


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


def _async_chat(
    task_executor: TaskExecutor,
    user_input: str,
    model: str,
    stream: bool = False,
    images: List[str] = None,
    audio_files: List[str] = None,
    video_files: List[str] = None,
    files: List[str] = None,
    skill_sets: List[str] = None,
    template: Optional[str] = None,
    params: dict = None,
    save_media: bool = False,
    media_dir: Optional[str] = None,
    options: Dict[str, Any] = None,
    tools_filter: Optional[Callable] = None,
    system_message_appender=None,
):
    # Create multimedia response handler
    response_handler = MultimediaResponseHandler(media_dir)

    try:
        if stream:
            # For streaming, we need to run the async generator
            return asyncio.run(
                _stream_chat(
                    task_executor,
                    user_input,
                    images,
                    audio_files,
                    video_files,
                    files,
                    model,
                    skill_sets,
                    template,
                    params,
                    save_media,
                    media_dir,
                    options,
                    tools_filter,
                    system_message_appender,
                )
            )
        else:
            response = task_executor.process_user_input_sync(
                user_input=user_input,
                images=images,
                audio_files=audio_files,
                video_files=video_files,
                files=files,
                model=model,
                skill_sets=skill_sets,
                template=template,
                params=params,
                options=options,
                filter_tools=tools_filter,
                system_message_appender=system_message_appender,
            )
            if response:
                # Get the LLM engine used for this response
                llm_engine = task_executor.get_llm_engine(model)

                # Process response with multimedia handler
                multimedia_response = response_handler.process_response(
                    response, llm_engine=llm_engine, auto_save_media=save_media
                )

                # Display text content
                if multimedia_response.has_text():
                    click.echo(multimedia_response.text_content)

                # Display media information
                if multimedia_response.has_media():
                    media_count = len(multimedia_response.media_content)
                    click.echo(f"\n[Media Content: {media_count} items]")
                    for i, media in enumerate(multimedia_response.media_content):
                        info = f"  {i + 1}. {media.type.value}"
                        if media.filename:
                            info += f" ({media.filename})"
                        if media.mime_type:
                            info += f" [{media.mime_type}]"
                        click.echo(info)
    except ValueError as e:
        if "File upload is not supported" in str(e):
            click.echo(f"Error: {e}")
            click.echo("Will use content embedding as fallback.")
        else:
            raise
        return 1
    except Exception as e:
        click.echo(f"Error: {e}")
        return 1
    return 0


async def _stream_chat(
    task_executor: TaskExecutor,
    user_input: str,
    images: List[str] = None,
    audio_files: List[str] = None,
    video_files: List[str] = None,
    files: List[str] = None,
    model: str = None,
    skill_sets: List[str] = None,
    template: Optional[str] = None,
    params: dict = None,
    save_media: bool = False,
    media_dir: Optional[str] = None,
    options: Dict[str, Any] = None,
    tools_filter: Optional[Callable] = None,
    system_message_appender=None,
):
    """Stream the chat response character by character."""
    # Create multimedia response handler
    response_handler = MultimediaResponseHandler(media_dir)

    try:
        accumulated_chunk = None
        async for chunk in task_executor.stream_user_input(
            user_input=user_input,
            images=images,
            audio_files=audio_files,
            video_files=video_files,
            files=files,
            model=model,
            skill_sets=skill_sets,
            template=template,
            params=params,
            options=options,
            filter_tools=tools_filter,
            system_message_appender=system_message_appender,
        ):
            # Handle different types of chunks - some may have content, some may have
            # tool calls
            if accumulated_chunk is None:
                accumulated_chunk = chunk
            else:
                # Concatenate the chunks using the + operator
                accumulated_chunk = accumulated_chunk + chunk

            if hasattr(chunk, "content") and chunk.content:
                # For streaming, we accumulate content and display it as it comes
                chunk_content = str(chunk.content)
                # Print each character with a small delay to simulate streaming
                for char in chunk_content:
                    sys.stdout.write(char)
                    sys.stdout.flush()
                    # Reduced delay for better streaming experience
                    time.sleep(0.005)  # Faster streaming

        # After streaming is complete, process the accumulated content
        if accumulated_chunk:
            # Get the LLM engine used for this response
            llm_engine = task_executor.get_llm_engine(model)
            multimedia_response = response_handler.process_response(
                accumulated_chunk, llm_engine=llm_engine, auto_save_media=save_media
            )

            # Display media information if any
            if multimedia_response.has_media():
                print()  # New line
                media_count = len(multimedia_response.media_content)
                print(f"\n[Media Content: {media_count} items]")
                for i, media in enumerate(multimedia_response.media_content):
                    info = f"  {i + 1}. {media.type.value}"
                    if media.filename:
                        info += f" ({media.filename})"
                    if media.mime_type:
                        info += f" [{media.mime_type}]"
                    print(info)

        print()  # New line at the end
        return 0
    except ValueError as e:
        if "File upload is not supported" in str(e):
            click.echo(f"Error: {e}")
            click.echo("Will use content embedding as fallback.")
        else:
            raise
        return 1
    except Exception as e:
        click.echo(f"Error: {e}")
        return 1
