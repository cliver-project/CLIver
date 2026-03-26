import asyncio
import logging
from typing import Any, List, Optional

import click
from langchain_core.messages import AIMessage, HumanMessage

from cliver.cli import Cliver, interact, pass_cliver
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

    # now we can get from the session_options if any
    try:
        _session_options = cliver.session_options or {}
        use_model = _session_options.get("model", model)
        task_executor = cliver.task_executor
        llm_engine = task_executor.get_llm_engine(use_model)
        if not llm_engine:
            # print message that model is not found
            cliver.output(f"Model '{use_model}' not found.")
            return 1

        # Check capabilities before making calls
        if len(file) > 0 and not llm_engine.supports_capability(ModelCapability.FILE_UPLOAD):
            logger.debug(
                "Model '%s' does not support file uploads. Will embed file contents in the prompt.",
                use_model,
            )

        if len(image) > 0 and not llm_engine.supports_capability(ModelCapability.IMAGE_TO_TEXT):
            cliver.output(f"Model '{use_model}' does not support image processing.")
            return 1

        if len(audio) > 0 and not llm_engine.supports_capability(ModelCapability.AUDIO_TO_TEXT):
            cliver.output(f"Model '{use_model}' does not support audio processing.")
            return 1

        if len(video) > 0 and not llm_engine.supports_capability(ModelCapability.VIDEO_TO_TEXT):
            cliver.output(f"Model '{use_model}' does not support video processing.")
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

        # Record user turn to session (interactive mode)
        cliver.record_turn("user", sentence)

        # Compress conversation history if it exceeds context window budget
        model_config = task_executor._get_llm_model(use_model)
        if model_config and cliver.conversation_messages:
            _compress_if_needed(cliver, task_executor, model_config, use_model, sentence)

        # Snapshot prior turns (before this turn) to pass as history
        conv_history = list(cliver.conversation_messages) if cliver.conversation_messages else None

        # Append user turn now so it precedes the assistant response in order
        cliver.conversation_messages.append(HumanMessage(content=sentence))

        # Callback to record assistant response to session and conversation history
        def on_response(text: str):
            cliver.record_turn("assistant", text)
            cliver.conversation_messages.append(AIMessage(content=text))

        from cliver.cli_llm_call import LLMCallOptions, llm_call

        result = llm_call(
            cliver,
            LLMCallOptions(
                user_input=sentence,
                model=use_model,
                stream=use_stream,
                images=list(image),
                audio_files=list(audio),
                video_files=list(video),
                files=list(file),
                template=template,
                params=params,
                options=_llm_options,
                save_media=save_media,
                media_dir=media_dir,
                tools_filter=tools_filter,
                system_message_appender=system_message_appender,
                conversation_history=conv_history,
                on_response=on_response,
            ),
        )
        return 0 if result.success else 1
    except Exception as e:
        cliver.output(f"[red]Error: {e}[/red]")
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


def _compress_if_needed(cliver, task_executor, model_config, model_name, new_input):
    """Check and compress conversation history if it exceeds context window budget."""
    from cliver.conversation_compressor import ConversationCompressor, estimate_tokens, get_context_window

    context_window = get_context_window(model_config)
    compressor = ConversationCompressor(context_window)

    if not compressor.needs_compression([], cliver.conversation_messages, new_input):
        return

    before_tokens = estimate_tokens(cliver.conversation_messages)
    llm_engine = task_executor.get_llm_engine(model_name)

    try:
        compressed = asyncio.get_event_loop().run_until_complete(
            compressor.compress(cliver.conversation_messages, llm_engine)
        )
    except RuntimeError:
        # No running event loop — create one
        compressed = asyncio.run(compressor.compress(cliver.conversation_messages, llm_engine))

    cliver.conversation_messages = compressed
    after_tokens = estimate_tokens(cliver.conversation_messages)
    cliver.output(f"\\[Compressed conversation: ~{before_tokens} → ~{after_tokens} tokens]")
