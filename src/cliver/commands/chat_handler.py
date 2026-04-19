"""
Shared chat execution logic used by both CLI and TUI.

This module extracts the core chat logic from the Click command handler
so it can be reused by the TUI CommandDispatcher.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage

from cliver.cli_llm_call import LLMCallOptions, LLMCallResult, llm_call
from cliver.model_capabilities import ModelCapability
from cliver.util import create_tools_filter

if TYPE_CHECKING:
    from cliver.cli import Cliver

logger = logging.getLogger(__name__)


@dataclass
class ChatOptions:
    """Options for a chat call."""

    text: str
    model: Optional[str] = None
    stream: bool = True
    images: List[str] = field(default_factory=list)
    audio_files: List[str] = field(default_factory=list)
    video_files: List[str] = field(default_factory=list)
    files: List[str] = field(default_factory=list)
    template: Optional[str] = None
    params: Optional[Dict[str, str]] = None
    options: Optional[Dict[str, Any]] = None
    save_media: bool = False
    media_dir: Optional[str] = None
    included_tools: Optional[str] = None
    system_message_content: Optional[str] = None
    timeout_s: Optional[int] = None
    auto_fallback: bool = True
    on_pending_input: Optional[Callable[[], Optional[str]]] = None


def run_chat(cliver: "Cliver", opts: ChatOptions) -> LLMCallResult:
    """Execute a chat call with the LLM.

    This is the synchronous version that calls llm_call (which handles both
    streaming and non-streaming internally).

    Args:
        cliver: The Cliver CLI instance.
        opts: Chat options.

    Returns:
        LLMCallResult with the response text and metadata.
    """
    # Get model from session_options or opts
    _session_options = cliver.session_options or {}
    use_model = _session_options.get("model", opts.model)

    task_executor = cliver.task_executor
    llm_engine = task_executor.get_llm_engine(use_model)

    if not llm_engine:
        return LLMCallResult(success=False, error=f"Model '{use_model}' not found.")

    # Check capabilities
    error_msg = _check_capabilities(llm_engine, opts, use_model)
    if error_msg:
        return LLMCallResult(success=False, error=error_msg)

    # Get stream setting from session or opts
    use_stream = _session_options.get("stream", opts.stream)

    # Get included_tools from session or opts
    use_included_tools = _session_options.get("included_tools", opts.included_tools)
    tools_filter = None
    if use_included_tools and len(use_included_tools.strip()) > 0:
        tools_filter = create_tools_filter(use_included_tools)

    # Build system_message_appender
    system_message_appender = None
    system_message_content = opts.system_message_content
    # Get system message from session if not set
    if not system_message_content or len(system_message_content.strip()) == 0:
        system_message_content = _session_options.get("system_message_content")
    if system_message_content and len(system_message_content.strip()) > 0:

        def system_message_appender():
            return system_message_content

    # Get LLM options from session and merge with opts
    _llm_options: dict = _session_options.get("options", {})
    if opts.options:
        _llm_options.update(opts.options)

    # Record user turn to session (interactive mode)
    cliver.record_turn("user", opts.text)

    # Compress conversation history if needed
    model_config = task_executor._get_llm_model(use_model)
    if model_config and cliver.conversation_messages:
        _compress_if_needed(cliver, task_executor, model_config, use_model, opts.text)

    # Snapshot prior turns (before this turn) to pass as history
    conv_history = list(cliver.conversation_messages) if cliver.conversation_messages else None

    # Append user turn now so it precedes the assistant response in order
    cliver.conversation_messages.append(HumanMessage(content=opts.text))

    # Callback to record assistant response to session and conversation history
    def on_response(text: str):
        cliver.record_turn("assistant", text)
        cliver.conversation_messages.append(AIMessage(content=text))

    # Execute the LLM call
    result = llm_call(
        cliver,
        LLMCallOptions(
            user_input=opts.text,
            model=use_model,
            stream=use_stream,
            images=opts.images,
            audio_files=opts.audio_files,
            video_files=opts.video_files,
            files=opts.files,
            template=opts.template,
            params=opts.params,
            options=_llm_options,
            save_media=opts.save_media,
            media_dir=opts.media_dir,
            tools_filter=tools_filter,
            system_message_appender=system_message_appender,
            conversation_history=conv_history,
            on_response=on_response,
            timeout_s=opts.timeout_s,
            auto_fallback=opts.auto_fallback,
            on_pending_input=opts.on_pending_input,
        ),
    )

    return result


def _check_capabilities(llm_engine, opts: ChatOptions, model_name: str) -> Optional[str]:
    """Check if the model supports the requested capabilities.

    Args:
        llm_engine: The LLM engine instance.
        opts: Chat options.
        model_name: Name of the model.

    Returns:
        Error message if capabilities not supported, None otherwise.
    """
    if len(opts.files) > 0 and not llm_engine.supports_capability(ModelCapability.FILE_UPLOAD):
        logger.debug(
            "Model '%s' does not support file uploads. Will embed file contents in the prompt.",
            model_name,
        )

    if len(opts.images) > 0 and not llm_engine.supports_capability(ModelCapability.IMAGE_TO_TEXT):
        return f"Model '{model_name}' does not support image processing."

    if len(opts.audio_files) > 0 and not llm_engine.supports_capability(ModelCapability.AUDIO_TO_TEXT):
        return f"Model '{model_name}' does not support audio processing."

    if len(opts.video_files) > 0 and not llm_engine.supports_capability(ModelCapability.VIDEO_TO_TEXT):
        return f"Model '{model_name}' does not support video processing."

    return None


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
