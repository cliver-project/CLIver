"""
Reusable LLM call component with CLI presentation.

Handles the full request/response lifecycle including:
- Thinking spinner (start/stop)
- Streaming or sync LLM calls
- Token usage display
- Error handling

This is a CLI-layer component — AgentCore and API layer have no dependency on it.
"""

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from langchain_core.messages import BaseMessage

from cliver.llm.errors import TaskTimeoutError

if TYPE_CHECKING:
    from cliver.agent import Agent
    from cliver.cli import Cliver

logger = logging.getLogger(__name__)


def _response_color_start() -> str:
    from cliver.themes import get_theme

    return get_theme().response_ansi_start


def _response_color_reset() -> str:
    from cliver.themes import get_theme

    return get_theme().response_ansi_reset


@dataclass
class LLMCallResult:
    """Result of an LLM call."""

    success: bool = True
    text: str = ""
    response: Optional[BaseMessage] = None
    error: Optional[str] = None


@dataclass
class LLMCallOptions:
    """Options for an LLM call.

    Only set fields you need — defaults work for simple calls like skill generation.
    Full chat features (multimedia, templates) set the relevant fields.
    """

    user_input: str = ""
    model: Optional[str] = None
    stream: bool = True
    images: List[str] = field(default_factory=list)
    audio_files: List[str] = field(default_factory=list)
    video_files: List[str] = field(default_factory=list)
    files: List[str] = field(default_factory=list)
    template: Optional[str] = None
    params: Optional[dict] = None
    options: Optional[Dict[str, Any]] = None
    save_media: bool = False
    media_dir: Optional[str] = None
    skill_name: Optional[str] = None
    tools_filter: Optional[Callable] = None
    system_message_appender: Optional[Callable] = None
    conversation_history: Optional[List[BaseMessage]] = None
    on_response: Optional[Callable[[str], None]] = None
    timeout_s: Optional[int] = None
    auto_fallback: Optional[bool] = None
    on_pending_input: Optional[Callable[[], Optional[str]]] = None
    agent_name: Optional[str] = None


def llm_call(cliver: "Cliver", opts: LLMCallOptions) -> LLMCallResult:
    """Execute an LLM call with full CLI presentation.

    Handles spinner, streaming/sync dispatch, multimedia, token display, errors.
    This is the single entry point for all CLI commands that need LLM interaction.
    Routes through the default Agent (via AgentFactory) so agent config
    (model, auto_fallback) is applied consistently.

    Args:
        cliver: The Cliver CLI instance (provides console, thinking, token_tracker).
        opts: LLM call options.

    Returns:
        LLMCallResult with the response text and metadata.
    """
    agent = cliver.agent_factory.create(opts.agent_name)
    console = cliver.console
    thinking = cliver.thinking

    # Start spinner
    if thinking:
        thinking.start(opts.model or "")

    try:
        if opts.stream:
            result = _stream_call(agent, opts, thinking, console)
        else:
            result = _sync_call(agent, opts, thinking, console)
    except TaskTimeoutError:
        if thinking:
            thinking.stop()
        raise  # Let the caller (chat command) handle timeout
    except Exception as e:
        if thinking:
            thinking.stop()
        cliver.output(f"Error: {e}")
        return LLMCallResult(success=False, error=str(e))
    finally:
        if thinking:
            thinking.stop()

    # Notify caller of response text
    if result.success and result.text and opts.on_response:
        opts.on_response(result.text)

    # Display token usage
    _show_token_usage(cliver)

    return result


def _build_agent_kwargs(opts: LLMCallOptions) -> dict:
    """Build kwargs dict for Agent.run() / Agent.stream() from LLMCallOptions."""
    kwargs: dict = {}
    if opts.skill_name:
        kwargs["skill_name"] = opts.skill_name
    if opts.images:
        kwargs["images"] = opts.images
    if opts.audio_files:
        kwargs["audio_files"] = opts.audio_files
    if opts.video_files:
        kwargs["video_files"] = opts.video_files
    if opts.files:
        kwargs["files"] = opts.files
    if opts.model:
        kwargs["model"] = opts.model
    if opts.template:
        kwargs["template"] = opts.template
    if opts.params:
        kwargs["params"] = opts.params
    if opts.options:
        kwargs["options"] = opts.options
    if opts.tools_filter:
        kwargs["filter_tools"] = opts.tools_filter
    if opts.system_message_appender:
        kwargs["system_message_appender"] = opts.system_message_appender
    if opts.conversation_history is not None:
        kwargs["conversation_history"] = opts.conversation_history
    if opts.timeout_s is not None:
        kwargs["timeout_s"] = opts.timeout_s
    if opts.auto_fallback is not None:
        kwargs["auto_fallback"] = opts.auto_fallback
    if opts.on_pending_input:
        kwargs["on_pending_input"] = opts.on_pending_input
    return kwargs


def _stream_call(
    agent: "Agent",
    opts: LLMCallOptions,
    thinking,
    console,
) -> LLMCallResult:
    """Execute a streaming LLM call via Agent."""
    from cliver.media_handler import MultimediaResponseHandler

    response_handler = MultimediaResponseHandler(opts.media_dir)

    first_token_emitted = False

    def on_first_token():
        nonlocal first_token_emitted
        if thinking:
            thinking.stop()
        if not first_token_emitted:
            first_token_emitted = True
            console.print("─" * 50 + "")
            print(_response_color_start(), end="")

    try:
        accumulated_chunk = None

        async def _run():
            nonlocal accumulated_chunk, first_token_emitted
            kwargs = _build_agent_kwargs(opts)
            async for chunk in agent.stream(opts.user_input, **kwargs):
                if accumulated_chunk is None:
                    accumulated_chunk = chunk
                else:
                    try:
                        accumulated_chunk = accumulated_chunk + chunk
                    except Exception:
                        pass

                if hasattr(chunk, "content") and chunk.content:
                    on_first_token()
                    print(str(chunk.content), end="")
                elif first_token_emitted and not (hasattr(chunk, "content") and chunk.content):
                    # Between tool calls — LLM is thinking again. Reset so
                    # the next text chunk triggers the separator + color.
                    if thinking:
                        thinking.start(opts.model or "")
                    first_token_emitted = False
                    print("", flush=True)

            # Reset color and flush
            print(_response_color_reset(), flush=True)

        from cliver.util import run_async

        run_async(_run())

        text = ""
        if accumulated_chunk:
            llm_engine = agent._agent_core.get_llm_engine(opts.model)
            multimedia_response = response_handler.process_response(
                accumulated_chunk, llm_engine=llm_engine, auto_save_media=opts.save_media
            )

            if multimedia_response.has_media():
                console.print()
                media_count = len(multimedia_response.media_content)
                console.print(f"\n\\[Media Content: {media_count} items]")
                for i, media in enumerate(multimedia_response.media_content):
                    info = f"  {i + 1}. {media.type.value}"
                    if media.filename:
                        info += f" ({media.filename})"
                    if media.mime_type:
                        info += f" \\[{media.mime_type}]"
                    console.print(info)

            if hasattr(accumulated_chunk, "content") and accumulated_chunk.content:
                text = str(accumulated_chunk.content)

        console.print()  # trailing newline
        return LLMCallResult(success=True, text=text, response=accumulated_chunk)

    except ValueError as e:
        if "File upload is not supported" in str(e):
            console.print(f"Error: {e}")
            console.print("Will use content embedding as fallback.")
        else:
            raise
        return LLMCallResult(success=False, error=str(e))


def _sync_call(
    agent: "Agent",
    opts: LLMCallOptions,
    thinking,
    console,
) -> LLMCallResult:
    """Execute a synchronous (non-streaming) LLM call via Agent."""
    from cliver.media_handler import MultimediaResponseHandler

    response_handler = MultimediaResponseHandler(opts.media_dir)

    from cliver.util import run_async

    kwargs = _build_agent_kwargs(opts)
    response = run_async(agent.run(opts.user_input, **kwargs))

    # Stop spinner before printing response
    if thinking:
        thinking.stop()

    console.print("─" * 50 + "")

    text = ""
    if response:
        llm_engine = agent._agent_core.get_llm_engine(opts.model)
        multimedia_response = response_handler.process_response(
            response, llm_engine=llm_engine, auto_save_media=opts.save_media
        )

        if multimedia_response.has_text():
            text = multimedia_response.text_content
            console.print(f"{_response_color_start()}{text}{_response_color_reset()}")

        if multimedia_response.has_media():
            media_count = len(multimedia_response.media_content)
            console.print(f"\n\\[Media Content: {media_count} items]")
            for i, media in enumerate(multimedia_response.media_content):
                info = f"  {i + 1}. {media.type.value}"
                if media.filename:
                    info += f" ({media.filename})"
                if media.mime_type:
                    info += f" \\[{media.mime_type}]"
                console.print(info)

    return LLMCallResult(success=True, text=text, response=response)


def _show_token_usage(cliver: "Cliver") -> None:
    """Display token usage after a chat response using Rich formatting."""
    tracker = cliver.token_tracker
    if not tracker or not tracker.last_usage:
        return

    from cliver.cost_tracker import format_cost
    from cliver.token_tracker import format_tokens

    last = tracker.last_usage
    session = tracker.get_session_total()
    model = tracker.last_model or "?"

    cache_info = ""
    if last.cached_tokens > 0:
        cache_info = f" cached: {format_tokens(last.cached_tokens)}"

    # Cost estimation
    cost_info = ""
    cost_tracker = cliver.cost_tracker
    if cost_tracker is None:
        return

    estimate = cost_tracker.estimate_cost(model, last.input_tokens, last.output_tokens, last.cached_tokens)
    if estimate.total_cost > 0:
        session_cost = cost_tracker.get_session_total()
        cost_info = f"  cost: {format_cost(estimate.total_cost, estimate.currency)}"
        if session_cost > estimate.total_cost:
            cost_info += f" (session: {format_cost(session_cost, estimate.currency)})"

    cliver.output(
        f"◆ {model}  "
        f"tokens: {format_tokens(last.total_tokens)} "
        f"(in: {format_tokens(last.input_tokens)}, out: {format_tokens(last.output_tokens)})"
        f"{cache_info}{cost_info}  "
        f"session: {format_tokens(session.total_tokens)}"
    )
