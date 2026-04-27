"""
Reusable LLM call component with CLI presentation.

Handles the full request/response lifecycle including:
- Thinking spinner (start/stop)
- Streaming or sync LLM calls
- Token usage display
- Error handling

This is a CLI-layer component — AgentCore and API layer have no dependency on it.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from langchain_core.messages import BaseMessage

from cliver.llm.errors import TaskTimeoutError

if TYPE_CHECKING:
    from cliver.cli import Cliver
    from cliver.llm import AgentCore

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
    tools_filter: Optional[Callable] = None
    system_message_appender: Optional[Callable] = None
    conversation_history: Optional[List[BaseMessage]] = None
    on_response: Optional[Callable[[str], None]] = None
    timeout_s: Optional[int] = None
    auto_fallback: Optional[bool] = None
    on_pending_input: Optional[Callable[[], Optional[str]]] = None


def llm_call(cliver: "Cliver", opts: LLMCallOptions) -> LLMCallResult:
    """Execute an LLM call with full CLI presentation.

    Handles spinner, streaming/sync dispatch, multimedia, token display, errors.
    This is the single entry point for all CLI commands that need LLM interaction.

    Args:
        cliver: The Cliver CLI instance (provides console, thinking, token_tracker).
        opts: LLM call options.

    Returns:
        LLMCallResult with the response text and metadata.
    """
    agent_core = cliver.agent_core
    console = cliver.console
    thinking = getattr(cliver, "thinking", None)

    # Start spinner
    if thinking:
        thinking.start(opts.model or "")

    try:
        if opts.stream:
            result = _stream_call(agent_core, opts, thinking, console)
        else:
            result = _sync_call(agent_core, opts, thinking, console)
    except TaskTimeoutError:
        if thinking:
            thinking.stop()
        raise  # Let the caller (chat command) handle timeout
    except Exception as e:
        if thinking:
            thinking.stop()
        cliver.output(f"[red]Error: {e}[/red]")
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


def _stream_call(
    agent_core: "AgentCore",
    opts: LLMCallOptions,
    thinking,
    console,
) -> LLMCallResult:
    """Execute a streaming LLM call."""
    from cliver.media_handler import MultimediaResponseHandler

    response_handler = MultimediaResponseHandler(opts.media_dir)

    first_token_emitted = False

    def on_first_token():
        nonlocal first_token_emitted
        if thinking:
            thinking.stop()
        if not first_token_emitted:
            first_token_emitted = True
            console.print("[dim]─" * 50 + "[/dim]")
            print(_response_color_start(), end="")

    try:
        accumulated_chunk = None

        async def _run():
            nonlocal accumulated_chunk
            async for chunk in agent_core.stream_user_input(
                user_input=opts.user_input,
                images=opts.images or None,
                audio_files=opts.audio_files or None,
                video_files=opts.video_files or None,
                files=opts.files or None,
                model=opts.model,
                template=opts.template,
                params=opts.params,
                options=opts.options,
                filter_tools=opts.tools_filter,
                system_message_appender=opts.system_message_appender,
                conversation_history=opts.conversation_history,
                timeout_s=opts.timeout_s,
                auto_fallback=opts.auto_fallback,
                on_pending_input=opts.on_pending_input,
            ):
                if accumulated_chunk is None:
                    accumulated_chunk = chunk
                else:
                    accumulated_chunk = accumulated_chunk + chunk

                if hasattr(chunk, "content") and chunk.content:
                    on_first_token()
                    print(str(chunk.content), end="")

            # Reset color and flush
            print(_response_color_reset(), flush=True)

        asyncio.run(_run())

        text = ""
        if accumulated_chunk:
            llm_engine = agent_core.get_llm_engine(opts.model)
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
            console.print(f"[red]Error: {e}[/red]")
            console.print("Will use content embedding as fallback.")
        else:
            raise
        return LLMCallResult(success=False, error=str(e))


def _sync_call(
    agent_core: "AgentCore",
    opts: LLMCallOptions,
    thinking,
    console,
) -> LLMCallResult:
    """Execute a synchronous (non-streaming) LLM call."""
    from cliver.media_handler import MultimediaResponseHandler

    response_handler = MultimediaResponseHandler(opts.media_dir)

    response = agent_core.process_user_input_sync(
        user_input=opts.user_input,
        images=opts.images or None,
        audio_files=opts.audio_files or None,
        video_files=opts.video_files or None,
        files=opts.files or None,
        model=opts.model,
        template=opts.template,
        params=opts.params,
        options=opts.options,
        filter_tools=opts.tools_filter,
        system_message_appender=opts.system_message_appender,
        conversation_history=opts.conversation_history,
        timeout_s=opts.timeout_s,
        auto_fallback=opts.auto_fallback,
        on_pending_input=opts.on_pending_input,
    )

    # Stop spinner before printing response
    if thinking:
        thinking.stop()

    console.print("[dim]─" * 50 + "[/dim]")

    text = ""
    if response:
        llm_engine = agent_core.get_llm_engine(opts.model)
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
    tracker = getattr(cliver, "token_tracker", None)
    if not tracker or not tracker.last_usage:
        return

    from cliver.cost_tracker import format_cost
    from cliver.token_tracker import format_tokens

    last = tracker.last_usage
    session = tracker.get_session_total()
    model = tracker.last_model or "?"

    cache_info = ""
    if last.cached_tokens > 0:
        cache_info = f" [dim green]cached: {format_tokens(last.cached_tokens)}[/dim green]"

    # Cost estimation
    cost_info = ""
    cost_tracker = getattr(cliver, "cost_tracker", None)
    if cost_tracker is None:
        return

    estimate = cost_tracker.estimate_cost(model, last.input_tokens, last.output_tokens, last.cached_tokens)
    if estimate.total_cost > 0:
        session_cost = cost_tracker.get_session_total()
        cost_info = f"  [dim]cost:[/dim] [bold]{format_cost(estimate.total_cost, estimate.currency)}[/bold]"
        if session_cost > estimate.total_cost:
            cost_info += f" [dim](session: {format_cost(session_cost, estimate.currency)})[/dim]"

    cliver.output(
        f"[dim]◆ {model}[/dim]  "
        f"[dim]tokens:[/dim] [bold]{format_tokens(last.total_tokens)}[/bold] "
        f"[dim](in: {format_tokens(last.input_tokens)}, out: {format_tokens(last.output_tokens)})[/dim]"
        f"{cache_info}{cost_info}  "
        f"[dim]session:[/dim] [bold]{format_tokens(session.total_tokens)}[/bold]"
    )
