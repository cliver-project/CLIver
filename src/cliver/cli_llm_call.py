"""
Reusable LLM call component with CLI presentation.

Handles streaming/sync LLM calls, spinner, token display, error handling.
Uses the new AgentCore (langchain-free).
"""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from cliver.messages import CLIverMessage

if TYPE_CHECKING:
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
    error: str | None = None


@dataclass
class LLMCallOptions:
    """Options for an LLM call."""

    user_input: str = ""
    model: str | None = None
    stream: bool = True
    system_prompt: str | None = None
    # Additional system content (skill instructions, IM context, etc.).
    # Merged after the builtin system prompt.
    options: dict[str, Any] | None = None
    conversation_history: list[CLIverMessage] | None = None
    mcp_servers: list[str] | None = None
    # Which MCP servers to use (None = all configured servers).
    on_response: Callable[[str], None] | None = None
    timeout_s: int | None = None


def llm_call(cliver: "Cliver", opts: LLMCallOptions) -> LLMCallResult:
    """Execute an LLM call with full CLI presentation."""
    console = cliver.console
    thinking = getattr(cliver, "thinking", None)
    model = opts.model

    if thinking:
        thinking.start(model or "")

    try:
        if opts.stream:
            result = _stream_call(cliver, opts, thinking, console)
        else:
            result = _sync_call(cliver, opts, thinking, console)
    except Exception as e:
        if thinking:
            thinking.stop()
        cliver.output(f"Error: {e}")
        return LLMCallResult(success=False, error=str(e))
    finally:
        if thinking:
            thinking.stop()

    if result.success and result.text and opts.on_response:
        opts.on_response(result.text)

    _show_token_usage(cliver)

    return result


def _merge_system_prompt(cliver, agent, extra: str | None) -> str | None:
    parts = []
    builtin_extra = cliver.build_system_prompt()
    if builtin_extra:
        parts.append(builtin_extra)
    if extra:
        parts.append(extra)
    return "\n\n".join(parts) or None


def _stream_call(cliver, opts, thinking, console) -> LLMCallResult:
    agent = cliver.get_agent_core(opts.model)
    system_prompt = _merge_system_prompt(cliver, agent, opts.system_prompt)

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
        text_parts: list[str] = []

        async def _run_stream():
            async for chunk in agent.stream(
                user_input=opts.user_input,
                system_prompt=system_prompt,
                conversation=opts.conversation_history,
                mcp_servers=opts.mcp_servers,
                options=opts.options,
            ):
                if chunk.content:
                    on_first_token()
                    print(chunk.content, end="")
                    text_parts.append(chunk.content)
                if chunk.vendor_ext.get("reasoning_content"):
                    pass

        import asyncio

        asyncio.run(_run_stream())

        if first_token_emitted:
            print(_response_color_reset(), flush=True)

        text = "".join(text_parts)
        console.print()
        return LLMCallResult(success=True, text=text)

    except Exception:
        raise


def _sync_call(cliver, opts, thinking, console) -> LLMCallResult:
    agent = cliver.get_agent_core(opts.model)
    system_prompt = _merge_system_prompt(cliver, agent, opts.system_prompt)

    import asyncio

    response = asyncio.run(
        agent.chat(
            user_input=opts.user_input,
            system_prompt=system_prompt,
            conversation=opts.conversation_history,
            mcp_servers=opts.mcp_servers,
            options=opts.options,
        )
    )

    if thinking:
        thinking.stop()

    console.print("─" * 50 + "")

    text = response.message.text or ""
    if text:
        console.print(f"{_response_color_start()}{text}{_response_color_reset()}")

    return LLMCallResult(success=True, text=text)


def _show_token_usage(cliver) -> None:
    """Display token usage after a chat response."""
    tracker = getattr(cliver, "token_tracker", None)
    if not tracker or not tracker.last_usage:
        return

    from cliver.token_tracker import format_tokens

    last = tracker.last_usage
    session = tracker.get_session_total()
    model = tracker.last_model or "?"

    cache_info = ""
    if last.cached_tokens > 0:
        cache_info = f" cached: {format_tokens(last.cached_tokens)}"

    cost_tracker = getattr(cliver, "cost_tracker", None)
    if cost_tracker is not None:
        from cliver.cost_tracker import format_cost

        estimate = cost_tracker.estimate_cost(model, last.input_tokens, last.output_tokens, last.cached_tokens)
        cost_info = ""
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
