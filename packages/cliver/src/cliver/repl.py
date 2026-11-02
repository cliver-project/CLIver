"""Interactive REPL for the ``cliver`` CLI.

The REPL is a thin session shell over the same ``AgentCore`` the one-shot
command uses: every turn streams the answer, the visible conversation is kept
between turns, and a failed or cancelled turn never ends the session.  Tool
activity continues to arrive on stderr through the CLI's event logger, and
ASK tools keep prompting through the shared permission callback.

Commands are intentionally small: ``/help``, ``/clear``, and ``/exit``.
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Callable, TextIO

from cliver_core import CLIverMessage, ProviderError

PROMPT = "cliver> "

HELP = """Commands:
  /help    show this help
  /clear   forget the conversation and start a new one
  /exit    leave the REPL (or just "exit"; Ctrl-D also works)"""


async def run_repl(
    agent,
    *,
    input_fn: Callable[[str], str] = input,
    output: TextIO | None = None,
    error: TextIO | None = None,
    max_iterations: int = 50,
    model: str = "",
    endpoint: str = "",
) -> int:
    """Run the interactive session until ``/exit``, ``/quit``, or EOF."""
    out = sys.stdout if output is None else output
    err = sys.stderr if error is None else error
    conversation: list[CLIverMessage] = []
    banner = "cliver REPL" + (f" ({model})" if model else "")
    _writeln(out, f"{banner} — /help for commands, /exit to quit.")

    while True:
        try:
            line = input_fn(PROMPT)
        except EOFError:
            _writeln(out, "")
            return 0
        except KeyboardInterrupt:
            _writeln(out, "")
            continue

        text = line.strip()
        if not text:
            continue
        if text.lower() in {"exit", "quit"}:
            return 0
        if text.startswith("/"):
            command = text.split()[0].lower()
            if command in {"/exit", "/quit"}:
                return 0
            if command == "/help":
                _writeln(out, HELP)
                continue
            if command == "/clear":
                conversation.clear()
                _writeln(out, "Conversation cleared.")
                continue
            _writeln(err, f"unknown command: {command} (try /help)")
            continue

        try:
            assistant = await _stream_turn(
                agent,
                text,
                conversation,
                max_iterations=max_iterations,
                output=out,
            )
        except (KeyboardInterrupt, asyncio.CancelledError):
            # Ctrl-C under asyncio.run cancels the session task; treat it as
            # cancelling this turn, not the REPL.
            _writeln(out, "\n[cancelled]")
            continue
        except Exception as exc:  # keep the session alive after a failed turn
            _writeln(err, f"error: {exc}")
            if isinstance(exc, ProviderError) and endpoint:
                _writeln(err, f"endpoint: {endpoint}")
            continue

        conversation.append(CLIverMessage(role="user", content=text))
        if assistant is not None:
            conversation.append(assistant)
        _writeln(out, "")


async def _stream_turn(
    agent,
    text: str,
    conversation: list[CLIverMessage],
    *,
    max_iterations: int,
    output: TextIO,
) -> CLIverMessage | None:
    """Stream one answer and return the assistant turn for the history.

    The visible text streams to *output*.  The history takes the final
    assistant message from the terminal trace, so narration from intermediate
    tool-calling iterations is displayed but not replayed into later turns.
    """
    stream = agent.stream(text, conversation=list(conversation), max_iterations=max_iterations)
    trace: list[dict] | None = None
    visible: list[str] = []
    try:
        async for chunk in stream:
            if chunk.content:
                visible.append(chunk.content)
                output.write(chunk.content)
                output.flush()
            payload = (chunk.vendor_ext or {}).get("__llm_trace__")
            if payload:
                trace = json.loads(payload)
    finally:
        try:
            await stream.aclose()
        except asyncio.CancelledError:
            # The generator's own finally already released the provider stream.
            pass

    final = trace[-1].get("message") if trace else None
    if final and not final.get("tool_calls"):
        return CLIverMessage.model_validate(final)
    visible_text = "".join(visible).strip()
    if visible_text:
        return CLIverMessage(role="assistant", content=visible_text)
    return None


def _writeln(stream: TextIO, text: str) -> None:
    stream.write(text + "\n")
    stream.flush()
