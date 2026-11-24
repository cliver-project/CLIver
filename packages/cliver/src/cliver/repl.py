"""Interactive REPL for the ``cliver`` CLI.

The REPL is a thin session shell over the same ``AgentCore`` the one-shot
command uses: every turn streams the answer, the visible conversation is kept
between turns, and a failed or cancelled turn never ends the session.  Tool
activity continues to arrive on stderr through the CLI's event logger, and
ASK tools keep prompting through the shared permission callback.

Commands are intentionally small: ``/help``, ``/clear``, ``/exit``, and the
media commands (``/image``, ``/video``, ``/audio``/``/tts``, ``/asr``).
"""

from __future__ import annotations

import asyncio
import json
import shlex
import sys
from pathlib import Path
from typing import Callable, TextIO

from cliver_core import CLIverMessage, ProviderError

from cliver import media

PROMPT = "cliver> "

HELP = """Commands:
  /help             show this help
  /clear            forget the conversation and start a new one
  /image <prompt>   generate an image into the current directory
  /video <prompt>   generate a video into the current directory
  /audio <text>     synthesize speech (/tts is an alias)
  /asr <file|url> [--language zh]   transcribe audio and print the text
  /exit             leave the REPL (or just "exit"; Ctrl-D also works)"""

_MEDIA_COMMANDS = {f"/{name}" for name in media.MEDIA_COMMANDS}


async def run_repl(
    agent,
    *,
    input_fn: Callable[[str], str] = input,
    output: TextIO | None = None,
    error: TextIO | None = None,
    max_iterations: int = 50,
    model: str = "",
    endpoint: str = "",
    config_hint: str = "",
) -> int:
    """Run the interactive session until ``/exit``, ``/quit``, or EOF.

    *config_hint* names the JSON config file in "no media model configured"
    messages; it is the file the CLI actually read.
    """
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
            parts = text.split(maxsplit=1)
            command = parts[0].lower()
            rest = parts[1].strip() if len(parts) > 1 else ""
            if command in {"/exit", "/quit"}:
                return 0
            if command == "/help":
                _writeln(out, HELP)
                continue
            if command == "/clear":
                conversation.clear()
                _writeln(out, "Conversation cleared.")
                continue
            if command in _MEDIA_COMMANDS:
                await _media_turn(agent, command[1:], rest, out, err, config_hint)
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


async def _media_turn(agent, command: str, rest: str, out: TextIO, err: TextIO, config_hint: str) -> None:
    """Run one media command without touching the conversation history."""
    if command == "asr":
        try:
            source, language = _parse_asr_args(rest)
        except ValueError as exc:
            _writeln(err, f"asr: {exc}")
            return
        if not source:
            _writeln(err, "asr needs an audio file or URL")
            return
    else:
        source, language = rest, None
        if not source:
            _writeln(err, f"{command} needs a prompt")
            return

    purpose = media.bound_purpose(getattr(agent, "bindings", None), command)
    if purpose is None:
        _writeln(err, media.not_configured_message(command, config_hint))
        return

    _writeln(err, media.status_line(command, source))
    try:
        response = await media.run_in_agent(agent, command, purpose, source, language=language)
        failure = media.failure_message(response)
        if failure is not None:
            _writeln(err, f"error: {failure}")
            return
        if command == "asr":
            content = (response.message.content or "").strip()
            if content:
                _writeln(out, content)
            return
        saved = media.save_media(response, command, Path.cwd())
        if not saved:
            detail = (response.message.content or "the provider returned no media").strip()
            _writeln(err, f"error: {detail}")
            return
        for path in saved:
            _writeln(out, media.display_path(path))
    except (KeyboardInterrupt, asyncio.CancelledError):
        _writeln(out, "\n[cancelled]")
    except Exception as exc:  # media failures never end the session
        _writeln(err, f"error: {exc}")


def _parse_asr_args(rest: str) -> tuple[str, str | None]:
    """Split ``/asr SOURCE [--language LANG]`` into the source and the hint."""
    tokens = shlex.split(rest)
    language: str | None = None
    source: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == "--language" and index + 1 < len(tokens):
            language = tokens[index + 1]
            index += 2
            continue
        if token.startswith("--language="):
            language = token.split("=", 1)[1]
            index += 1
            continue
        source.append(token)
        index += 1
    return " ".join(source), language


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
