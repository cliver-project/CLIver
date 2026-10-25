"""Application-side wiring for the CLI.

The core stays config-free: this module supplies CLIver's conventions — the
eight skill sources, the builtin toolset, the identity prompt, the model
endpoint from environment variables, and the interactive permission callback.
"""

from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import TextIO

from cliver_core import AgentCore, Decision, ModelBinding, ModelBindings, ToolCallContext
from cliver_core.events import InferenceEvent, InferenceEventType, ToolEvent, ToolEventType
from cliver_core.provider.providers import API_KEY_ENV
from cliver_core.tools import discover_builtin_tools
from dotenv import load_dotenv

from cliver.paths import get_config_dir
from cliver.skills import create_skill_manager

__all__ = [
    "build_agent",
    "llm_binding_from_env",
    "load_environment",
    "make_event_logger",
    "make_permission_callback",
]


def load_environment(config_dir: Path | None = None) -> None:
    """Load ``{config_dir}/.env`` without overriding the process environment."""
    config_path = get_config_dir() if config_dir is None else Path(config_dir).expanduser()
    env_file = config_path / ".env"
    if env_file.is_file():
        load_dotenv(env_file, override=False)


def llm_binding_from_env(
    env: Mapping[str, str] | None = None,
    *,
    model: str | None = None,
    provider: str | None = None,
    base_url: str | None = None,
    protocol: str | None = None,
    api_key: str | None = None,
) -> ModelBinding:
    """Build the chat binding from explicit values, then ``CLIVER_*`` variables.

    A known ``provider`` supplies its default ``base_url``; otherwise an
    explicit ``base_url`` is required.  Without an explicit key or
    ``CLIVER_API_KEY``, the key comes from the brand's conventional
    environment variable (``DEEPSEEK_API_KEY``, ``MINIMAX_API_KEY``, ...).
    """
    values = os.environ if env is None else env

    resolved_model = model or values.get("CLIVER_MODEL", "")
    if not resolved_model:
        raise ValueError("No chat model configured: pass --model or set CLIVER_MODEL.")

    resolved_provider = provider if provider is not None else values.get("CLIVER_PROVIDER", "")
    resolved_base_url = base_url if base_url is not None else values.get("CLIVER_BASE_URL", "")
    if not resolved_provider and not resolved_base_url:
        raise ValueError(
            "No endpoint configured: pass --provider or --base-url (or set CLIVER_PROVIDER / CLIVER_BASE_URL)."
        )

    resolved_protocol = protocol or values.get("CLIVER_PROTOCOL", "openai")
    resolved_api_key = api_key if api_key is not None else values.get("CLIVER_API_KEY", "")
    if not resolved_api_key:
        brand = (resolved_provider or "").strip().lower()
        resolved_api_key = next(
            (values[name] for name in API_KEY_ENV.get(brand, ()) if values.get(name)),
            "",
        )

    return ModelBinding.llm(
        model=resolved_model,
        api_key=resolved_api_key,
        base_url=resolved_base_url,
        protocol=resolved_protocol,
        provider_name=resolved_provider or None,
    )


def build_agent(
    binding: ModelBinding,
    *,
    cwd: Path | None = None,
    config_dir: Path | None = None,
    home: Path | None = None,
    permission_callback=None,
    on_event=None,
) -> AgentCore:
    """Build an ``AgentCore`` with CLIver's tools, skills, and identity."""
    working_dir = Path.cwd() if cwd is None else Path(cwd)
    return AgentCore(
        ModelBindings(llm=binding),
        builtin_tools=discover_builtin_tools(),
        skill_manager=create_skill_manager(config_dir=config_dir, cwd=working_dir, home=home),
        permission_callback=permission_callback,
        on_event=on_event,
        builtin_system_prompt=_system_prompt(working_dir),
    )


def make_permission_callback(*, assume_yes: bool = False):
    """Return the ASK-tool gate used by the one-shot CLI.

    ``--yes`` approves everything.  Otherwise an interactive terminal is
    prompted, and a non-interactive run denies with a hint instead of hanging.
    """

    async def decide(context: ToolCallContext) -> Decision:
        if assume_yes:
            return Decision.allow()
        if not sys.stdin.isatty():
            return Decision.deny("non-interactive session; rerun with --yes to approve tool calls")
        answer = await asyncio.to_thread(input, f"Allow {context.tool_name}? [y/N] ")
        if answer.strip().lower() in {"y", "yes"}:
            return Decision.allow()
        return Decision.deny("user denied the tool call")

    return decide


def make_event_logger(stream: TextIO | None = None):
    """Return an event handler that reports tool activity on stderr."""

    output = sys.stderr if stream is None else stream

    def write(line: str) -> None:
        print(line, file=output, flush=True)

    async def log(event: InferenceEvent | ToolEvent) -> None:
        if isinstance(event, ToolEvent):
            if event.event is ToolEventType.START:
                write(f"[tool] {event.tool_name} ...")
            elif event.event is ToolEventType.END:
                duration = f" ({event.duration_ms:.0f} ms)" if event.duration_ms is not None else ""
                write(f"[tool] {event.tool_name} done{duration}")
            elif event.event is ToolEventType.ERROR:
                write(f"[tool] {event.tool_name} failed: {event.error}")
            elif event.event is ToolEventType.PERMISSION_DENIED:
                write(f"[tool] {event.tool_name} denied: {event.error}")
        elif event.event is InferenceEventType.ERROR:
            error = (event.data or {}).get("error", "unknown error")
            write(f"[llm] error: {error}")

    return log


def _system_prompt(cwd: Path) -> str:
    return (
        "You are CLIver, a terminal AI assistant. Use the available tools to inspect the "
        "workspace and do work accurately; prefer reading files over guessing. When a task "
        "matches an available skill, activate it before proceeding. "
        f"The current working directory is {cwd}."
    )
