"""Shared helpers for the gated live-verification suite.

Nothing in this package runs by default: every module is skipped unless
``CLIVER_LIVE_VERIFICATION`` is truthy (``1``, ``true``, ``yes``, ``on``).
With the gate on, a test whose provider key is missing warns and skips
instead of failing, naming the environment variable it needs.

Keys are read from the process environment.  ``~/.cliver/.env`` (or the file
named by ``CLIVER_LIVE_ENV_FILE``) is loaded as a convenience without
overriding variables that are already set.  Values are never printed.
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Any

import pytest

_TRUTHY = {"1", "true", "yes", "on"}
_ENV_FILE_VAR = "CLIVER_LIVE_ENV_FILE"
_DEFAULT_ENV_FILE = Path.home() / ".cliver" / ".env"

LIVE_OFF_REASON = "set CLIVER_LIVE_VERIFICATION=1 to run the live verification suite"


def live_enabled() -> bool:
    """Whether the live gate is on for this process."""
    return os.environ.get("CLIVER_LIVE_VERIFICATION", "").strip().lower() in _TRUTHY


def load_env_file() -> None:
    """Load the developer env file without overriding the process environment."""
    path = Path(os.environ.get(_ENV_FILE_VAR) or _DEFAULT_ENV_FILE).expanduser()
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        if "=" not in line:
            continue
        name, value = line.split("=", 1)
        os.environ.setdefault(name.strip(), value.strip().strip('"').strip("'"))


def skip_unless_keys(*names: str) -> None:
    """Warn and skip the calling test when a required variable is missing."""
    load_env_file()
    missing = [name for name in names if not os.environ.get(name, "").strip()]
    if not missing:
        return
    message = (
        "live verification skipped: set "
        + " or ".join(missing)
        + " (in the environment or ~/.cliver/.env) to run this check"
    )
    warnings.warn(message, UserWarning, stacklevel=2)
    pytest.skip(message)


def skip_with_warning(message: str) -> None:
    """Warn and skip with the same shape as :func:`skip_unless_keys`."""
    warnings.warn(message, UserWarning, stacklevel=2)
    pytest.skip(message)


class EventCollector:
    """Collect provider and tool events in arrival order."""

    def __init__(self) -> None:
        self.events: list[Any] = []

    def __call__(self, event: Any) -> None:
        self.events.append(event)

    def run_ids(self) -> set[str]:
        return {event.run_id for event in self.events if event.run_id is not None}

    def for_run(self, run_id: str) -> list[Any]:
        return [event for event in self.events if event.run_id == run_id]


def stream_text(chunks: list[Any]) -> str:
    """Concatenate the text deltas of a collected stream."""
    return "".join(chunk.content or "" for chunk in chunks)


def stream_trace(chunks: list[Any]) -> list[dict[str, Any]]:
    """The terminal diagnostic trace of a streamed run."""
    for chunk in chunks:
        payload = (chunk.vendor_ext or {}).get("__llm_trace__")
        if payload:
            return json.loads(payload)
    raise AssertionError("stream ended without a __llm_trace__ chunk")


def tool_names(trace_entry: dict[str, Any]) -> set[str]:
    """Names of the tools advertised in one diagnostic trace entry."""
    request = trace_entry.get("request") or {}
    tools = request.get("tools") or []
    return {tool.get("name") for tool in tools if tool.get("name")}


async def drain(stream) -> list[Any]:
    """Collect an async chunk stream into a list (run it in ``asyncio.gather``)."""
    return [chunk async for chunk in stream]
