"""Test-only helpers shared by the blog samples.

Samples accept an injected ``AgentCore`` so their tests run offline: the
scripted provider returns prepared responses and records the requests it saw.
Nothing here is part of a published package or of any sample's demo path.
"""

from __future__ import annotations

import os
import warnings
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest
from cliver_core import AgentCore, CLIverMessage, CLIverMessageChunk, ModelBinding, ModelBindings
from cliver_core.messages import ToolCall
from cliver_core.provider import CLIverRequest, CLIverResponse, Provider
from cliver_core.tools import discover_builtin_tools

_TRUTHY = {"1", "true", "yes", "on"}
_DEFAULT_ENV_FILE = Path.home() / ".cliver" / ".env"


def _load_env_file() -> None:
    """Load the developer env file without overriding the process environment."""
    path = Path(os.environ.get("CLIVER_LIVE_ENV_FILE") or _DEFAULT_ENV_FILE).expanduser()
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


def skip_unless_live(*names: str) -> None:
    """Skip a live sample check unless the gate and its keys are available."""
    if os.environ.get("CLIVER_LIVE_VERIFICATION", "").strip().lower() not in _TRUTHY:
        pytest.skip("set CLIVER_LIVE_VERIFICATION=1 to run the sample live checks")
    _load_env_file()
    missing = [name for name in names if not os.environ.get(name, "").strip()]
    if not missing:
        return
    message = (
        "live sample check skipped: set " + " or ".join(missing) + " (in the environment or ~/.cliver/.env) to run it"
    )
    warnings.warn(message, UserWarning, stacklevel=2)
    pytest.skip(message)


@dataclass(frozen=True)
class Streamed:
    """A streamed answer: each item is yielded as one chunk."""

    chunks: Sequence[str]


Script = str | CLIverResponse | Exception | Streamed


class ScriptedProvider(Provider):
    """Return prepared responses in order and record every request."""

    protocol = "openai"
    brand = "scripted"

    def __init__(self, responses: Sequence[Script]) -> None:
        super().__init__(api_key="", base_url="")
        self._responses = list(responses)
        self.requests: list[CLIverRequest] = []

    def _next(self, request: CLIverRequest) -> Script:
        self.requests.append(request)
        if not self._responses:
            raise AssertionError("scripted provider ran out of responses")
        return self._responses.pop(0)

    async def chat(self, request: CLIverRequest) -> CLIverResponse:
        script = self._next(request)
        if isinstance(script, Exception):
            raise script
        if isinstance(script, Streamed):
            raise AssertionError("a Streamed script can only be used with stream()")
        if isinstance(script, str):
            return text_response(script)
        return script

    async def stream(self, request: CLIverRequest) -> AsyncIterator[CLIverMessageChunk]:
        script = self._next(request)
        if isinstance(script, Streamed):
            for content in script.chunks:
                yield CLIverMessageChunk(content=content)
            answer = "".join(script.chunks)
        elif isinstance(script, Exception):
            raise script
        else:
            answer = script if isinstance(script, str) else script.message.content or ""
            yield CLIverMessageChunk(content=answer)
        # AgentCore consumes the terminal native message to build the
        # assistant turn; without it the stream is considered truncated.
        yield CLIverMessageChunk(native={"role": "assistant", "content": answer})


def text_response(text: str) -> CLIverResponse:
    """A final assistant answer with no tool calls."""
    return CLIverResponse(message=CLIverMessage(role="assistant", content=text))


def tool_call_response(name: str, args: dict, *, call_id: str = "call-1") -> CLIverResponse:
    """An assistant turn that asks the agent to run one tool."""
    return CLIverResponse(
        message=CLIverMessage(
            role="assistant",
            content=None,
            tool_calls=[ToolCall(id=call_id, name=name, args=args)],
        )
    )


def build_scripted_agent(provider: Provider, *, tools: list | None = None) -> AgentCore:
    """Build an agent whose chat model is the scripted provider."""
    return AgentCore(
        ModelBindings(llm=ModelBinding.llm(model="scripted", provider=provider)),
        builtin_tools=discover_builtin_tools() if tools is None else tools,
    )
