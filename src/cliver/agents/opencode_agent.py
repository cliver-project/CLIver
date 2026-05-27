"""OpenCodeAgent — OpenCode CLI agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from cliver.agent import AgentResult
from cliver.agents.cli_agent import CliAgent


@dataclass
class OpenCodeTokenUsage:
    """Token usage from a step_finish event."""

    input: int = 0
    output: int = 0
    cache: Optional[dict] = None


@dataclass
class OpenCodeToolState:
    """Tool invocation state within a tool_use event."""

    status: str = ""
    input: Optional[dict] = None
    output: Optional[str] = None


@dataclass
class OpenCodeEventPart:
    """The 'part' field within an OpenCode stream event."""

    id: str = ""
    message_id: str = ""
    session_id: str = ""
    type: str = ""
    text: str = ""
    tool: str = ""
    call_id: str = ""
    state: Optional[OpenCodeToolState] = None
    tokens: Optional[OpenCodeTokenUsage] = None


@dataclass
class OpenCodeErrorInfo:
    """Error information from an OpenCode error event."""

    name: str = ""
    data: Optional[dict] = None


@dataclass
class OpenCodeStreamEvent:
    """A single JSON event from `opencode run --format json`."""

    type: str
    timestamp: int = 0
    session_id: str = ""
    part: OpenCodeEventPart = field(default_factory=OpenCodeEventPart)
    error: Optional[OpenCodeErrorInfo] = None


class OpenCodeAgent(CliAgent):
    DEFAULT_COMMAND = "opencode"

    ENV_MAPPING: Dict[str, str] = {
        "api_key": "OPENAI_API_KEY",
        "api_url": "OPENAI_BASE_URL",
    }

    BLOCKED_ARGS: Dict[str, bool] = {
        "-f": True,  # takes value: format
        "--format": True,  # takes value
    }

    def _build_command(self, prompt: str) -> List[str]:
        return [
            self._resolved_command,
            "-p",
            "-f",
            "json",
            *self._user_args,
            prompt,
        ]

    def _build_env(self) -> dict:
        env = self._base_env()
        if self._model_config:
            env["OPENAI_MODEL"] = self._model_config.api_model_name
        env["OPENCODE_PERMISSION"] = '{"*":"allow"}'
        return env

    def _parse_response(self, raw_json: dict) -> AgentResult:
        error = raw_json.get("error")
        return AgentResult(
            text=str(raw_json.get("response", raw_json.get("result", ""))),
            status="error" if error else "completed",
            error=error.get("message") if isinstance(error, dict) else None,
        )

    def _parse_stream_chunk(self, chunk_json: dict) -> str:
        part_dict = chunk_json.get("part", {})
        state_dict = part_dict.get("state")
        tokens_dict = part_dict.get("tokens")
        error_dict = chunk_json.get("error")

        evt = OpenCodeStreamEvent(
            type=chunk_json.get("type", ""),
            timestamp=chunk_json.get("timestamp", 0),
            session_id=chunk_json.get("sessionID", chunk_json.get("session_id", "")),
            part=OpenCodeEventPart(
                id=part_dict.get("id", ""),
                message_id=part_dict.get("messageID", part_dict.get("message_id", "")),
                session_id=part_dict.get("sessionID", part_dict.get("session_id", "")),
                type=part_dict.get("type", ""),
                text=part_dict.get("text", ""),
                tool=part_dict.get("tool", ""),
                call_id=part_dict.get("callID", part_dict.get("call_id", "")),
                state=OpenCodeToolState(**state_dict) if isinstance(state_dict, dict) else None,
                tokens=OpenCodeTokenUsage(**tokens_dict) if isinstance(tokens_dict, dict) else None,
            ),
            error=OpenCodeErrorInfo(**error_dict) if isinstance(error_dict, dict) else None,
        )

        if evt.type == "text":
            return evt.part.text
        elif evt.type == "tool_use":
            tool_name = evt.part.tool or "unknown"
            tool_input = evt.part.state.input if evt.part.state and evt.part.state.input else {}
            return f"<tool_call name='{tool_name}'>{tool_input}</tool_call>"
        elif evt.type == "error":
            if evt.error and evt.error.data:
                msg = evt.error.data.get("message", evt.error.name)
            elif evt.error:
                msg = evt.error.name
            else:
                msg = "unknown opencode error"
            return f"[opencode:error] {msg}"
        elif evt.type in ("step_start", "step_finish"):
            return ""
        return ""
