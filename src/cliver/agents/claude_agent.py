"""ClaudeAgent — Claude Code CLI agent."""

from __future__ import annotations

import json
import mimetypes
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from cliver.agent import AgentResult, Artifact
from cliver.agents.cli_agent import CliAgent


@dataclass
class ClaudeStreamEvent:
    """A single JSONL event from Claude's --output-format stream-json."""

    type: str
    message: Optional[dict] = None
    subtype: Optional[str] = None
    session_id: Optional[str] = None
    # result event fields
    result: Optional[str] = None
    is_error: bool = False
    duration_ms: Optional[float] = None
    num_turns: Optional[int] = None
    # log event fields
    log: Optional[dict] = None
    # control request fields
    request_id: Optional[str] = None
    request: Optional[dict] = None


@dataclass
class ClaudeContentBlock:
    """A content block within a Claude assistant/user message."""

    type: str
    text: Optional[str] = None
    id: Optional[str] = None
    name: Optional[str] = None
    input: Optional[dict] = None
    tool_use_id: Optional[str] = None
    content: Optional[str] = None


@dataclass
class ClaudeMessageContent:
    """The parsed content of a Claude message."""

    role: str = ""
    model: str = ""
    content: List[ClaudeContentBlock] = field(default_factory=list)
    usage: Optional[dict] = None


class ClaudeAgent(CliAgent):
    DEFAULT_COMMAND = "claude"

    ENV_MAPPING: Dict[str, str] = {
        "api_key": "ANTHROPIC_AUTH_TOKEN",
        "api_url": "ANTHROPIC_BASE_URL",
    }

    BLOCKED_ARGS: Dict[str, bool] = {
        "-p": False,  # standalone: non-interactive mode
        "--output-format": True,  # takes value
        "--input-format": True,  # takes value
        "--permission-mode": True,  # takes value
        "--mcp-config": True,  # takes value
    }

    def _build_command(self, prompt: str) -> List[str]:
        return [
            self._resolved_command,
            "-p",
            "--output-format",
            "stream-json",
            "--permission-mode",
            "bypassPermissions",
            *self._user_args,
            prompt,
        ]

    def _build_env(self) -> dict:
        env = self._base_env()
        if self._model_config:
            env["ANTHROPIC_MODEL"] = self._model_config.api_model_name
            env["ANTHROPIC_DEFAULT_OPUS_MODEL"] = self._model_config.api_model_name
        return env

    def _parse_response(self, raw_json: dict) -> AgentResult:
        is_error = raw_json.get("is_error", False)
        return AgentResult(
            text=str(raw_json.get("result", "")),
            status="error" if is_error else "completed",
            model=raw_json.get("model"),
            token_usage=raw_json.get("usage"),
            error=str(raw_json.get("result", "")) if is_error else None,
        )

    def _extract_artifacts(self, raw_json: dict) -> List[Artifact]:
        artifacts = []
        for msg in raw_json.get("messages", []):
            if msg.get("role") != "assistant":
                continue
            for tc in msg.get("tool_calls", []):
                if tc.get("name") in ("Write", "Edit"):
                    path = tc.get("input", {}).get("file_path")
                    if path:
                        mime = mimetypes.guess_type(path)[0] or "text/plain"
                        artifacts.append(Artifact(path=path, media_type=mime))
        return artifacts

    def _parse_stream_chunk(self, chunk_json: dict) -> str:
        evt = ClaudeStreamEvent(**chunk_json)
        if evt.type == "assistant":
            return self._handle_assistant_event(evt)
        elif evt.type == "result":
            return evt.result or ""
        elif evt.type == "log":
            log = evt.log or {}
            level = log.get("level", "info")
            message = log.get("message", "")
            return f"[claude:{level}] {message}" if message else ""
        return ""

    def _handle_assistant_event(self, evt: ClaudeStreamEvent) -> str:
        if not evt.message:
            return ""
        try:
            content = ClaudeMessageContent(
                role=evt.message.get("role", ""),
                model=evt.message.get("model", ""),
                content=[ClaudeContentBlock(**b) for b in evt.message.get("content", [])],
                usage=evt.message.get("usage"),
            )
        except (TypeError, AttributeError):
            return ""

        parts: List[str] = []
        for block in content.content:
            if block.type == "text" and block.text:
                parts.append(block.text)
            elif block.type == "thinking" and block.text:
                parts.append(f"<thinking>{block.text}</thinking>")
            elif block.type == "tool_use":
                tool_name = block.name or "unknown"
                tool_input = json.dumps(block.input or {}, ensure_ascii=False)
                parts.append(f"<tool_call name='{tool_name}'>{tool_input}</tool_call>")
        return "".join(parts)
