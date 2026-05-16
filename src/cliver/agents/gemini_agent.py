"""GeminiAgent — Gemini CLI agent."""

from __future__ import annotations

import mimetypes
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from cliver.agent import AgentResult, Artifact
from cliver.agents.cli_agent import CliAgent


@dataclass
class GeminiStreamEvent:
    """A single JSONL event from Gemini's --output-format stream-json."""

    type: str
    timestamp: Optional[str] = None
    session_id: Optional[str] = None
    model: Optional[str] = None
    # message event fields
    role: Optional[str] = None
    content: str = ""
    delta: bool = False
    # tool_use event fields
    tool_name: Optional[str] = None
    tool_id: Optional[str] = None
    parameters: Optional[dict] = None
    # tool_result event fields
    status: Optional[str] = None
    output: Optional[str] = None
    # error event fields
    severity: Optional[str] = None
    message: Optional[str] = None
    # result event fields
    error: Optional[dict] = None
    stats: Optional[dict] = None
    # extra fields we don't model explicitly
    extra: dict = field(default_factory=dict)


class GeminiAgent(CliAgent):
    DEFAULT_COMMAND = "gemini"

    ENV_MAPPING: Dict[str, str] = {
        "api_key": "GEMINI_API_KEY",
    }

    BLOCKED_ARGS: Dict[str, bool] = {
        "-p": True,  # takes value: the prompt
        "--yolo": False,  # standalone
        "-o": True,  # takes value: output format
        "--output-format": True,  # takes value
    }

    def _build_command(self, prompt: str) -> List[str]:
        return [
            self._resolved_command,
            "-p",
            "-o",
            "stream-json",
            "--yolo",
            *self._user_args,
            prompt,
        ]

    def _build_env(self) -> dict:
        env = self._base_env()
        if self._model_config:
            env["GEMINI_MODEL"] = self._model_config.api_model_name
        return env

    def _parse_response(self, raw_json: dict) -> AgentResult:
        stats = raw_json.get("stats", {})
        error = raw_json.get("error")
        return AgentResult(
            text=str(raw_json.get("response", "")),
            status="error" if error else "completed",
            token_usage=stats.get("token_usage"),
            error=error.get("message") if isinstance(error, dict) else None,
        )

    def _extract_artifacts(self, raw_json: dict) -> List[Artifact]:
        artifacts = []
        stats = raw_json.get("stats", {})
        for f in stats.get("file_modifications", []):
            path = f.get("path", str(f)) if isinstance(f, dict) else str(f)
            mime = mimetypes.guess_type(path)[0] or "text/plain"
            artifacts.append(Artifact(path=path, media_type=mime))
        return artifacts

    def _parse_stream_chunk(self, chunk_json: dict) -> str:
        evt = GeminiStreamEvent(**{k: v for k, v in chunk_json.items() if k in GeminiStreamEvent.__dataclass_fields__})
        if evt.type == "message":
            return evt.content if evt.role == "assistant" else ""
        elif evt.type == "tool_use":
            tool_name = evt.tool_name or "unknown"
            return f"<tool_call name='{tool_name}'>{evt.parameters or {}}</tool_call>"
        elif evt.type == "tool_result":
            return f"<tool_result>{evt.output or ''}</tool_result>"
        elif evt.type == "error":
            return f"[gemini:{evt.severity or 'error'}] {evt.message or 'unknown error'}"
        elif evt.type == "result":
            if evt.error and isinstance(evt.error, dict):
                return f"[gemini:error] {evt.error.get('message', str(evt.error))}"
            return ""
        elif evt.type == "init":
            return ""
        return ""
