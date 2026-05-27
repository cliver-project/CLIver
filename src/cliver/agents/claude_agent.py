"""ClaudeAgent — Claude Code CLI agent."""

from __future__ import annotations

import mimetypes
from typing import List, Optional

from cliver.agent import AgentResult, Artifact
from cliver.agents.cli_agent import CliAgent


class ClaudeAgent(CliAgent):
    DEFAULT_COMMAND = "claude"
    DEFAULT_ARGS = ["-p"]
    DEFAULT_OUTPUT_FORMAT = ["--output-format", "json"]

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

    def _get_stream_format(self) -> Optional[List[str]]:
        return ["--output-format", "stream-json"]
