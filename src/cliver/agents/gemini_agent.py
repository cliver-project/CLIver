"""GeminiAgent — Gemini CLI agent."""

from __future__ import annotations

import mimetypes
from typing import List, Optional

from cliver.agent import AgentResult, Artifact
from cliver.agents.cli_agent import CliAgent


class GeminiAgent(CliAgent):
    DEFAULT_COMMAND = "gemini"
    DEFAULT_ARGS = ["-p"]
    DEFAULT_OUTPUT_FORMAT = ["--output-format", "json"]

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

    def _get_stream_format(self) -> Optional[List[str]]:
        return ["--output-format", "stream-json"]
