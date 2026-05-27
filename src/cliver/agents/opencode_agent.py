"""OpenCodeAgent — OpenCode CLI agent."""

from __future__ import annotations

from cliver.agent import AgentResult
from cliver.agents.cli_agent import CliAgent


class OpenCodeAgent(CliAgent):
    DEFAULT_COMMAND = "opencode"
    DEFAULT_ARGS = ["-p"]
    DEFAULT_OUTPUT_FORMAT = ["-f", "json"]

    def _parse_response(self, raw_json: dict) -> AgentResult:
        error = raw_json.get("error")
        return AgentResult(
            text=str(raw_json.get("response", raw_json.get("result", ""))),
            status="error" if error else "completed",
            error=error.get("message") if isinstance(error, dict) else None,
        )
