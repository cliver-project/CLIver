"""The read-only review agent, offline."""

from __future__ import annotations

import json
from pathlib import Path

from _shared.testing import ScriptedProvider, text_response, tool_call_response
from code_review import FIXTURE, Finding, ReviewReport, build_agent, review_repo

REPORT = {
    "summary": "Two loaders and one cache need attention.",
    "findings": [
        {
            "file": "aggregate.py",
            "line": 8,
            "severity": "medium",
            "message": "The cache default is shared by every call.",
        },
        {
            "file": "loaders.py",
            "line": 6,
            "severity": "high",
            "message": "The loop stops before the last path.",
        },
    ],
}


def _tool_messages(provider: ScriptedProvider) -> list[str]:
    return [message.content for message in provider.requests[-1].messages if message.role == "tool"]


async def test_the_agent_lists_reads_and_returns_a_validated_report() -> None:
    provider = ScriptedProvider(
        [
            tool_call_response("InspectRepo", {"path": ""}, call_id="call-1"),
            tool_call_response("InspectRepo", {"path": "loaders.py"}, call_id="call-2"),
            text_response(json.dumps(REPORT)),
        ]
    )
    agent = build_agent(provider=provider)

    result = await review_repo(agent)

    assert isinstance(result.value, ReviewReport)
    assert isinstance(result.value.findings[0], Finding)
    assert result.value.findings[0].severity == "medium"
    assert result.attempts == 1
    listing, source = _tool_messages(provider)
    assert "aggregate.py" in listing and "loaders.py" in listing
    assert "range(len(paths) - 1)" in source
    assert "6\t" in source  # line numbers are part of the read result


async def test_only_the_read_only_tool_is_advertised() -> None:
    provider = ScriptedProvider([text_response(json.dumps(REPORT))])
    agent = build_agent(provider=provider)

    await review_repo(agent)

    advertised = {tool.name for tool in provider.requests[0].tools or []}
    assert advertised == {"InspectRepo"}
    assert agent.tool_registry.tool_names == ["InspectRepo"]


async def test_a_read_outside_the_policy_is_refused(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "loaders.py").write_text("value = 1\n", encoding="utf-8")
    (tmp_path / "outside.py").write_text("SECRET = 'do not read me'\n", encoding="utf-8")
    provider = ScriptedProvider(
        [
            tool_call_response("InspectRepo", {"path": "../outside.py"}, call_id="call-1"),
            text_response(json.dumps({"summary": "blocked", "findings": []})),
        ]
    )
    agent = build_agent(provider=provider)

    await review_repo(agent, root)

    (tool_result,) = _tool_messages(provider)
    assert "outside the allowed read paths" in tool_result
    assert "do not read me" not in tool_result


async def test_the_fixture_keeps_the_seeded_issues() -> None:
    aggregate = (FIXTURE / "aggregate.py").read_text(encoding="utf-8")
    loaders = (FIXTURE / "loaders.py").read_text(encoding="utf-8")

    assert "cache={}" in aggregate
    assert "len(paths) - 1" in loaders
    assert "except:" in loaders
