"""Offline tests: a denial reaches the model, and usage totals span the run."""

from __future__ import annotations

from pathlib import Path

from _shared.testing import ScriptedProvider, text_response, tool_call_response
from agent import build_agent
from cliver_core import UsageInfo
from cliver_core.events import ToolEvent, ToolEventType


# region: denial
async def test_a_denied_tool_call_reaches_the_model_as_a_tool_result(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    provider = ScriptedProvider(
        [
            tool_call_response("SaveNote", {"title": "plan", "body": "Ship it."}),
            "The note was not saved.",
        ]
    )
    events: list[ToolEvent] = []
    agent = build_agent(provider, on_event=events.append)

    response = await agent.chat("Save a note titled plan.")

    assert response.message.content == "The note was not saved."
    assert not (tmp_path / "notes").exists()
    tool_message = [message for message in provider.requests[1].messages if message.role == "tool"][-1]
    assert tool_message.content == "Error: Permission denied: notes are disabled in this test"
    assert [event.event for event in events] == [
        ToolEventType.START,
        ToolEventType.PERMISSION_DENIED,
        ToolEventType.END,
    ]


# endregion: denial


# region: usage
async def test_usage_total_sums_every_model_call(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    provider = ScriptedProvider(
        [
            tool_call_response(
                "SaveNote",
                {"title": "plan", "body": "Ship it."},
                usage=UsageInfo(input_tokens=10, output_tokens=2),
            ),
            text_response("The note was not saved.", usage=UsageInfo(input_tokens=20, output_tokens=5)),
        ]
    )
    agent = build_agent(provider)

    response = await agent.chat("Save a note titled plan.")

    assert response.usage == UsageInfo(input_tokens=20, output_tokens=5)
    assert response.usage_total == UsageInfo(input_tokens=30, output_tokens=7)


# endregion: usage
