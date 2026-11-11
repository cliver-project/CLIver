"""run_skill() pins the skill body and narrows the tools it advertises."""

from __future__ import annotations

from _shared.testing import ScriptedProvider, tool_call_response
from skill_hello import build_agent, run


async def test_run_skill_pins_the_body_and_narrows_the_tools() -> None:
    provider = ScriptedProvider(
        [
            tool_call_response("Read", {"file_path": "references/phrases.md"}, call_id="call-1"),
            "こんにちは！",
        ]
    )

    response = await run(build_agent(provider=provider))

    system = provider.requests[0].messages[0].content
    assert "# Active skill: hello" in system
    assert "Skill location:" in system
    assert "Always greet in the language" in system
    assert [tool.name for tool in provider.requests[0].tools] == ["Read"]
    tool_messages = [m for m in provider.requests[-1].messages if m.role == "tool"]
    assert "こんにちは" in tool_messages[0].content
    assert response.message.content == "こんにちは！"


async def test_a_tool_outside_the_allowlist_is_rejected_before_it_runs() -> None:
    provider = ScriptedProvider(
        [
            tool_call_response("Write", {"file_path": "blocked.txt", "content": "nope"}, call_id="call-1"),
            "I cannot write files in this skill.",
        ]
    )

    await run(build_agent(provider=provider))

    assert [tool.name for tool in provider.requests[0].tools] == ["Read"]
    tool_messages = [m for m in provider.requests[-1].messages if m.role == "tool"]
    assert "Tool 'Write' is not available for this call" in tool_messages[0].content
