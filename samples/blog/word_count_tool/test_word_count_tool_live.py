"""Live check — needs CLIVER_LIVE_VERIFICATION=1 and DEEPSEEK_API_KEY."""

from __future__ import annotations

from _shared.testing import skip_unless_live
from cliver_core import AgentCore, ModelBindings, presets
from word_count_tool import ask_word_count, word_count


async def test_word_count_tool_live() -> None:
    skip_unless_live("DEEPSEEK_API_KEY")

    events = []
    agent = AgentCore(
        ModelBindings(llm=presets.deepseek()),
        builtin_tools=[word_count],
        on_event=events.append,
    )
    try:
        response = await ask_word_count(agent)
    finally:
        await agent.aclose()

    assert response.message.content
    assert any(getattr(event, "tool_name", None) == "word_count" for event in events)
