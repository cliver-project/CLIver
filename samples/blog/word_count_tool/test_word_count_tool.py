"""The custom-tool sample runs offline and exercises the real tool loop."""

from __future__ import annotations

from _shared.testing import ScriptedProvider, build_scripted_agent, tool_call_response
from word_count_tool import ask_word_count, word_count


async def test_the_model_calls_the_tool_and_reads_its_result() -> None:
    provider = ScriptedProvider(
        [
            tool_call_response("word_count", {"text": "Agents are loops with tools"}),
            "The sentence has 5 words.",
        ]
    )
    agent = build_scripted_agent(provider, tools=[word_count])

    response = await ask_word_count(agent)

    assert response.message.content == "The sentence has 5 words."
    advertised = [registered.name for registered in provider.requests[0].tools]
    assert advertised == ["word_count"]
    tool_messages = [message for message in provider.requests[1].messages if message.role == "tool"]
    assert tool_messages[-1].content == "5"


def test_word_count_counts_whitespace_separated_words() -> None:
    assert word_count.execute(text="one two  three")[0]["text"] == "3"
