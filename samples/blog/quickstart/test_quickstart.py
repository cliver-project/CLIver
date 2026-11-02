"""The quickstart sample runs offline against a scripted provider."""

from __future__ import annotations

from _shared.testing import ScriptedProvider, build_scripted_agent
from quickstart import PROMPT, run


async def test_run_returns_the_model_answer_and_advertises_tools() -> None:
    provider = ScriptedProvider(["main.py\nREADME.md"])
    agent = build_scripted_agent(provider)

    response = await run(agent)

    assert response.message.content == "main.py\nREADME.md"
    assert provider.requests[0].messages[-1].content == PROMPT
    assert provider.requests[0].tools


async def test_run_leaves_an_injected_agent_open() -> None:
    provider = ScriptedProvider(["first", "second"])
    agent = build_scripted_agent(provider)

    first = await run(agent)
    second = await run(agent)

    assert first.message.content == "first"
    assert second.message.content == "second"
