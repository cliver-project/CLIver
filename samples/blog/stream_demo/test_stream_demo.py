"""The streaming sample runs offline against chunked scripted responses."""

from __future__ import annotations

from _shared.testing import ScriptedProvider, Streamed, build_scripted_agent
from stream_demo import stream_answer


async def test_stream_answer_prints_and_returns_each_delta(capsys) -> None:
    provider = ScriptedProvider([Streamed(["An agent loop ", "calls a model ", "and runs tools."])])
    agent = build_scripted_agent(provider)

    answer = await stream_answer(agent)

    captured = capsys.readouterr()
    assert answer == "An agent loop calls a model and runs tools."
    assert captured.out == answer + "\n"
