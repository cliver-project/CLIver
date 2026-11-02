"""The interactive REPL: multi-turn history, streaming, and commands."""

from __future__ import annotations

import io
import json

import pytest
from cliver.repl import run_repl
from cliver_core import CLIverMessageChunk, ProviderError


class FakeStreamAgent:
    """Async stream fake that records the conversation it was given."""

    def __init__(self, answers: list, *, prelude: str = "") -> None:
        self.answers = list(answers)
        self.prelude = prelude
        self.calls: list[tuple[str, list]] = []

    async def stream(self, prompt: str, **kwargs):
        self.calls.append((prompt, list(kwargs.get("conversation") or [])))
        answer = self.answers.pop(0) if self.answers else "answer"
        if isinstance(answer, Exception):
            raise answer
        if self.prelude:
            yield CLIverMessageChunk(content=self.prelude)
        half = len(answer) // 2
        yield CLIverMessageChunk(content=answer[:half])
        yield CLIverMessageChunk(content=answer[half:])
        yield CLIverMessageChunk(
            vendor_ext={"__llm_trace__": json.dumps([{"message": {"role": "assistant", "content": answer}}])}
        )


def scripted(*lines: str):
    """A readline function that returns *lines*, then raises EOF."""
    iterator = iter(lines)

    def read(_prompt: str) -> str:
        try:
            return next(iterator)
        except StopIteration:
            raise EOFError from None

    return read


async def test_repl_streams_answers_and_keeps_multi_turn_history():
    agent = FakeStreamAgent(["first answer", "second answer"])
    out, err = io.StringIO(), io.StringIO()

    code = await run_repl(agent, input_fn=scripted("hello", "again", "/exit"), output=out, error=err)

    assert code == 0
    assert "first answer" in out.getvalue()
    assert "second answer" in out.getvalue()
    assert err.getvalue() == ""
    second_conversation = agent.calls[1][1]
    assert [(message.role, message.content) for message in second_conversation] == [
        ("user", "hello"),
        ("assistant", "first answer"),
    ]


async def test_repl_history_uses_the_final_message_not_intermediate_narration():
    agent = FakeStreamAgent(["the answer"], prelude="Let me check. ")
    out = io.StringIO()

    await run_repl(agent, input_fn=scripted("go", "again", "/exit"), output=out, error=io.StringIO())

    assert "Let me check. " in out.getvalue()
    second_conversation = agent.calls[1][1]
    assert [(message.role, message.content) for message in second_conversation] == [
        ("user", "go"),
        ("assistant", "the answer"),
    ]


async def test_repl_help_clear_and_empty_lines_do_not_call_the_model():
    agent = FakeStreamAgent(["ok"])
    out, err = io.StringIO(), io.StringIO()

    code = await run_repl(
        agent,
        input_fn=scripted("", "/help", "/clear", "hi", "/exit"),
        output=out,
        error=err,
    )

    assert code == 0
    assert len(agent.calls) == 1
    assert "/clear" in out.getvalue()
    assert "/exit" in out.getvalue()
    assert "Conversation cleared." in out.getvalue()
    assert err.getvalue() == ""


async def test_repl_clear_resets_the_conversation():
    agent = FakeStreamAgent(["one", "two"])

    await run_repl(agent, input_fn=scripted("first", "/clear", "second", "/exit"), output=io.StringIO())

    assert agent.calls[0][1] == []
    assert agent.calls[1][1] == []


async def test_repl_survives_a_failed_turn_and_keeps_the_session_alive():
    agent = FakeStreamAgent([RuntimeError("boom"), "recovered"])
    out, err = io.StringIO(), io.StringIO()

    code = await run_repl(agent, input_fn=scripted("one", "two", "/exit"), output=out, error=err)

    assert code == 0
    assert "boom" in err.getvalue()
    assert "recovered" in out.getvalue()
    assert len(agent.calls) == 2
    # The failed turn is discarded: the next turn starts from empty history.
    assert agent.calls[1][1] == []


async def test_repl_unknown_command_is_reported_without_a_model_call():
    agent = FakeStreamAgent(["unused"])
    err = io.StringIO()

    await run_repl(agent, input_fn=scripted("/nope", "/exit"), output=io.StringIO(), error=err)

    assert agent.calls == []
    assert "unknown command: /nope" in err.getvalue()


async def test_repl_exits_on_eof():
    agent = FakeStreamAgent(["unused"])
    out = io.StringIO()

    code = await run_repl(agent, input_fn=scripted(), output=out)

    assert code == 0
    assert "cliver REPL" in out.getvalue()
    assert agent.calls == []


async def test_repl_ctrl_c_at_the_prompt_clears_the_line_and_continues():
    agent = FakeStreamAgent(["unused"])
    calls = 0

    def read(_prompt: str) -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise KeyboardInterrupt
        return "/exit"

    code = await run_repl(agent, input_fn=read, output=io.StringIO())

    assert code == 0
    assert agent.calls == []


@pytest.mark.parametrize("word", ["exit", "quit"])
async def test_repl_bare_exit_and_quit_leave_without_a_model_call(word):
    agent = FakeStreamAgent(["unused"])

    code = await run_repl(agent, input_fn=scripted(word), output=io.StringIO())

    assert code == 0
    assert agent.calls == []


async def test_repl_reports_the_endpoint_on_a_provider_error():
    agent = FakeStreamAgent([ProviderError("Error code: 404"), "recovered"])
    err = io.StringIO()

    await run_repl(
        agent,
        input_fn=scripted("first", "second", "/exit"),
        output=io.StringIO(),
        error=err,
        endpoint="https://api.deepseek.com/anthropic (openai protocol)",
    )

    assert "Error code: 404" in err.getvalue()
    assert "endpoint: https://api.deepseek.com/anthropic (openai protocol)" in err.getvalue()
    assert len(agent.calls) == 2
