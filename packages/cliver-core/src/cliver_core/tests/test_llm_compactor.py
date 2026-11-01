"""``LlmCompactor``: the opt-in reference policy built on ``CompactionPlan``.

Core never installs a compactor automatically.  These tests pin the
reference implementation's behavior: the threshold check is cheap and
local, the summarized prefix is exactly the dropped prefix, the
summarizer's usage reaches the run's meter, and an empty summary never
drops history.
"""

from __future__ import annotations

import pytest

from cliver_core.agent import AgentCore
from cliver_core.bindings import ModelBinding, ModelBindings
from cliver_core.compaction import CompactionContext
from cliver_core.compactors import LlmCompactor
from cliver_core.messages import CLIverMessage, ToolCall, UsageInfo
from cliver_core.provider import CLIverRequest, CLIverResponse, Provider
from cliver_core.tool import tool


class _FakeSummarizer:
    def __init__(self, content: str | None = "summary", usage: UsageInfo | None = None) -> None:
        self.content = content
        self.usage = usage
        self.prompts: list[str] = []

    async def chat(self, prompt: str) -> CLIverResponse:
        self.prompts.append(prompt)
        return CLIverResponse(
            message=CLIverMessage(role="assistant", content=self.content),
            usage=self.usage,
            usage_total=self.usage,
        )


def _context(
    messages: list[CLIverMessage],
    *,
    last_usage: UsageInfo | None = None,
    recorded: list[UsageInfo] | None = None,
) -> CompactionContext:
    recorded = [] if recorded is None else recorded
    return CompactionContext(
        messages=list(messages),
        iteration=1,
        run_id="run-1",
        model="mock-model",
        last_usage=last_usage,
        record_usage=recorded.append,
    )


def _turns(count: int) -> list[CLIverMessage]:
    messages: list[CLIverMessage] = []
    for index in range(1, count + 1):
        messages.append(CLIverMessage(role="user", content=f"q{index}"))
        messages.append(CLIverMessage(role="assistant", content=f"a{index}"))
    return messages


@pytest.mark.asyncio
async def test_below_threshold_returns_none_without_a_summarizer_call():
    summarizer = _FakeSummarizer()
    compactor = LlmCompactor(summarizer, threshold_tokens=100)
    context = _context(
        [*_turns(2), CLIverMessage(role="user", content="q3")],
        last_usage=UsageInfo(input_tokens=99, output_tokens=1),
    )

    assert await compactor(context) is None
    assert summarizer.prompts == []


@pytest.mark.asyncio
async def test_summarizes_older_turns_as_a_plan():
    usage = UsageInfo(input_tokens=7, output_tokens=3)
    summarizer = _FakeSummarizer("summary text", usage)
    compactor = LlmCompactor(summarizer, threshold_tokens=100, keep_turns=1)
    recorded: list[UsageInfo] = []
    context = _context(
        [*_turns(2), CLIverMessage(role="user", content="q3")],
        last_usage=UsageInfo(input_tokens=100, output_tokens=1),
        recorded=recorded,
    )

    plan = await compactor(context)

    assert plan is not None
    assert plan.keep_from == 4  # the last user turn is retained
    assert plan.summary == "summary text"
    assert plan.usage == usage
    assert recorded == []  # the plan path records usage in core

    prompt = summarizer.prompts[0]
    assert "user: q1" in prompt
    assert "assistant: a1" in prompt
    assert "user: q2" in prompt
    assert "assistant: a2" in prompt
    assert "q3" not in prompt


@pytest.mark.asyncio
async def test_no_older_turns_returns_none():
    summarizer = _FakeSummarizer()
    compactor = LlmCompactor(summarizer, threshold_tokens=1, keep_turns=1)
    context = _context(
        [CLIverMessage(role="user", content="q1")],
        last_usage=UsageInfo(input_tokens=10, output_tokens=1),
    )

    assert await compactor(context) is None
    assert summarizer.prompts == []


@pytest.mark.asyncio
@pytest.mark.parametrize("content", [None, ""])
async def test_empty_summary_leaves_history_unchanged_and_records_usage(content):
    usage = UsageInfo(input_tokens=7, output_tokens=3)
    summarizer = _FakeSummarizer(content, usage)
    compactor = LlmCompactor(summarizer, threshold_tokens=1, keep_turns=1)
    recorded: list[UsageInfo] = []
    context = _context(
        [*_turns(2), CLIverMessage(role="user", content="q3")],
        last_usage=UsageInfo(input_tokens=10, output_tokens=1),
        recorded=recorded,
    )

    assert await compactor(context) is None
    assert recorded == [usage]


def test_constructor_requires_a_summarizer_with_chat():
    with pytest.raises(ValueError, match="chat"):
        LlmCompactor(object(), threshold_tokens=1)  # type: ignore[arg-type]


@pytest.mark.parametrize("threshold", [0, -1, True, 1.5, "10"])
def test_invalid_threshold_is_rejected(threshold):
    with pytest.raises(ValueError, match="threshold_tokens"):
        LlmCompactor(_FakeSummarizer(), threshold_tokens=threshold)  # type: ignore[arg-type]


@pytest.mark.parametrize("keep_turns", [0, -1, True, 1.5, "2"])
def test_invalid_keep_turns_is_rejected(keep_turns):
    with pytest.raises(ValueError, match="keep_turns"):
        LlmCompactor(_FakeSummarizer(), threshold_tokens=1, keep_turns=keep_turns)  # type: ignore[arg-type]


class _ScriptedProvider(Provider):
    protocol = "openai"
    brand = "mock-scripted"

    def __init__(self, responses: list[CLIverResponse]) -> None:
        super().__init__(api_key="", base_url="")
        self.responses = list(responses)
        self.requests: list[CLIverRequest] = []

    async def chat(self, request: CLIverRequest) -> CLIverResponse:
        self.requests.append(request)
        return self.responses.pop(0)

    async def stream(self, request):  # pragma: no cover
        raise AssertionError("stream() is not used here")


@tool(name="echo", description="Echoes the input back.")
def echo(value: str) -> str:
    return f"echo: {value}"


def _response(content: str | None, *, tool_calls: list[ToolCall] | None = None, usage=None) -> CLIverResponse:
    return CLIverResponse(
        message=CLIverMessage(role="assistant", content=content, tool_calls=tool_calls),
        usage=usage,
    )


@pytest.mark.asyncio
async def test_agent_uses_llm_compactor_and_records_summary_usage():
    usage_1 = UsageInfo(input_tokens=3, output_tokens=1)
    usage_2 = UsageInfo(input_tokens=4, output_tokens=2)
    summary_usage = UsageInfo(input_tokens=10, output_tokens=2)
    provider = _ScriptedProvider(
        [
            _response(None, tool_calls=[ToolCall(id="call_1", name="echo", args={"value": "hi"})], usage=usage_1),
            _response("done", usage=usage_2),
        ]
    )
    summarizer = _FakeSummarizer("summary text", summary_usage)
    compactor = LlmCompactor(summarizer, threshold_tokens=1, keep_turns=1)
    agent = AgentCore(
        ModelBindings(llm=ModelBinding.llm(model="mock-model", provider=provider)),
        builtin_tools=[echo],
        context_compactor=compactor,
    )

    response = await agent.chat(
        "new question",
        system_prompt="sys",
        conversation=[
            CLIverMessage(role="user", content="q1"),
            CLIverMessage(role="assistant", content="a1"),
        ],
    )

    assert response.message.content == "done"

    second = provider.requests[1].messages
    assert [message.role for message in second] == ["system", "user", "user", "assistant", "tool"]
    assert second[0].content == "sys"
    assert second[1].content == "[Compacted earlier context]\nsummary text"
    assert second[2].content == "new question"
    assert response.usage_total == UsageInfo(input_tokens=17, output_tokens=5)
