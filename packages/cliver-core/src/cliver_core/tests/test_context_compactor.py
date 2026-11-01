"""Context-window seam: the application compacts history between iterations.

Core owns the Re-Act loop and the single system message; it ships no
tokenizer, threshold, summarization policy, or persistence.  A
``context_compactor`` is called before every model call with the run's
non-system history and may return a replacement history (or ``None`` to
leave it unchanged).  These tests pin the seam contract: where it is
invoked, what it sees, what a returned history must satisfy, and how it
interacts with usage accounting, failover, streaming, ``run_skill`` and
``chat_structured``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from cliver_core.agent import AgentCore
from cliver_core.bindings import ModelBinding, ModelBindings
from cliver_core.compaction import CompactionContext
from cliver_core.errors import ProviderError, RunTimeoutError
from cliver_core.messages import CLIverMessage, CLIverMessageChunk, ToolCall, UsageInfo
from cliver_core.provider import CLIverRequest, CLIverResponse, Provider
from cliver_core.skill import Skill
from cliver_core.tool import tool


class _ScriptedProvider(Provider):
    """Returns one scripted response per chat call and records each request."""

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


class _StreamingProvider(Provider):
    """Two streaming iterations: a tool turn, then a final answer."""

    protocol = "openai"
    brand = "mock-streaming"

    def __init__(self) -> None:
        super().__init__(api_key="", base_url="")
        self.requests: list[CLIverRequest] = []
        self.snapshots: list[list[CLIverMessage]] = []

    async def chat(self, request: CLIverRequest) -> CLIverResponse:  # pragma: no cover
        raise AssertionError("chat() is not used here")

    async def stream(self, request):
        self.requests.append(request)
        self.snapshots.append(list(request.messages))
        if len(self.requests) == 1:
            yield CLIverMessageChunk(content="checking")
            yield CLIverMessageChunk(
                native={
                    "role": "assistant",
                    "content": "checking",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "echo", "arguments": '{"value": "hi"}'},
                        }
                    ],
                },
                usage=UsageInfo(input_tokens=3, output_tokens=4),
            )
        else:
            yield CLIverMessageChunk(content="done")
            yield CLIverMessageChunk(
                native={"role": "assistant", "content": "done"},
                usage=UsageInfo(input_tokens=5, output_tokens=6),
            )


class _RetryableFailureProvider(Provider):
    protocol = "openai"
    brand = "mock-failing"

    def __init__(self) -> None:
        super().__init__(api_key="", base_url="")
        self.requests: list[CLIverRequest] = []

    async def chat(self, request: CLIverRequest) -> CLIverResponse:
        self.requests.append(request)
        raise ProviderError("temporary failure", retryable=True)

    async def stream(self, request):  # pragma: no cover
        raise AssertionError("stream() is not used here")


@tool(name="echo", description="Echoes the input back.")
def echo(value: str) -> str:
    return f"echo: {value}"


def _agent(provider: Provider, **kwargs) -> AgentCore:
    return AgentCore(
        ModelBindings(llm=ModelBinding.llm(model="mock-model", provider=provider)),
        builtin_tools=[echo],
        **kwargs,
    )


def _response(content: str | None, *, tool_calls: list[ToolCall] | None = None, usage=None) -> CLIverResponse:
    return CLIverResponse(
        message=CLIverMessage(role="assistant", content=content, tool_calls=tool_calls),
        usage=usage,
    )


# ── invocation and replacement ────────────────────────────────


@pytest.mark.asyncio
async def test_compactor_sees_history_without_the_system_message_and_replaces_it():
    usage_1 = UsageInfo(input_tokens=3, output_tokens=1)
    usage_2 = UsageInfo(input_tokens=4, output_tokens=2)
    provider = _ScriptedProvider(
        [
            _response(None, tool_calls=[ToolCall(id="call_1", name="echo", args={"value": "hi"})], usage=usage_1),
            _response("done", usage=usage_2),
        ]
    )
    seen: list[CompactionContext] = []

    def compactor(context: CompactionContext):
        seen.append(context)
        if context.iteration == 0:
            return None
        return [CLIverMessage(role="user", content="summary of earlier turns")]

    conversation = [
        CLIverMessage(role="user", content="earlier question"),
        CLIverMessage(role="assistant", content="earlier answer"),
    ]
    agent = _agent(provider)

    response = await agent.chat(
        "new question",
        system_prompt="sys",
        conversation=conversation,
        context_compactor=compactor,
        run_id="run-1",
    )

    assert response.message.content == "done"
    assert [context.iteration for context in seen] == [0, 1]
    assert all(context.run_id == "run-1" for context in seen)
    assert all(context.model == "mock-model" for context in seen)

    # Core-owned system message is never part of the exposed history.
    assert [message.role for message in seen[0].messages] == ["user", "assistant", "user"]
    assert [message.role for message in seen[1].messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
        "tool",
    ]

    # The replacement history is what the next model call receives; the
    # system message is re-attached by core.
    second = provider.requests[1].messages
    assert [message.role for message in second] == ["system", "user"]
    assert second[0].content == "sys"
    assert second[1].content == "summary of earlier turns"

    # The caller's conversation is never mutated.
    assert [message.content for message in conversation] == ["earlier question", "earlier answer"]


@pytest.mark.asyncio
async def test_async_compactor_is_awaited():
    provider = _ScriptedProvider([_response("ok")])
    calls: list[CompactionContext] = []

    async def compactor(context: CompactionContext):
        calls.append(context)
        return None

    agent = _agent(provider)
    response = await agent.chat("hi", context_compactor=compactor)

    assert response.message.content == "ok"
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_returning_none_leaves_the_history_unchanged():
    provider = _ScriptedProvider([_response("ok")])

    agent = _agent(provider)
    response = await agent.chat("hi", context_compactor=lambda context: None)

    assert response.message.content == "ok"
    assert [message.role for message in provider.requests[0].messages] == ["user"]


# ── per-call override ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_per_call_compactor_overrides_and_none_disables_the_constructor_default():
    provider = _ScriptedProvider([_response("one"), _response("two"), _response("three")])
    constructor_calls: list[int] = []
    per_call_calls: list[int] = []

    def constructor_compactor(context: CompactionContext):
        constructor_calls.append(context.iteration)
        return None

    def per_call_compactor(context: CompactionContext):
        per_call_calls.append(context.iteration)
        return None

    agent = _agent(provider, context_compactor=constructor_compactor)

    await agent.chat("hi")
    await agent.chat("hi", context_compactor=None)
    await agent.chat("hi", context_compactor=per_call_compactor)

    assert constructor_calls == [0]
    assert per_call_calls == [0]


def test_invalid_compactor_value_is_rejected():
    provider = _ScriptedProvider([])
    with pytest.raises(ValueError, match="context_compactor"):
        AgentCore(
            ModelBindings(llm=ModelBinding.llm(model="m", provider=provider)),
            context_compactor="not-callable",  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
async def test_invalid_per_call_compactor_is_rejected_before_any_model_call():
    provider = _ScriptedProvider([_response("unused")])
    agent = _agent(provider)

    with pytest.raises(ValueError, match="context_compactor"):
        await agent.chat("hi", context_compactor="not-callable")  # type: ignore[arg-type]

    assert provider.requests == []


# ── returned-history validation ───────────────────────────────


@pytest.mark.parametrize(
    ("returned", "match"),
    [
        ([], "empty"),
        ([CLIverMessage(role="system", content="replacement")], "system"),
        ([CLIverMessage(role="assistant", content="answer")], "end with"),
        ([CLIverMessage(role="tool", content="result", tool_call_id="missing")], "matching"),
        (
            [
                CLIverMessage(
                    role="assistant",
                    content=None,
                    tool_calls=[ToolCall(id="call_1", name="echo", args={})],
                )
            ],
            "tool result",
        ),
        (
            [
                CLIverMessage(
                    role="assistant",
                    content=None,
                    tool_calls=[ToolCall(id="call_1", name="echo", args={})],
                ),
                CLIverMessage(role="user", content="next"),
            ],
            "matching tool result",
        ),
    ],
)
@pytest.mark.asyncio
async def test_invalid_returned_history_raises_before_the_model_call(returned, match):
    provider = _ScriptedProvider([_response("unused")])
    agent = _agent(provider)

    with pytest.raises(ValueError, match=match):
        await agent.chat("hi", context_compactor=lambda context: returned)

    assert provider.requests == []


@pytest.mark.asyncio
async def test_compactor_exception_propagates():
    provider = _ScriptedProvider([_response("unused")])
    agent = _agent(provider)

    def boom(context: CompactionContext):
        raise RuntimeError("compactor exploded")

    with pytest.raises(RuntimeError, match="compactor exploded"):
        await agent.chat("hi", context_compactor=boom)

    assert provider.requests == []


# ── usage accounting ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_record_usage_folds_compaction_calls_into_usage_total_and_last_usage_is_visible():
    usage_1 = UsageInfo(input_tokens=5, output_tokens=1)
    usage_2 = UsageInfo(input_tokens=6, output_tokens=2)
    provider = _ScriptedProvider(
        [
            _response(None, tool_calls=[ToolCall(id="call_1", name="echo", args={"value": "hi"})], usage=usage_1),
            _response("done", usage=usage_2),
        ]
    )
    last_usages: list[UsageInfo | None] = []

    def compactor(context: CompactionContext):
        last_usages.append(context.last_usage)
        if context.iteration == 0:
            context.record_usage(UsageInfo(input_tokens=10, output_tokens=2))
        return None

    agent = _agent(provider)
    response = await agent.chat("hi", context_compactor=compactor)

    assert last_usages == [None, usage_1]
    assert response.usage == usage_2
    assert response.usage_total == UsageInfo(input_tokens=21, output_tokens=5)


# ── failover ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_failover_reuses_the_compacted_history_without_recompacting():
    primary = _RetryableFailureProvider()
    fallback = _ScriptedProvider([_response("from fallback")])
    calls: list[int] = []

    def compactor(context: CompactionContext):
        calls.append(context.iteration)
        return [CLIverMessage(role="user", content="compacted once")]

    agent = AgentCore(
        ModelBindings(
            llm=[
                ModelBinding.llm(model="primary", provider=primary),
                ModelBinding.llm(model="fallback", provider=fallback),
            ]
        ),
    )

    response = await agent.chat("hi", system_prompt="sys", context_compactor=compactor)

    assert response.message.content == "from fallback"
    assert calls == [0]  # once per iteration, not once per binding attempt
    assert [message.role for message in fallback.requests[0].messages] == ["system", "user"]
    assert fallback.requests[0].messages[1].content == "compacted once"


# ── streaming ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_streaming_compactor_replaces_history_and_records_usage_in_the_trace():
    provider = _StreamingProvider()
    seen: list[CompactionContext] = []

    async def compactor(context: CompactionContext):
        seen.append(context)
        if context.iteration == 0:
            context.record_usage(UsageInfo(input_tokens=10, output_tokens=2))
            return None
        return [CLIverMessage(role="user", content="summary")]

    agent = _agent(provider)
    chunks = [chunk async for chunk in agent.stream("hi", system_prompt="sys", context_compactor=compactor)]

    assert [chunk.content for chunk in chunks if chunk.content] == ["checking", "done"]
    assert [context.iteration for context in seen] == [0, 1]

    second = provider.snapshots[1]
    assert [message.role for message in second] == ["system", "user"]
    assert second[1].content == "summary"

    trace_chunk = next(chunk for chunk in chunks if "__llm_trace__" in chunk.vendor_ext)
    import json

    trace = json.loads(trace_chunk.vendor_ext["__llm_trace__"])
    assert trace[0]["usage_total"] == {
        "input_tokens": 13,
        "output_tokens": 6,
        "cache_read_input_tokens": None,
    }
    assert trace[1]["usage_total"] == {
        "input_tokens": 18,
        "output_tokens": 12,
        "cache_read_input_tokens": None,
    }


# ── run_skill / chat_structured forwarding ────────────────────


@pytest.mark.asyncio
async def test_run_skill_forwards_the_compactor(tmp_path: Path):
    provider = _ScriptedProvider([_response("skill done")])
    calls: list[int] = []

    def compactor(context: CompactionContext):
        calls.append(context.iteration)
        return None

    agent = _agent(provider)
    skill = Skill(name="demo", description="demo skill", body="Do the thing.", base_dir=tmp_path)

    response = await agent.run_skill(skill, "go", context_compactor=compactor)

    assert response.message.content == "skill done"
    assert calls == [0]


@pytest.mark.asyncio
async def test_chat_structured_forwards_the_compactor():
    provider = _ScriptedProvider([_response('{"part": "ok"}')])
    calls: list[int] = []

    def compactor(context: CompactionContext):
        calls.append(context.iteration)
        return None

    agent = _agent(provider)
    result = await agent.chat_structured(
        "hi",
        schema={"type": "object", "properties": {"part": {"type": "string"}}},
        context_compactor=compactor,
    )

    assert result.value == {"part": "ok"}
    assert calls == [0]


# ── run deadline ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_compaction_runs_inside_the_run_deadline():
    provider = _ScriptedProvider([_response("unused")])

    async def slow_compactor(context: CompactionContext):
        await asyncio.sleep(10)

    agent = _agent(provider)
    with pytest.raises(RunTimeoutError):
        await agent.chat("hi", context_compactor=slow_compactor, run_timeout=0.05)

    assert provider.requests == []
