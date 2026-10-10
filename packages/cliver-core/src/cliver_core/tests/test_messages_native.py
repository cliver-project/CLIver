"""CLIverMessage native wire forms — construction, passthrough, rebuild (no network).

Covers:
- ``from_native`` for both protocols (fields derive from the native structure)
- ``to_native`` passthrough — deep copy, vendor extras and thinking
  signatures survive a second request; the cached native is never mutated
- ``to_native`` rebuild — protocol mismatch, ``native=None``, tool messages
- the stream terminal chunk (``native`` / ``usage`` fields)
- unknown-protocol ``ValueError`` on both constructors
- ``request_info`` / ``trace`` flow through one AgentCore chat turn
"""

from __future__ import annotations

import pytest

from cliver_core.messages import CLIverMessage, CLIverMessageChunk, ToolCall, UsageInfo
from cliver_core.provider.providers.openai import OpenAIProvider

# ── from_native (response side) ────────────────────────────────


def test_from_native_openai_derives_fields() -> None:
    native = {
        "role": "assistant",
        "content": None,
        "reasoning_content": "why",
        "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "echo", "arguments": '{"v": 1}'}}],
    }
    msg = CLIverMessage.from_native(native, "openai")

    assert msg.role == "assistant"
    assert msg.content is None  # tool_calls-only message
    assert msg.tool_calls[0] == ToolCall(id="call_1", name="echo", args={"v": 1})
    # pydantic copies on validation — the stored native equals but never
    # aliases the caller's dict
    assert msg.native == native and msg.native is not native
    assert msg.native_protocol == "openai"
    assert msg.native["reasoning_content"] == "why"  # vendor extras readable in native
    # passthrough round-trip: the native dict is returned as-is, so vendor
    # extras survive a re-send unchanged
    assert msg.to_native("openai")["reasoning_content"] == "why"


def test_from_native_anthropic_joins_blocks_and_reads_tool_results() -> None:
    assistant = {
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "hmm", "signature": "sig"},
            {"type": "text", "text": "line1"},
            {"type": "text", "text": "line2"},
            {"type": "tool_use", "id": "tu_1", "name": "echo", "input": {"v": 2}},
        ],
    }
    msg = CLIverMessage.from_native(assistant, "anthropic")
    assert msg.content == "line1\nline2"
    assert msg.tool_calls[0] == ToolCall(id="tu_1", name="echo", args={"v": 2})
    assert msg.native_protocol == "anthropic"

    tool_result = {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "tu_1", "content": [{"type": "text", "text": "done"}]}],
    }
    msg = CLIverMessage.from_native(tool_result, "anthropic")
    assert msg.role == "tool"
    assert msg.tool_call_id == "tu_1"
    assert msg.content == "done"


# ── to_native: passthrough (request side) ──────────────────────


def test_to_native_passthrough_preserves_thinking_signature() -> None:
    """A native message echoed back IS the wire form of the next request."""
    native = {
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "hmm", "signature": "sig"},
            {"type": "text", "text": "ok"},
            {"type": "tool_use", "id": "tu_1", "name": "echo", "input": {"v": 1}},
        ],
    }
    msg = CLIverMessage.from_native(native, "anthropic")

    assert msg.to_native("anthropic") == native

    # deep copy — mutating the request body leaves the cached native intact
    out = msg.to_native("anthropic")
    out["content"].append({"type": "text", "text": "extra"})
    assert msg.native["content"] == native["content"]


def test_to_native_rebuilds_on_protocol_mismatch() -> None:
    msg = CLIverMessage.from_native({"role": "assistant", "content": "hi"}, "anthropic")
    assert msg.to_native("openai") == {"role": "assistant", "content": "hi"}


# ── to_native: rebuild (agnostic request messages) ─────────────


def test_to_native_builds_plain_and_tool_messages() -> None:
    user = CLIverMessage(role="user", content="hi")
    assert user.to_native("openai") == {"role": "user", "content": "hi"}
    assert user.to_native("anthropic") == {"role": "user", "content": [{"type": "text", "text": "hi"}]}

    tool = CLIverMessage(role="tool", content="done", tool_call_id="call_1")
    openai_native = tool.to_native("openai")
    assert openai_native == {"role": "tool", "content": "done", "tool_call_id": "call_1"}
    anthropic_native = tool.to_native("anthropic")
    assert anthropic_native == {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "done"}],
    }


def test_to_native_builds_assistant_tool_calls() -> None:
    msg = CLIverMessage(
        role="assistant",
        content=None,
        tool_calls=[ToolCall(id="call_1", name="echo", args={"v": 1})],
    )
    openai_native = msg.to_native("openai")
    assert openai_native["tool_calls"] == [
        {"id": "call_1", "type": "function", "function": {"name": "echo", "arguments": '{"v": 1}'}}
    ]
    anthropic_native = msg.to_native("anthropic")
    assert anthropic_native["content"] == [{"type": "tool_use", "id": "call_1", "name": "echo", "input": {"v": 1}}]


def test_unknown_protocol_raises_on_both_constructors() -> None:
    with pytest.raises(ValueError, match="Unknown protocol 'mqtt'"):
        CLIverMessage.from_native({"role": "user", "content": "hi"}, "mqtt")
    with pytest.raises(ValueError, match="Unknown protocol 'mqtt'"):
        CLIverMessage(role="user", content="hi").to_native("mqtt")


# ── stream terminal chunk ──────────────────────────────────────


def test_chunk_terminal_fields_carry_native_and_usage() -> None:
    delta = CLIverMessageChunk(content="hi")
    assert delta.native is None and delta.usage is None  # ordinary delta chunk

    terminal = CLIverMessageChunk(
        native={"role": "assistant", "content": "hi"},
        usage=UsageInfo(input_tokens=1, output_tokens=2),
    )
    assert terminal.model_dump()["usage"] == {"input_tokens": 1, "output_tokens": 2}
    assert terminal.model_dump()["native"] == {"role": "assistant", "content": "hi"}


# ── request_info / trace through one chat turn ─────────────────


class _StubTransport:
    """Transport stub: records wire bodies, replays one canned completion."""

    label = "stub"

    def __init__(self, message: dict) -> None:
        self.message = message
        self.calls: list[dict] = []

    async def chat(self, body: dict) -> dict:
        self.calls.append(body)
        return {"choices": [{"message": dict(self.message)}]}


class _TestOpenAIProvider(OpenAIProvider):
    brand = "test-openai"


@pytest.mark.asyncio
async def test_request_info_and_trace_through_one_chat_turn() -> None:
    from cliver_core.agent import AgentCore
    from cliver_core.bindings import ModelBinding, ModelBindings

    transport = _StubTransport({"role": "assistant", "content": "ok", "reasoning_content": "why"})
    provider = _TestOpenAIProvider(api_key="k", base_url="https://example.test/v1", transport=transport)
    agent = AgentCore(
        ModelBindings(llm=ModelBinding.llm(model="m", provider=provider)),
        builtin_tools=[],
    )

    response = await agent.chat("hi")
    msg = response.message

    # the final message stayed a thin wrapper — native readable on it
    assert msg.content == "ok"
    assert msg.native["reasoning_content"] == "why"

    # one Re-Act iteration recorded: request echo + native response + fields
    (entry,) = msg.trace
    assert entry["request"]["model"] == "m"
    assert entry["request"]["provider"] == "test-openai"
    last = entry["request"]["messages"][-1]
    assert (last["role"], last["content"]) == ("user", "hi")
    assert entry["response"]["reasoning_content"] == "why"
    assert entry["message"]["content"] == "ok"
    assert "native" not in entry["message"]

    # per-message conversion happened on the message, not the provider
    assert transport.calls[0]["messages"][-1] == {"role": "user", "content": "hi"}
