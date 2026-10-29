"""Consumer-facing inference contracts: metered events and error boundaries.

These pin the behaviour OnePilot relies on: provider ``inference.completed``
events carry the call's usage, handler failures abort the call, and the
Re-Act run's aggregate usage is available on ``CLIverResponse.usage_total``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cliver_core.agent import AgentCore
from cliver_core.bindings import ModelBinding, ModelBindings
from cliver_core.events import InferenceEvent, InferenceEventType
from cliver_core.media import MediaContent, MediaType
from cliver_core.messages import CLIverMessage, UsageInfo
from cliver_core.provider import CLIverRequest
from cliver_core.provider.providers.anthropic import AnthropicProvider
from cliver_core.provider.providers.openai import OpenAIProvider
from cliver_core.skill import SkillManager
from cliver_core.tool import CLIverTool


class _FakeTransport:
    """Pure transport pipe: records bodies, replays canned payloads or raises."""

    label = "fake"

    def __init__(
        self,
        *,
        chat_payload: dict | None = None,
        chat_payloads: list[dict] | None = None,
        stream_payloads: list[dict] | None = None,
        chat_error: Exception | None = None,
        stream_error: Exception | None = None,
    ) -> None:
        self.chat_payload = chat_payload
        self.chat_payloads = chat_payloads
        self.stream_payloads = stream_payloads or []
        self.chat_error = chat_error
        self.stream_error = stream_error
        self.chat_calls: list[dict] = []
        self.stream_calls: list[dict] = []

    async def chat(self, body: dict) -> dict:
        self.chat_calls.append(body)
        if self.chat_error is not None:
            raise self.chat_error
        if self.chat_payloads is not None:
            return self.chat_payloads[min(len(self.chat_calls) - 1, len(self.chat_payloads) - 1)]
        return self.chat_payload or {}

    async def stream(self, body: dict):
        self.stream_calls.append(body)
        if self.stream_error is not None:
            raise self.stream_error
        for payload in self.stream_payloads:
            yield payload

    async def post(self, path: str, body: dict, *, files=None, binary=False, headers=None):
        raise AssertionError("post() is not used here")

    async def close(self) -> None:
        return


def _request(text: str = "hi") -> CLIverRequest:
    return CLIverRequest(messages=[CLIverMessage(role="user", content=text)], model="m")


def _echo_tool() -> CLIverTool:
    return CLIverTool(
        name="echo",
        description="Echo the input.",
        parameters={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        },
        execute=lambda value: [{"text": f"echo: {value}"}],
        safety="safe",
    )


def _openai_final(content: str = "done", prompt: int = 3, completion: int = 2) -> dict:
    return {
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": prompt, "completion_tokens": completion},
    }


def _openai_tool_call(prompt: int = 10, completion: int = 2) -> dict:
    return {
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "echo", "arguments": '{"value": "hi"}'},
                        }
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": prompt, "completion_tokens": completion},
    }


def _openai_stream(final_usage: dict | None, text: str = "done") -> list[dict]:
    return [
        {"choices": [{"index": 0, "delta": {"content": text}}]},
        {
            "__cliver_final__": {
                "message": {"role": "assistant", "content": text},
                "usage": final_usage,
            }
        },
    ]


def _anthropic_final(content: str = "done", prompt: int = 5, completion: int = 2) -> dict:
    return {
        "role": "assistant",
        "content": [{"type": "text", "text": content}],
        "usage": {"input_tokens": prompt, "output_tokens": completion},
    }


def _anthropic_stream(final_usage: dict | None, text: str = "done") -> list[dict]:
    return [
        {"type": "message_start", "message": {"usage": {"input_tokens": 1}}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}},
        {"type": "message_stop"},
        {
            "__cliver_final__": {
                "message": {"role": "assistant", "content": [{"type": "text", "text": text}]},
                "usage": final_usage,
            }
        },
    ]


async def _collect_events(provider: OpenAIProvider | AnthropicProvider) -> list[InferenceEvent]:
    events: list[InferenceEvent] = []

    async def handler(event: InferenceEvent) -> None:
        events.append(event)

    provider.on_event = handler
    return events


def _completed(events: list[InferenceEvent]) -> list[InferenceEvent]:
    return [event for event in events if event.event is InferenceEventType.COMPLETED]


# ── R1: inference.completed carries usage ─────────────────────


@pytest.mark.asyncio
async def test_openai_chat_completed_event_carries_the_call_usage() -> None:
    transport = _FakeTransport(chat_payload=_openai_final(prompt=11, completion=4))
    provider = OpenAIProvider("k", "https://example.test/v1", transport=transport)
    events = await _collect_events(provider)

    response = await provider.chat(_request())

    assert response.usage == UsageInfo(input_tokens=11, output_tokens=4)
    completed = _completed(events)
    assert len(completed) == 1
    assert completed[0].data == {
        "total_latency_ms": completed[0].data["total_latency_ms"],
        "usage": {"input_tokens": 11, "output_tokens": 4, "cache_read_input_tokens": None},
    }


@pytest.mark.asyncio
async def test_openai_stream_completed_event_carries_the_terminal_usage() -> None:
    transport = _FakeTransport(stream_payloads=_openai_stream({"prompt_tokens": 7, "completion_tokens": 3}))
    provider = OpenAIProvider("k", "https://example.test/v1", transport=transport)
    events = await _collect_events(provider)

    chunks = [chunk async for chunk in provider.stream(_request())]

    assert chunks[-1].usage == UsageInfo(input_tokens=7, output_tokens=3)
    completed = _completed(events)
    assert len(completed) == 1
    assert completed[0].data["usage"] == {
        "input_tokens": 7,
        "output_tokens": 3,
        "cache_read_input_tokens": None,
    }


@pytest.mark.asyncio
async def test_anthropic_chat_completed_event_carries_the_call_usage() -> None:
    transport = _FakeTransport(chat_payload=_anthropic_final(prompt=9, completion=6))
    provider = AnthropicProvider("k", "https://example.test", transport=transport)
    events = await _collect_events(provider)

    response = await provider.chat(_request())

    assert response.usage == UsageInfo(input_tokens=9, output_tokens=6)
    completed = _completed(events)
    assert len(completed) == 1
    assert completed[0].data["usage"] == {
        "input_tokens": 9,
        "output_tokens": 6,
        "cache_read_input_tokens": None,
    }


@pytest.mark.asyncio
async def test_anthropic_stream_completed_event_carries_the_terminal_usage() -> None:
    transport = _FakeTransport(stream_payloads=_anthropic_stream({"input_tokens": 8, "output_tokens": 1}))
    provider = AnthropicProvider("k", "https://example.test", transport=transport)
    events = await _collect_events(provider)

    chunks = [chunk async for chunk in provider.stream(_request())]

    assert chunks[-1].usage == UsageInfo(input_tokens=8, output_tokens=1)
    completed = _completed(events)
    assert len(completed) == 1
    assert completed[0].data["usage"] == {
        "input_tokens": 8,
        "output_tokens": 1,
        "cache_read_input_tokens": None,
    }


@pytest.mark.asyncio
async def test_react_tool_loop_emits_one_completed_per_call_summing_to_usage_total() -> None:
    transport = _FakeTransport(
        chat_payloads=[
            _openai_tool_call(prompt=10, completion=2),
            _openai_final("done", prompt=20, completion=5),
        ]
    )
    provider = OpenAIProvider("k", "https://example.test/v1", transport=transport)
    events = await _collect_events(provider)
    agent = AgentCore(
        ModelBindings(llm=ModelBinding.llm(model="m", provider=provider)),
        builtin_tools=[_echo_tool()],
    )

    response = await agent.chat("hi", skill_dispatch=False)

    assert len(transport.chat_calls) == 2
    completed = _completed(events)
    assert len(completed) == 2
    usage_total = UsageInfo(input_tokens=30, output_tokens=7)
    assert response.usage_total == usage_total
    assert response.usage == UsageInfo(input_tokens=20, output_tokens=5)
    assert sum(event.data["usage"]["input_tokens"] for event in completed) == usage_total.input_tokens
    assert sum(event.data["usage"]["output_tokens"] for event in completed) == usage_total.output_tokens


@pytest.mark.asyncio
async def test_single_shot_emits_one_completed_with_non_zero_usage() -> None:
    transport = _FakeTransport(chat_payload=_openai_final("done", prompt=4, completion=1))
    provider = OpenAIProvider("k", "https://example.test/v1", transport=transport)
    events = await _collect_events(provider)
    agent = AgentCore(ModelBindings(llm=ModelBinding.llm(model="m", provider=provider)))

    response = await agent.chat("hi", max_iterations=1, skill_dispatch=False)

    assert len(transport.chat_calls) == 1
    completed = _completed(events)
    assert len(completed) == 1
    assert completed[0].data["usage"]["input_tokens"] > 0
    assert completed[0].data["usage"]["output_tokens"] > 0
    assert response.usage_total == response.usage


# ── R2: event-handler exceptions propagate ────────────────────


class Boom(Exception):
    """A consumer fuse tripping — must never be swallowed or wrapped."""


async def test_event_handler_exception_propagates_before_the_transport_is_called() -> None:
    async def handler(event: InferenceEvent) -> None:
        if event.event is InferenceEventType.STARTED:
            raise Boom("budget exceeded")

    transport = _FakeTransport(chat_payload=_openai_final())
    provider = OpenAIProvider("k", "https://example.test/v1", on_event=handler, transport=transport)

    with pytest.raises(Boom, match="budget exceeded"):
        await provider.chat(_request())

    assert transport.chat_calls == []


async def test_stream_handler_exception_propagates_before_the_transport_is_called() -> None:
    async def handler(event: InferenceEvent) -> None:
        if event.event is InferenceEventType.STARTED:
            raise Boom("budget exceeded")

    transport = _FakeTransport(stream_payloads=_openai_stream({"prompt_tokens": 1, "completion_tokens": 1}))
    provider = OpenAIProvider("k", "https://example.test/v1", on_event=handler, transport=transport)

    with pytest.raises(Boom, match="budget exceeded"):
        [chunk async for chunk in provider.stream(_request())]

    assert transport.stream_calls == []


async def test_agent_chat_surfaces_the_handler_exception_unchanged() -> None:
    async def handler(event: InferenceEvent) -> None:
        if event.event is InferenceEventType.STARTED:
            raise Boom("budget exceeded")

    transport = _FakeTransport(chat_payload=_openai_final())
    provider = OpenAIProvider("k", "https://example.test/v1", on_event=handler, transport=transport)
    agent = AgentCore(ModelBindings(llm=ModelBinding.llm(model="m", provider=provider)))

    with pytest.raises(Boom, match="budget exceeded"):
        await agent.chat("hi", max_iterations=1, skill_dispatch=False)

    assert transport.chat_calls == []


async def test_handler_exception_on_a_later_event_is_not_wrapped() -> None:
    async def handler(event: InferenceEvent) -> None:
        if event.event is InferenceEventType.FIRST_TOKEN:
            raise Boom("metering failed")

    transport = _FakeTransport(stream_payloads=_openai_stream({"prompt_tokens": 1, "completion_tokens": 1}))
    provider = OpenAIProvider("k", "https://example.test/v1", on_event=handler, transport=transport)

    with pytest.raises(Boom, match="metering failed"):
        [chunk async for chunk in provider.stream(_request())]


async def test_handler_exception_on_completed_is_not_swallowed() -> None:
    async def handler(event: InferenceEvent) -> None:
        if event.event is InferenceEventType.COMPLETED:
            raise Boom("ledger write failed")

    transport = _FakeTransport(chat_payload=_openai_final())
    provider = OpenAIProvider("k", "https://example.test/v1", on_event=handler, transport=transport)

    with pytest.raises(Boom, match="ledger write failed"):
        await provider.chat(_request())


async def test_sync_event_handler_is_supported() -> None:
    events: list[InferenceEvent] = []

    def handler(event: InferenceEvent) -> None:
        events.append(event)

    transport = _FakeTransport(chat_payload=_openai_final())
    provider = OpenAIProvider("k", "https://example.test/v1", on_event=handler, transport=transport)

    await provider.chat(_request())

    assert [event.event for event in events] == [
        InferenceEventType.STARTED,
        InferenceEventType.COMPLETED,
    ]


# ── R3: on_event is reachable from the binding ───────────────


def test_binding_builds_its_provider_with_on_event() -> None:
    seen: list[InferenceEvent] = []

    def handler(event: InferenceEvent) -> None:
        seen.append(event)

    binding = ModelBinding.llm(
        model="m",
        api_key="k",
        base_url="https://example.test/v1",
        provider_name="openai",
        on_event=handler,
    )

    provider = binding.get_provider()

    assert isinstance(provider, OpenAIProvider)
    assert provider.on_event is handler


def test_create_provider_accepts_on_event() -> None:
    from cliver_core.provider.providers import create_provider

    seen: list[InferenceEvent] = []

    def handler(event: InferenceEvent) -> None:
        seen.append(event)

    provider = create_provider(
        api_key="k",
        base_url="https://example.test/v1",
        provider_name="openai",
        on_event=handler,
    )

    assert provider.on_event is handler


def test_binding_rejects_on_event_when_a_provider_instance_is_supplied() -> None:
    def handler(event: InferenceEvent) -> None:
        return None

    provider = OpenAIProvider("k", "https://example.test/v1", transport=_FakeTransport())
    binding = ModelBinding.llm(model="m", provider=provider, on_event=handler)

    with pytest.raises(ValueError, match="on_event"):
        binding.get_provider()


# ── R5: provider failures raise ProviderError(retryable=...) ──


class _VendorError(Exception):
    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class APITimeoutError(Exception):
    """Vendor SDK classes are matched by their conventional names too."""


def test_provider_error_defaults_to_not_retryable() -> None:
    from cliver_core.provider import ProviderError

    error = ProviderError("nope")

    assert error.retryable is False
    assert str(error) == "nope"


async def test_retryable_transport_error_becomes_provider_error_with_cause() -> None:
    from cliver_core.provider import ProviderError

    vendor = _VendorError("rate limited", status_code=429)
    transport = _FakeTransport(chat_error=vendor)
    provider = OpenAIProvider("k", "https://example.test/v1", transport=transport)

    with pytest.raises(ProviderError) as info:
        await provider.chat(_request())

    assert info.value.retryable is True
    assert info.value.__cause__ is vendor
    assert str(info.value) == "rate limited"


async def test_non_retryable_transport_error_becomes_provider_error_with_cause() -> None:
    from cliver_core.provider import ProviderError

    vendor = _VendorError("bad request", status_code=400)
    transport = _FakeTransport(chat_error=vendor)
    provider = OpenAIProvider("k", "https://example.test/v1", transport=transport)

    with pytest.raises(ProviderError) as info:
        await provider.chat(_request())

    assert info.value.retryable is False
    assert info.value.__cause__ is vendor


async def test_vendor_timeout_class_is_retryable() -> None:
    from cliver_core.provider import ProviderError

    transport = _FakeTransport(chat_error=APITimeoutError("timed out"))
    provider = AnthropicProvider("k", "https://example.test", transport=transport)

    with pytest.raises(ProviderError) as info:
        await provider.chat(_request())

    assert info.value.retryable is True


async def test_stream_transport_error_becomes_provider_error_with_cause() -> None:
    from cliver_core.provider import ProviderError

    vendor = _VendorError("server exploded", status_code=503)
    transport = _FakeTransport(stream_error=vendor)
    provider = OpenAIProvider("k", "https://example.test/v1", transport=transport)

    with pytest.raises(ProviderError) as info:
        [chunk async for chunk in provider.stream(_request())]

    assert info.value.retryable is True
    assert info.value.__cause__ is vendor


async def test_provider_error_event_reports_retryability() -> None:
    from cliver_core.provider import ProviderError

    events: list[InferenceEvent] = []

    async def handler(event: InferenceEvent) -> None:
        events.append(event)

    vendor = _VendorError("rate limited", status_code=429)
    transport = _FakeTransport(chat_error=vendor)
    provider = OpenAIProvider("k", "https://example.test/v1", on_event=handler, transport=transport)

    with pytest.raises(ProviderError):
        await provider.chat(_request())

    error = next(event for event in events if event.event is InferenceEventType.ERROR)
    assert error.data == {"error": "rate limited", "retryable": True}


async def test_agent_does_not_wrap_provider_error() -> None:
    from cliver_core.provider import ProviderError

    vendor = _VendorError("rate limited", status_code=429)
    transport = _FakeTransport(chat_error=vendor)
    provider = OpenAIProvider("k", "https://example.test/v1", transport=transport)
    agent = AgentCore(ModelBindings(llm=ModelBinding.llm(model="m", provider=provider)))

    with pytest.raises(ProviderError) as info:
        await agent.chat("hi", max_iterations=1, skill_dispatch=False)

    assert type(info.value) is ProviderError
    assert info.value.__cause__ is vendor


# ── R6: single-shot contract — one provider call, no agentic extras ──


def _skill_manager(tmp_path: Path) -> SkillManager:
    root = tmp_path / "skills"
    (root / "reports").mkdir(parents=True)
    (root / "reports" / "SKILL.md").write_text(
        "---\nname: reports\ndescription: Writes business reports.\n---\n\n# Reports\n\nDo reports.\n",
        encoding="utf-8",
    )
    return SkillManager(sources=[root])


@pytest.mark.asyncio
async def test_single_shot_makes_exactly_one_bare_provider_call(tmp_path: Path) -> None:
    transport = _FakeTransport(chat_payload=_openai_final("ok"))
    provider = OpenAIProvider("k", "https://example.test/v1", transport=transport)
    # A skill manager is mounted, so a dispatch-on call would inject the
    # catalogue and FindSkills/ActivateSkill tools — dispatch-off must not.
    agent = AgentCore(
        ModelBindings(llm=ModelBinding.llm(model="m", provider=provider)),
        skill_manager=_skill_manager(tmp_path),
    )

    response = await agent.chat("hi", max_iterations=1, skill_dispatch=False)

    assert len(transport.chat_calls) == 1
    body = transport.chat_calls[0]
    assert "tools" not in body
    # No injected system/catalogue turn and no follow-up message: the call
    # carries the caller's input and nothing AgentCore added.
    assert body["messages"] == [{"role": "user", "content": "hi"}]
    assert response.message.content == "ok"


# ── R7: image content parts reach the provider body unchanged ──


def test_openai_native_preserves_image_content_parts() -> None:
    parts = [
        {"type": "text", "text": "describe"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ]

    native = CLIverMessage(role="user", content=parts).to_native("openai")

    assert native == {"role": "user", "content": parts}


@pytest.mark.asyncio
async def test_agent_vision_passthrough_keeps_data_uri_in_transport_body() -> None:
    transport = _FakeTransport(chat_payload=_openai_final("ok"))
    provider = OpenAIProvider("k", "https://example.test/v1", transport=transport)
    agent = AgentCore(ModelBindings(llm=ModelBinding.llm(model="m", provider=provider)))

    media = MediaContent(type=MediaType.IMAGE, data="AAAA", mime_type="image/png")
    await agent.chat("describe", media=[media], max_iterations=1, skill_dispatch=False)

    content = transport.chat_calls[0]["messages"][0]["content"]
    assert {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}} in content
