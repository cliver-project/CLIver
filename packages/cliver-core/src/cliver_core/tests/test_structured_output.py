"""``chat_structured()`` — schema instruction, protocol mapping, validation."""

from __future__ import annotations

import asyncio
import json
from typing import Literal

import pytest
from pydantic import BaseModel, ValidationError

from cliver_core.agent import AgentCore
from cliver_core.bindings import ModelBinding, ModelBindings
from cliver_core.errors import RunTimeoutError
from cliver_core.media import MediaContent, MediaType
from cliver_core.messages import CLIverMessage, UsageInfo
from cliver_core.provider import CLIverRequest, CLIverResponse, Provider
from cliver_core.structured import StructuredOutputError, format_error_for_retry


class PartVerdict(BaseModel):
    part_id: str
    verdict: Literal["pass", "fail"]
    confidence: float


PART_JSON = '{"part_id": "P-7", "verdict": "pass", "confidence": 0.93}'


class WideVerdict(BaseModel):
    """Twelve required fields — enough to exercise the retry-error cap."""

    f01: int
    f02: int
    f03: int
    f04: int
    f05: int
    f06: int
    f07: int
    f08: int
    f09: int
    f10: int
    f11: int
    f12: int


class _ScriptedProvider(Provider):
    protocol = "openai"
    brand = "mock-scripted"

    def __init__(self, *responses: CLIverResponse) -> None:
        super().__init__(api_key="", base_url="")
        self.responses = list(responses)
        self.requests: list[CLIverRequest] = []

    async def chat(self, request: CLIverRequest) -> CLIverResponse:
        self.requests.append(request)
        return self.responses.pop(0)

    async def stream(self, request):  # pragma: no cover
        raise AssertionError("stream() is not used here")


class _SlowProvider(Provider):
    protocol = "openai"
    brand = "mock-slow"

    def __init__(self) -> None:
        super().__init__(api_key="", base_url="")
        self.cancelled = False

    async def chat(self, request: CLIverRequest) -> CLIverResponse:
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        return CLIverResponse(message=CLIverMessage(role="assistant", content=PART_JSON))

    async def stream(self, request):  # pragma: no cover
        raise AssertionError("stream() is not used here")


def _text(content: str, usage: tuple[int, int] = (1, 1)) -> CLIverResponse:
    return CLIverResponse(
        message=CLIverMessage(role="assistant", content=content),
        usage=UsageInfo(input_tokens=usage[0], output_tokens=usage[1]),
    )


def _agent(provider: Provider, *, protocol: str = "openai", **options) -> AgentCore:
    binding = ModelBinding.llm(model="mock-model", protocol=protocol, provider=provider)
    return AgentCore(ModelBindings(llm=binding), **options)


def _last_user_text(request: CLIverRequest) -> str:
    for message in reversed(request.messages):
        if message.role == "user" and isinstance(message.content, str):
            return message.content
    raise AssertionError("request has no string user turn")


@pytest.mark.asyncio
async def test_invalid_json_then_valid_retries_and_sums_usage():
    provider = _ScriptedProvider(
        _text("not json", (1, 2)),
        _text(PART_JSON, (3, 4)),
    )
    conversation = [CLIverMessage(role="user", content="earlier turn")]
    before = list(conversation)

    result = await _agent(provider).chat_structured(
        "inspect the part",
        schema=PartVerdict,
        conversation=conversation,
        run_id="structured-1",
    )

    assert result.attempts == 2
    assert result.value == PartVerdict(part_id="P-7", verdict="pass", confidence=0.93)
    assert result.usage_total is not None
    assert result.usage_total.input_tokens == 4
    assert result.usage_total.output_tokens == 6
    assert result.run_id == "structured-1"
    assert conversation == before  # the caller's list is never mutated
    assert [request.run_id for request in provider.requests] == ["structured-1", "structured-1"]

    # One system message carrying the schema instruction, on every attempt.
    systems = [[m.content for m in request.messages if m.role == "system"] for request in provider.requests]
    assert all(len(parts) == 1 for parts in systems)
    assert all("JSON" in parts[0] for parts in systems)

    # The retry carries the bad answer and a corrective user turn.
    retry_contents = [getattr(m.content, "text", m.content) for m in provider.requests[1].messages]
    assert "not json" in retry_contents
    assert any(isinstance(c, str) and "did not conform" in c for c in retry_contents)


@pytest.mark.asyncio
async def test_openai_auto_mode_sets_json_object_response_format():
    provider = _ScriptedProvider(_text(PART_JSON))

    await _agent(provider).chat_structured("go", schema=PartVerdict, mode="auto")

    response_format = provider.requests[0].options["response_format"]
    assert response_format == {"type": "json_object"}


@pytest.mark.asyncio
async def test_openai_json_schema_mode_sends_the_strict_schema():
    provider = _ScriptedProvider(_text(PART_JSON))

    await _agent(provider).chat_structured("go", schema=PartVerdict, mode="json_schema")

    response_format = provider.requests[0].options["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["name"] == "PartVerdict"
    assert response_format["json_schema"]["strict"] is True
    assert response_format["json_schema"]["schema"] == PartVerdict.model_json_schema()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["auto", "json_schema"])
async def test_anthropic_protocol_sets_output_config(mode):
    provider = _ScriptedProvider(_text(PART_JSON))
    conversation: list[CLIverMessage] = []

    await _agent(provider, protocol="anthropic").chat_structured(
        "go", schema=PartVerdict, mode=mode, conversation=conversation
    )

    output_config = provider.requests[0].options["output_config"]
    assert output_config == {"format": {"type": "json_schema", "schema": PartVerdict.model_json_schema()}}


@pytest.mark.asyncio
async def test_anthropic_json_object_mode_falls_back_to_prompt():
    provider = _ScriptedProvider(_text(PART_JSON))

    await _agent(provider, protocol="anthropic").chat_structured("go", schema=PartVerdict, mode="json_object")

    assert "response_format" not in provider.requests[0].options
    assert "output_config" not in provider.requests[0].options


@pytest.mark.asyncio
async def test_prompt_mode_sets_no_wire_option():
    provider = _ScriptedProvider(_text(PART_JSON))

    await _agent(provider).chat_structured("go", schema=PartVerdict, mode="prompt")

    assert "response_format" not in provider.requests[0].options
    assert "output_config" not in provider.requests[0].options


@pytest.mark.asyncio
async def test_caller_options_win_over_the_wire_mapping():
    provider = _ScriptedProvider(_text(PART_JSON))
    caller_format = {"type": "json_object", "custom": True}

    await _agent(provider).chat_structured(
        "go", schema=PartVerdict, mode="json_schema", options={"response_format": caller_format}
    )

    assert provider.requests[0].options["response_format"] == caller_format


@pytest.mark.asyncio
async def test_dict_schema_returns_a_plain_dict():
    schema = {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}
    provider = _ScriptedProvider(_text('{"a": 1}'))

    result = await _agent(provider).chat_structured("go", schema=schema)

    assert result.value == {"a": 1}
    assert result.attempts == 1


@pytest.mark.asyncio
async def test_fenced_json_is_accepted():
    provider = _ScriptedProvider(_text("```json\n" + PART_JSON + "\n```"))

    result = await _agent(provider).chat_structured("go", schema=PartVerdict)

    assert result.attempts == 1
    assert result.value.part_id == "P-7"


@pytest.mark.asyncio
async def test_prose_around_json_fails_and_retries():
    provider = _ScriptedProvider(_text("Here you go: " + PART_JSON), _text(PART_JSON))

    result = await _agent(provider).chat_structured("go", schema=PartVerdict)

    assert result.attempts == 2


@pytest.mark.asyncio
async def test_exhausted_retries_raise_with_diagnostics():
    provider = _ScriptedProvider(_text("still not json"), _text("nope"))

    with pytest.raises(StructuredOutputError) as info:
        await _agent(provider).chat_structured("go", schema=PartVerdict, retries=1, run_id="so-fail")

    error = info.value
    assert error.attempts == 2
    assert error.raw_content == "nope"
    assert error.run_id == "so-fail"
    assert error.response is not None
    assert "nope" not in error.last_error  # the error is the parse/validation failure
    assert len(provider.requests) == 2


def test_format_error_for_retry_is_a_bounded_targeted_diff():
    with pytest.raises(ValidationError) as info:
        WideVerdict.model_validate({})

    feedback = format_error_for_retry(info.value)
    lines = feedback.splitlines()

    assert len(lines) == 11  # ten field diffs plus the remainder line
    assert lines[0] == "- f01: Field required"
    assert lines[-1] == "- ... and 2 more error(s)"
    assert "errors.pydantic.dev" not in feedback
    assert "input_value" not in feedback
    assert "input_type" not in feedback


def test_format_error_for_retry_keeps_short_errors_verbatim():
    with pytest.raises(json.JSONDecodeError) as info:
        json.loads("not json")

    assert format_error_for_retry(info.value) == str(info.value)
    assert format_error_for_retry(ValueError("the model returned no content")) == ("the model returned no content")


@pytest.mark.asyncio
async def test_retry_feedback_uses_the_targeted_diff():
    bad = '{"part_id": "P-7", "verdict": "maybe", "confidence": "high"}'
    provider = _ScriptedProvider(_text(bad), _text(PART_JSON))

    await _agent(provider).chat_structured("go", schema=PartVerdict)

    corrective = _last_user_text(provider.requests[1])
    assert "verdict: Input should be 'pass' or 'fail'" in corrective
    assert "confidence: Input should be a valid number" in corrective
    assert "errors.pydantic.dev" not in corrective
    assert "input_value" not in corrective


@pytest.mark.asyncio
async def test_retry_history_keeps_only_the_latest_bad_answer():
    provider = _ScriptedProvider(_text("first bad"), _text("second bad"), _text(PART_JSON))

    result = await _agent(provider).chat_structured("go", schema=PartVerdict, retries=2)

    assert result.attempts == 3
    third = provider.requests[2].messages
    contents = [getattr(m.content, "text", m.content) for m in third]
    assert "second bad" in contents
    assert "first bad" not in contents  # older attempts are dropped, not replayed
    assert contents.count("go") == 1  # the original task is replayed exactly once
    assert sum(1 for m in third if m.role == "assistant") == 1


@pytest.mark.asyncio
async def test_last_error_keeps_the_full_validation_text():
    bad = '{"part_id": "P-7", "verdict": "maybe", "confidence": "high"}'
    provider = _ScriptedProvider(_text(bad))

    with pytest.raises(StructuredOutputError) as info:
        await _agent(provider).chat_structured("go", schema=PartVerdict, retries=0)

    # Diagnostics keep the raw pydantic dump even though the retry prompt does not.
    assert "errors.pydantic.dev" in info.value.last_error
    assert "input_value" in info.value.last_error


@pytest.mark.asyncio
async def test_retries_zero_makes_exactly_one_attempt():
    provider = _ScriptedProvider(_text("not json"))

    with pytest.raises(StructuredOutputError) as info:
        await _agent(provider).chat_structured("go", schema=PartVerdict, retries=0)

    assert info.value.attempts == 1
    assert len(provider.requests) == 1


@pytest.mark.asyncio
async def test_retries_zero_success_makes_exactly_one_attempt():
    provider = _ScriptedProvider(_text(PART_JSON))

    result = await _agent(provider).chat_structured("go", schema=PartVerdict, retries=0)

    assert result.attempts == 1
    assert len(provider.requests) == 1


@pytest.mark.asyncio
async def test_media_is_preserved_in_retry_conversations():
    media = MediaContent(
        type=MediaType.IMAGE, data="https://example.test/part.png", mime_type="image/png", source="url"
    )
    provider = _ScriptedProvider(_text("not json"), _text(PART_JSON))

    result = await _agent(provider).chat_structured("inspect", schema=PartVerdict, media=[media])

    assert result.attempts == 2
    user_messages = [m for m in provider.requests[1].messages if m.role == "user"]
    media_blocks = [
        block
        for message in user_messages
        if isinstance(message.content, list)
        for block in message.content
        if block.get("type") != "text"
    ]
    assert len(media_blocks) == 1  # embedded once, not duplicated by the retry


@pytest.mark.asyncio
async def test_run_timeout_bounds_the_structured_run():
    provider = _SlowProvider()

    with pytest.raises(RunTimeoutError):
        await _agent(provider).chat_structured("go", schema=PartVerdict, run_timeout=0.05, run_id="so-timeout")

    assert provider.cancelled is True


@pytest.mark.asyncio
async def test_constructor_run_timeout_applies_to_chat_structured():
    with pytest.raises(RunTimeoutError):
        await _agent(_SlowProvider(), run_timeout=0.05).chat_structured("go", schema=PartVerdict)


@pytest.mark.asyncio
async def test_invalid_arguments_are_rejected():
    agent = _agent(_ScriptedProvider(_text(PART_JSON)))

    with pytest.raises(ValueError, match="retries"):
        await agent.chat_structured("go", schema=PartVerdict, retries=-1)
    with pytest.raises(ValueError, match="mode"):
        await agent.chat_structured("go", schema=PartVerdict, mode="yaml")
    with pytest.raises(TypeError, match="schema"):
        await agent.chat_structured("go", schema=42)  # type: ignore[arg-type]
