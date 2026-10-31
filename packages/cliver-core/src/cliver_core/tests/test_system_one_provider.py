"""The HTTP System One provider — shared ``/v1/systemone`` wire.

The POST seam (``_post_json_endpoint``) is stubbed; no network.  The live
wire was verified against Aliyun Bailian's decision-model API, whose payload
this provider parses (see ``test_system_one_types.py``).
"""

from __future__ import annotations

import pytest

from cliver_core.errors import CapabilityNotSupported, ProviderError
from cliver_core.provider import Provider
from cliver_core.provider.providers.system_one import SystemOneProvider
from cliver_core.system_one import Choice, Noul, SystemOneResult


def _provider() -> SystemOneProvider:
    return SystemOneProvider(api_key="k", base_url="https://ws-123.cn-beijing.maas.aliyuncs.com/compatible-mode")


def _stub_post(monkeypatch, provider: SystemOneProvider, payload: dict):
    calls: list[dict] = []

    async def fake_post(path: str, body: dict, *, label: str, timeout: float = 120.0) -> dict:
        calls.append({"path": path, "body": body, "label": label, "timeout": timeout})
        return payload

    monkeypatch.setattr(provider, "_post_json_endpoint", fake_post)
    return calls


def test_system_one_provider_locks_its_surface() -> None:
    provider = _provider()
    assert provider.provider_name() == "system-one"
    assert provider.protocol == "system-one"
    assert SystemOneProvider.DEFAULT_PATH == "/v1/systemone"
    assert provider.timeout == 10.0


@pytest.mark.asyncio
async def test_system_one_posts_state_and_typed_questions(monkeypatch) -> None:
    provider = _provider()
    calls = _stub_post(
        monkeypatch,
        provider,
        {"model": "decision-model-preview", "answers": {"billing": {"type": "noul", "noul": 0.9}}},
    )

    result = await provider.system_one(
        state="I was charged twice",
        questions={"billing": Noul(instructions="Is this about billing?")},
        model="decision-model-preview",
    )

    call = calls[-1]
    assert call["path"] == "/v1/systemone"
    assert call["label"] == "system one"
    assert call["timeout"] == 10.0
    assert call["body"] == {
        "model": "decision-model-preview",
        "state": "I was charged twice",
        "questions": {"billing": {"type": "noul", "instructions": "Is this about billing?"}},
    }
    assert isinstance(result, SystemOneResult)
    assert result.answers["billing"].noul == pytest.approx(0.9)


@pytest.mark.asyncio
async def test_system_one_accepts_a_path_override_and_extra_options(monkeypatch) -> None:
    provider = _provider()
    calls = _stub_post(monkeypatch, provider, {"answers": {}})

    await provider.system_one(
        state={"message": "hi"},
        questions={"team": Choice(instructions="Which?", criteria={"a": "x", "b": "y"})},
        model="m",
        path="/custom/systemone",
        timeout=3.0,
        temperature=0.1,
    )

    assert calls[-1]["path"] == "/custom/systemone"
    assert calls[-1]["timeout"] == 3.0
    assert calls[-1]["body"]["temperature"] == 0.1


@pytest.mark.asyncio
async def test_system_one_rejects_an_empty_question_set() -> None:
    provider = _provider()
    with pytest.raises(ValueError, match="question"):
        await provider.system_one(state="x", questions={}, model="m")


@pytest.mark.asyncio
async def test_system_one_failure_is_a_retryable_provider_error(monkeypatch) -> None:
    provider = _provider()

    async def failing_post(*args, **kwargs):
        raise ConnectionError("connection reset")

    monkeypatch.setattr(provider, "_post_json_endpoint", failing_post)

    with pytest.raises(ProviderError) as info:
        await provider.system_one(
            state="x",
            questions={"q": Noul(instructions="Yes?")},
            model="m",
        )
    assert info.value.retryable is True
    assert isinstance(info.value.__cause__, ConnectionError)


@pytest.mark.asyncio
async def test_system_one_passes_a_provider_error_through(monkeypatch) -> None:
    provider = _provider()

    async def failing_post(*args, **kwargs):
        raise ProviderError("boom", retryable=False)

    monkeypatch.setattr(provider, "_post_json_endpoint", failing_post)

    with pytest.raises(ProviderError, match="boom") as info:
        await provider.system_one(state="x", questions={"q": Noul(instructions="Yes?")}, model="m")
    assert info.value.retryable is False


@pytest.mark.asyncio
async def test_chat_and_stream_report_the_capability_unsupported() -> None:
    provider = _provider()
    with pytest.raises(CapabilityNotSupported):
        await provider.chat(None)  # type: ignore[arg-type]
    with pytest.raises(CapabilityNotSupported):
        async for _ in provider.stream(None):  # type: ignore[arg-type]
            pass


@pytest.mark.asyncio
async def test_the_base_provider_reports_system_one_unsupported() -> None:
    class _Plain(Provider):
        protocol = "openai"
        brand = "plain"

        async def chat(self, request):  # pragma: no cover
            raise NotImplementedError

        async def stream(self, request):  # pragma: no cover
            if False:
                yield None

    with pytest.raises(CapabilityNotSupported, match="system one"):
        await _Plain(api_key="k", base_url="https://example/v1").system_one(
            state="x", questions={"q": Noul(instructions="Yes?")}, model="m"
        )
