"""Explicit provider construction and startup preflight.

``ModelBinding.build_provider()`` is the documented name for the idempotent
"build once for this loop" call (``get_provider()`` stays as a compatibility
alias).  ``preflight()`` builds every configured provider on the calling loop
without rebuilding or closing anything, so a missing endpoint or credential
fails at startup instead of on the first request.
"""

from __future__ import annotations

import asyncio

import pytest

from cliver_core.bindings import ModelBinding, ModelBindings
from cliver_core.errors import LoopAffinityError
from cliver_core.messages import CLIverMessage
from cliver_core.provider import CLIverRequest, CLIverResponse, Provider


class _FakeProvider(Provider):
    protocol = "openai"
    brand = "fake-preflight"

    def __init__(self) -> None:
        super().__init__(api_key="", base_url="")
        self.closed = 0
        self.built_with_key: str | None = None

    async def chat(self, request: CLIverRequest) -> CLIverResponse:
        return CLIverResponse(message=CLIverMessage(role="assistant", content="ok"))

    async def stream(self, request):  # pragma: no cover
        raise AssertionError("stream() is not used here")

    async def close(self) -> None:
        self.closed += 1


def _fake_builder(monkeypatch) -> list[_FakeProvider]:
    built: list[_FakeProvider] = []

    def fake_create(**kwargs):
        provider = _FakeProvider()
        provider.built_with_key = kwargs["api_key"]
        built.append(provider)
        return provider

    monkeypatch.setattr("cliver_core.bindings.create_provider", fake_create)
    return built


@pytest.mark.asyncio
async def test_build_provider_is_the_documented_name_for_get_provider(monkeypatch) -> None:
    built = _fake_builder(monkeypatch)
    binding = ModelBinding.llm(model="mock-model", api_key="k", base_url="https://example.test/v1")

    provider = binding.build_provider()

    assert provider is built[0]
    assert binding.get_provider() is provider  # compatibility alias
    assert binding.build_provider() is provider  # idempotent


@pytest.mark.asyncio
async def test_preflight_builds_once_and_is_idempotent(monkeypatch) -> None:
    built = _fake_builder(monkeypatch)
    binding = ModelBinding.llm(model="mock-model", api_key="k", base_url="https://example.test/v1")

    await binding.preflight()
    await binding.preflight()

    assert len(built) == 1
    assert built[0].closed == 0


@pytest.mark.asyncio
async def test_preflight_resolves_credential_provider_once(monkeypatch) -> None:
    built = _fake_builder(monkeypatch)
    calls: list[int] = []

    def credential_provider() -> str:
        calls.append(1)
        return "fresh-key"

    binding = ModelBinding.llm(
        model="mock-model",
        api_key="",
        base_url="https://example.test/v1",
        credential_provider=credential_provider,
    )

    await binding.preflight()
    await binding.preflight()

    assert calls == [1]  # resolved for the build; the built binding skips it
    assert built[0].built_with_key == "fresh-key"
    assert binding.api_key == "fresh-key"


@pytest.mark.asyncio
async def test_preflight_surfaces_a_missing_base_url() -> None:
    binding = ModelBinding.llm(model="mock-model", api_key="k")

    with pytest.raises(ValueError, match="No base_url"):
        await binding.preflight()


@pytest.mark.asyncio
async def test_preflight_surfaces_a_missing_credential() -> None:
    binding = ModelBinding.llm(model="mock-model", base_url="https://example.test/v1")

    with pytest.raises(ValueError, match="needs an API key"):
        await binding.preflight()


@pytest.mark.asyncio
async def test_preflight_leaves_the_binding_unchanged_when_the_build_fails(monkeypatch) -> None:
    attempts: list[str] = []

    def fail_then_succeed(**kwargs):
        attempts.append(kwargs["api_key"])
        if len(attempts) == 1:
            raise ValueError("build failed")
        return _FakeProvider()

    monkeypatch.setattr("cliver_core.bindings.create_provider", fail_then_succeed)
    binding = ModelBinding.llm(
        model="mock-model",
        api_key="old-key",
        base_url="https://example.test/v1",
        credential_provider=lambda: "new-key",
    )

    with pytest.raises(ValueError, match="build failed"):
        await binding.preflight()

    assert binding.api_key == "old-key"  # the previous key was restored after the failed build
    binding.build_provider()
    assert attempts == ["new-key", "old-key"]


@pytest.mark.asyncio
async def test_preflight_skips_credential_resolution_for_a_supplied_provider() -> None:
    calls: list[int] = []
    provider = _FakeProvider()

    def credential_provider() -> str:
        calls.append(1)
        return "k"

    binding = ModelBinding.llm(
        model="mock-model",
        provider=provider,
        credential_provider=credential_provider,
    )

    await binding.preflight()

    assert calls == []  # nothing is built, so nothing needs a credential
    assert provider.closed == 0


def test_preflight_claims_the_loop_for_a_loop_bound_supplied_provider() -> None:
    binding = ModelBinding.llm(model="mock-model", provider=_FakeProvider())

    asyncio.run(binding.preflight())
    with pytest.raises(LoopAffinityError, match="another event loop"):
        asyncio.run(binding.preflight())


@pytest.mark.asyncio
async def test_model_bindings_preflight_builds_primary_and_fallbacks_once(monkeypatch) -> None:
    built = _fake_builder(monkeypatch)
    bindings = ModelBindings(
        llm=[
            ModelBinding.llm(model="primary", api_key="k", base_url="https://example.test/v1"),
            ModelBinding.llm(model="fallback", api_key="k", base_url="https://example.test/v1"),
        ]
    )

    await bindings.preflight()
    await bindings.preflight()

    assert len(built) == 2
    assert built[0] is not built[1]


@pytest.mark.asyncio
async def test_model_bindings_preflight_names_the_failing_purpose() -> None:
    bindings = ModelBindings(llm=ModelBinding.llm(model="bad-model", api_key="k"))

    with pytest.raises(ValueError, match=r"bindings\.llm.*bad-model"):
        await bindings.preflight()
