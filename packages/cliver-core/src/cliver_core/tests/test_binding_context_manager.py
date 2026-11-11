"""Binding lifecycle as an async context manager.

``async with ModelBinding(...) as binding:`` and ``async with ModelBindings(...)
as bindings:`` close on exit, on normal and exceptional paths.  Ownership
matches ``aclose()``: only providers the binding built are closed; a
caller-supplied ``provider=`` stays open.
"""

from __future__ import annotations

import pytest

from cliver_core.bindings import ModelBinding, ModelBindings
from cliver_core.messages import CLIverMessage
from cliver_core.provider import CLIverRequest, CLIverResponse, Provider


class _FakeProvider(Provider):
    protocol = "openai"
    brand = "fake-context-manager"

    def __init__(self) -> None:
        super().__init__(api_key="", base_url="")
        self.closed = 0

    async def chat(self, request: CLIverRequest) -> CLIverResponse:
        return CLIverResponse(message=CLIverMessage(role="assistant", content="ok"))

    async def stream(self, request):  # pragma: no cover — chat test double only
        if False:
            yield None

    async def close(self) -> None:
        self.closed += 1


def _build_binding(monkeypatch) -> tuple[ModelBinding, _FakeProvider]:
    provider = _FakeProvider()
    monkeypatch.setattr("cliver_core.bindings.create_provider", lambda **kwargs: provider)
    return ModelBinding.llm(model="mock-model", base_url="https://example.test"), provider


async def test_model_binding_context_manager_closes_a_built_provider(monkeypatch) -> None:
    binding, provider = _build_binding(monkeypatch)

    async with binding as entered:
        assert entered is binding
        binding.build_provider()

    assert provider.closed == 1


async def test_model_binding_context_manager_keeps_a_supplied_provider_open() -> None:
    provider = _FakeProvider()
    binding = ModelBinding.llm(model="mock-model", provider=provider)

    async with binding as entered:
        assert entered is binding
        binding.build_provider()

    assert provider.closed == 0


async def test_model_binding_context_manager_closes_on_an_exception(monkeypatch) -> None:
    binding, provider = _build_binding(monkeypatch)

    with pytest.raises(RuntimeError, match="boom"):
        async with binding:
            binding.build_provider()
            raise RuntimeError("boom")

    assert provider.closed == 1


async def test_model_bindings_context_manager_closes_every_built_provider(monkeypatch) -> None:
    providers: list[_FakeProvider] = []

    def create(**kwargs) -> _FakeProvider:
        provider = _FakeProvider()
        providers.append(provider)
        return provider

    monkeypatch.setattr("cliver_core.bindings.create_provider", create)
    bindings = ModelBindings(
        llm=ModelBinding.llm(model="mock-llm", base_url="https://example.test"),
        embed=ModelBinding.embed(model="mock-embed", base_url="https://example.test"),
    )

    async with bindings as entered:
        assert entered is bindings
        bindings.llm.build_provider()
        bindings.embed.build_provider()

    assert [provider.closed for provider in providers] == [1, 1]
