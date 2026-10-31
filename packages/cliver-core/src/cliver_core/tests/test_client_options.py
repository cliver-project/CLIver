"""Per-binding SDK client options — ``client_timeout`` and ``client_max_retries``.

The built-in OpenAI / Anthropic SDK transports are built with SDK defaults
unless a binding tunes them.  These knobs are client-construction settings,
distinct from a binding's request ``params`` (where ``timeout`` means a
per-request timeout).
"""

from __future__ import annotations

import pytest

from cliver_core.bindings import ModelBinding
from cliver_core.messages import CLIverMessage, CLIverMessageChunk
from cliver_core.presets import deepseek
from cliver_core.provider import CLIverRequest, CLIverResponse, Provider
from cliver_core.provider.providers import create_provider
from cliver_core.provider.providers.system_one import SystemOneProvider


class _PlainProvider(Provider):
    """A custom provider class that knows nothing about SDK client options."""

    protocol = "openai"

    def __init__(self, api_key: str, base_url: str, **kwargs) -> None:
        super().__init__(api_key, base_url)

    async def chat(self, request: CLIverRequest) -> CLIverResponse:  # pragma: no cover - not called
        return CLIverResponse(message=CLIverMessage(role="assistant", content="ok"))

    async def stream(self, request: CLIverRequest):  # pragma: no cover - not called
        if False:
            yield CLIverMessageChunk()


@pytest.mark.asyncio
async def test_openai_transport_client_uses_the_binding_options():
    binding = ModelBinding.llm(
        model="m",
        api_key="k",
        base_url="https://example.test/v1",
        client_timeout=7.5,
        client_max_retries=0,
    )

    client = binding.get_provider().transport._client

    assert client.timeout.read == 7.5
    assert client.timeout.connect == 7.5
    assert client.max_retries == 0
    await binding.aclose()


@pytest.mark.asyncio
async def test_anthropic_transport_client_uses_the_binding_options():
    binding = ModelBinding.llm(
        model="m",
        api_key="k",
        base_url="https://example.test",
        protocol="anthropic",
        client_timeout=3.25,
        client_max_retries=1,
    )

    client = binding.get_provider().transport._client

    assert client.timeout.read == 3.25
    assert client.max_retries == 1
    await binding.aclose()


@pytest.mark.asyncio
async def test_defaults_are_the_sdk_defaults():
    binding = ModelBinding.llm(model="m", api_key="k", base_url="https://example.test/v1")

    client = binding.get_provider().transport._client

    assert client.max_retries == 2
    await binding.aclose()


@pytest.mark.asyncio
async def test_arefresh_rebuilds_with_the_same_client_options():
    binding = ModelBinding.llm(
        model="m",
        api_key="k",
        base_url="https://example.test/v1",
        credential_provider=lambda: "rotated",
        client_timeout=7.5,
        client_max_retries=0,
    )
    old = binding.get_provider()

    await binding.arefresh()

    new = binding.get_provider()
    assert new is not old
    assert new.transport._client.timeout.read == 7.5
    assert new.transport._client.max_retries == 0
    assert old.transport._client.is_closed() is True
    await binding.aclose()


@pytest.mark.asyncio
async def test_create_provider_forwards_client_options():
    provider = create_provider(
        api_key="k",
        base_url="https://example.test/v1",
        client_timeout=2.5,
        client_max_retries=0,
    )

    assert provider.transport._client.timeout.read == 2.5
    assert provider.transport._client.max_retries == 0
    await provider.close()


def test_client_options_rejected_for_a_provider_class_without_an_sdk_transport():
    with pytest.raises(ValueError, match="client_timeout"):
        create_provider(
            api_key="k",
            base_url="https://example.test",
            provider_class=SystemOneProvider,
            client_timeout=5,
        )


def test_client_options_rejected_with_a_supplied_provider():
    supplied = _PlainProvider("k", "https://example.test/v1")

    with pytest.raises(ValueError, match="client_timeout"):
        ModelBinding.llm(model="m", provider=supplied, client_timeout=5)


def test_client_options_absent_keep_custom_provider_classes_working():
    provider = create_provider(api_key="k", base_url="https://example.test", provider_class=_PlainProvider)

    assert isinstance(provider, _PlainProvider)


@pytest.mark.parametrize("value", [0, -1, True, float("nan"), float("inf"), "5"])
def test_invalid_client_timeout_is_rejected(value):
    with pytest.raises(ValueError, match="client_timeout"):
        ModelBinding.llm(model="m", api_key="k", base_url="https://example.test/v1", client_timeout=value)


@pytest.mark.parametrize("value", [-1, True, 1.5, "2"])
def test_invalid_client_max_retries_is_rejected(value):
    with pytest.raises(ValueError, match="client_max_retries"):
        ModelBinding.llm(model="m", api_key="k", base_url="https://example.test/v1", client_max_retries=value)


@pytest.mark.parametrize(
    "factory",
    [
        ModelBinding.llm,
        ModelBinding.image,
        ModelBinding.video,
        ModelBinding.tts,
        ModelBinding.asr,
        ModelBinding.embed,
        ModelBinding.rerank,
        ModelBinding.system_one,
    ],
)
def test_every_purpose_factory_accepts_client_options(factory):
    binding = factory(
        model="m",
        api_key="k",
        base_url="https://example.test/v1",
        client_timeout=5,
        client_max_retries=0,
    )

    assert binding.client_timeout == 5.0
    assert binding.client_max_retries == 0


def test_presets_pass_client_options_through():
    binding = deepseek(api_key="k", client_timeout=5, client_max_retries=0)

    assert binding.client_timeout == 5.0
    assert binding.client_max_retries == 0
