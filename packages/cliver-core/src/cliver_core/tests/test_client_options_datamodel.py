"""One data object for every HTTP client setting — pool limits, UA, timeout, retries.

``ClientOptions`` is the single configuration surface: a binding carries one,
the built provider derives both its SDK client (``httpx2``) and its shared
raw-HTTP client (``httpx``) from it.  The flat ``client_timeout`` /
``client_max_retries`` factory arguments keep working and merge into the same
object; setting both spellings to different values is a configuration error.
"""

from __future__ import annotations

import pytest

from cliver_core import ClientOptions
from cliver_core.bindings import ModelBinding
from cliver_core.messages import CLIverMessage, CLIverMessageChunk
from cliver_core.presets import deepseek
from cliver_core.provider import CLIverRequest, CLIverResponse, Provider
from cliver_core.provider.providers import create_provider
from cliver_core.provider.providers.system_one import SystemOneProvider


class _PlainProvider(Provider):
    """A custom provider class that knows nothing about client options."""

    protocol = "openai"

    def __init__(self, api_key: str, base_url: str, **kwargs) -> None:
        super().__init__(api_key, base_url)

    async def chat(self, request: CLIverRequest) -> CLIverResponse:  # pragma: no cover - not called
        return CLIverResponse(message=CLIverMessage(role="assistant", content="ok"))

    async def stream(self, request: CLIverRequest):  # pragma: no cover - not called
        if False:
            yield CLIverMessageChunk()


class _StrictProvider(Provider):
    """A custom provider class whose constructor takes no client options at all."""

    protocol = "openai"

    def __init__(self, api_key: str, base_url: str) -> None:
        super().__init__(api_key, base_url)

    async def chat(self, request: CLIverRequest) -> CLIverResponse:  # pragma: no cover - not called
        return CLIverResponse(message=CLIverMessage(role="assistant", content="ok"))

    async def stream(self, request: CLIverRequest):  # pragma: no cover - not called
        if False:
            yield CLIverMessageChunk()


# ── The data object itself ────────────────────────────────────


def test_client_options_defaults_are_all_none() -> None:
    options = ClientOptions()

    assert options.user_agent is None
    assert options.timeout is None
    assert options.max_retries is None
    assert options.max_connections is None
    assert options.max_keepalive_connections is None
    assert options.keepalive_expiry is None


def test_client_options_normalizes_timeout_and_keepalive_expiry_to_float() -> None:
    options = ClientOptions(timeout=5, keepalive_expiry=2)

    assert options.timeout == 5.0
    assert options.keepalive_expiry == 2.0


def test_client_options_is_frozen() -> None:
    options = ClientOptions(timeout=5)

    with pytest.raises(AttributeError):
        options.timeout = 6  # type: ignore[misc]


@pytest.mark.parametrize("value", [0, -1, True, float("nan"), float("inf"), "5"])
def test_invalid_timeout_is_rejected(value) -> None:
    with pytest.raises(ValueError, match="timeout"):
        ClientOptions(timeout=value)


@pytest.mark.parametrize("value", [-1, True, 1.5, "2"])
def test_invalid_max_retries_is_rejected(value) -> None:
    with pytest.raises(ValueError, match="max_retries"):
        ClientOptions(max_retries=value)


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "8"])
def test_invalid_max_connections_is_rejected(value) -> None:
    with pytest.raises(ValueError, match="max_connections"):
        ClientOptions(max_connections=value)


@pytest.mark.parametrize("value", [-1, True, 1.5, "4"])
def test_invalid_max_keepalive_connections_is_rejected(value) -> None:
    with pytest.raises(ValueError, match="max_keepalive_connections"):
        ClientOptions(max_keepalive_connections=value)


@pytest.mark.parametrize("value", [0, -1, True, float("nan"), "2"])
def test_invalid_keepalive_expiry_is_rejected(value) -> None:
    with pytest.raises(ValueError, match="keepalive_expiry"):
        ClientOptions(keepalive_expiry=value)


@pytest.mark.parametrize("value", ["", "   ", 5])
def test_invalid_user_agent_is_rejected(value) -> None:
    with pytest.raises(ValueError, match="user_agent"):
        ClientOptions(user_agent=value)


def test_keepalive_above_max_connections_is_rejected() -> None:
    with pytest.raises(ValueError, match="max_keepalive_connections"):
        ClientOptions(max_connections=4, max_keepalive_connections=8)


def test_keepalive_above_the_httpx_default_pool_is_rejected() -> None:
    # httpx defaults max_connections to 100; a larger keep-alive pool can
    # never be used and httpx would raise only at client construction.
    with pytest.raises(ValueError, match="max_keepalive_connections"):
        ClientOptions(max_keepalive_connections=200)


# ── Bindings carry the data object ───────────────────────────


def test_binding_carries_the_client_options_object() -> None:
    options = ClientOptions(timeout=7.5, max_retries=0, max_connections=16)

    binding = ModelBinding.llm(model="m", api_key="k", base_url="https://example.test/v1", client=options)

    assert binding.client is options
    assert binding.client_options is options
    assert binding.client_timeout == 7.5
    assert binding.client_max_retries == 0


def test_flat_options_merge_into_the_data_object() -> None:
    binding = ModelBinding.llm(
        model="m",
        api_key="k",
        base_url="https://example.test/v1",
        client=ClientOptions(max_connections=16),
        client_timeout=5,
        client_max_retries=0,
    )

    assert binding.client_options.timeout == 5.0
    assert binding.client_options.max_retries == 0
    assert binding.client_options.max_connections == 16
    assert binding.client_timeout == 5.0
    assert binding.client_max_retries == 0


def test_flat_options_matching_the_data_object_are_accepted() -> None:
    # The clone path (for_current_loop → dataclasses.replace) re-runs
    # __post_init__ with mirrored flat fields; identical values must merge.
    binding = ModelBinding.llm(
        model="m",
        api_key="k",
        base_url="https://example.test/v1",
        client=ClientOptions(timeout=5),
        client_timeout=5,
    )

    assert binding.client_options.timeout == 5.0


def test_conflicting_flat_and_datamodel_options_are_rejected() -> None:
    with pytest.raises(ValueError, match="client_timeout"):
        ModelBinding.llm(
            model="m",
            api_key="k",
            base_url="https://example.test/v1",
            client=ClientOptions(timeout=5),
            client_timeout=7,
        )


def test_client_options_rejected_with_a_supplied_provider() -> None:
    supplied = _PlainProvider("k", "https://example.test/v1")

    with pytest.raises(ValueError, match="client"):
        ModelBinding.llm(model="m", provider=supplied, client=ClientOptions(max_connections=4))


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
def test_every_purpose_factory_accepts_the_client_options_object(factory) -> None:
    binding = factory(
        model="m",
        api_key="k",
        base_url="https://example.test/v1",
        client=ClientOptions(timeout=5, max_retries=0),
    )

    assert binding.client_options.timeout == 5.0
    assert binding.client_options.max_retries == 0
    assert binding.client_timeout == 5.0
    assert binding.client_max_retries == 0


def test_presets_pass_the_client_options_object_through() -> None:
    options = ClientOptions(max_connections=8)
    binding = deepseek(api_key="k", client=options)

    assert binding.client is options
    assert binding.client_options.max_connections == 8


# ── The SDK clients the provider builds ──────────────────────


@pytest.mark.asyncio
async def test_openai_sdk_client_gets_pool_limits() -> None:
    binding = ModelBinding.llm(
        model="m",
        api_key="k",
        base_url="https://example.test/v1",
        client=ClientOptions(max_connections=3, max_keepalive_connections=2, keepalive_expiry=1.5),
    )

    http_client = binding.build_provider().transport._client._client
    pool = http_client._transport._pool

    assert pool._max_connections == 3
    assert pool._max_keepalive_connections == 2
    assert pool._keepalive_expiry == 1.5
    await binding.aclose()


@pytest.mark.asyncio
async def test_anthropic_sdk_client_gets_pool_limits() -> None:
    binding = ModelBinding.llm(
        model="m",
        api_key="k",
        base_url="https://example.test",
        protocol="anthropic",
        client=ClientOptions(max_connections=5, max_keepalive_connections=5),
    )

    pool = binding.build_provider().transport._client._client._transport._pool

    assert pool._max_connections == 5
    assert pool._max_keepalive_connections == 5
    await binding.aclose()


@pytest.mark.asyncio
async def test_user_agent_reaches_the_sdk_client() -> None:
    binding = ModelBinding.llm(
        model="m",
        api_key="k",
        base_url="https://example.test/v1",
        client=ClientOptions(user_agent="cliver-test/1.0"),
    )

    client = binding.build_provider().transport._client

    assert client._custom_headers["User-Agent"] == "cliver-test/1.0"
    await binding.aclose()


@pytest.mark.asyncio
async def test_create_provider_accepts_the_client_options_object() -> None:
    provider = create_provider(
        api_key="k",
        base_url="https://example.test/v1",
        client=ClientOptions(timeout=2.5, max_retries=0, max_connections=3),
    )

    transport_client = provider.transport._client
    assert transport_client.timeout.read == 2.5
    assert transport_client.max_retries == 0
    assert transport_client._client._transport._pool._max_connections == 3
    await provider.close()


def test_client_options_with_an_sdk_timeout_rejected_without_an_sdk_transport() -> None:
    with pytest.raises(ValueError, match="timeout"):
        create_provider(
            api_key="k",
            base_url="https://example.test",
            provider_class=SystemOneProvider,
            client=ClientOptions(timeout=5),
        )


def test_pool_options_rejected_for_a_class_that_takes_no_client_options() -> None:
    with pytest.raises(ValueError, match="_StrictProvider"):
        create_provider(
            api_key="k",
            base_url="https://example.test",
            provider_class=_StrictProvider,
            client=ClientOptions(max_connections=4),
        )
