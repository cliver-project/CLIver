"""Registry matrix: brand × protocol → protocol-locked provider class.

Every provider class is protocol-locked (``protocol`` + ``brand`` class
attrs); dual-endpoint brands (DeepSeek, GLM) split into two classes (each
leaf or sharing a mixin base).  ``resolve_provider_class`` resolves
``_BRAND_PROTOCOL_FACTORIES[brand][protocol]`` for an exact, case-insensitive
``provider_name`` and falls back to the protocol-default class for missing
combos (e.g. ollama + anthropic).  Default endpoints come only from an exact
registry hit or an explicit ``provider_class`` — never from a URL.
"""

from __future__ import annotations

import pytest

from cliver_core.provider.providers import create_provider, resolve_provider_class


def test_minimax_registry_matrix_and_default_urls() -> None:
    from cliver_core.provider.providers.minimax import MiniMaxAnthropicProvider
    from cliver_core.provider.providers.openai import OpenAIProvider

    assert resolve_provider_class(provider_name="minimax", protocol="anthropic") is MiniMaxAnthropicProvider
    # one scalar default URL — anthropic-shaped chat base (media maps to /v1)
    assert MiniMaxAnthropicProvider.default_base_url == "https://api.minimaxi.com/anthropic"
    # brand comes from the mixin — NOT name-derived ("minimaxanthropic" trap)
    assert MiniMaxAnthropicProvider.provider_name() == "minimax"
    # MiniMax chat is anthropic-endpoint only (the openai leaf was dropped,
    # 2026-10-09): the combo takes the documented protocol-default fallback
    # (and, having no registry entry, no default endpoint)
    assert resolve_provider_class(provider_name="minimax", protocol="openai") is OpenAIProvider


def test_seedance_registry_matrix_and_default_urls() -> None:
    from cliver_core.provider.providers.anthropic import AnthropicProvider
    from cliver_core.provider.providers.seedance import SeedanceProvider

    assert resolve_provider_class(provider_name="seedance", protocol="openai") is SeedanceProvider
    assert SeedanceProvider.default_base_url == "https://ark.cn-beijing.volces.com/api/v3"
    assert SeedanceProvider.provider_name() == "seedance"
    # video is the only media surface (the async-task wire); the anthropic
    # combo has no leaf and takes the documented protocol-default fallback
    assert resolve_provider_class(provider_name="seedance", protocol="anthropic") is AnthropicProvider


def test_dashscope_and_qwen_registry_matrix() -> None:
    from cliver_core.provider.providers.anthropic import AnthropicProvider
    from cliver_core.provider.providers.dashscope import DashScopeProvider
    from cliver_core.provider.providers.qwen import QwenProvider
    from cliver_core.provider.providers.qwen_image import QwenImageProvider

    # native media tasks (openai-protocol leaf); the anthropic combo takes
    # the documented protocol-default fallback
    assert resolve_provider_class(provider_name="dashscope", protocol="openai") is DashScopeProvider
    assert DashScopeProvider.default_base_url == "https://dashscope.aliyuncs.com/api/v1"
    assert DashScopeProvider.provider_name() == "dashscope"
    assert resolve_provider_class(provider_name="dashscope", protocol="anthropic") is AnthropicProvider

    # chat rides the sibling OpenAI-compatible endpoint, a separate brand
    assert resolve_provider_class(provider_name="qwen", protocol="openai") is QwenProvider
    assert QwenProvider.default_base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert QwenProvider.provider_name() == "qwen"

    # reference-aware image generation is its own brand on the native API root
    assert resolve_provider_class(provider_name="qwen-image", protocol="openai") is QwenImageProvider
    assert QwenImageProvider.default_base_url == "https://dashscope.aliyuncs.com/api/v1"
    assert QwenImageProvider.provider_name() == "qwen-image"


def test_deepseek_registry_matrix() -> None:
    from cliver_core.provider.providers.deepseek import DeepSeekAnthropicProvider, DeepSeekOpenAIProvider

    assert resolve_provider_class(provider_name="deepseek", protocol="openai") is DeepSeekOpenAIProvider
    assert resolve_provider_class(provider_name="deepseek", protocol="anthropic") is DeepSeekAnthropicProvider
    assert DeepSeekOpenAIProvider.default_base_url == "https://api.deepseek.com/v1"
    assert DeepSeekAnthropicProvider.default_base_url == "https://api.deepseek.com/anthropic"


def test_glm_registry_matrix_and_default_urls() -> None:
    from cliver_core.provider.providers.glm import GLMAnthropicProvider, GLMProvider

    assert resolve_provider_class(provider_name="glm", protocol="openai") is GLMProvider
    assert resolve_provider_class(provider_name="glm", protocol="anthropic") is GLMAnthropicProvider
    assert GLMProvider.default_base_url == "https://open.bigmodel.cn/api/paas/v4"
    assert GLMAnthropicProvider.default_base_url == "https://open.bigmodel.cn/api/anthropic"
    # the Anthropic-compatible endpoint takes Bearer auth (x-api-key also works)
    assert GLMAnthropicProvider._anthropic_use_bearer_auth is True
    # brand comes from the class attr — not derived from the class name
    assert GLMAnthropicProvider.provider_name() == "glm"


def test_glm_does_not_filter_request_options() -> None:
    # No brand parameter constraints on either endpoint: the OpenAI-style
    # penalty params pass through untouched.  Live-verified 2026-10-09:
    # both endpoints accept them (glm-5.3 on /anthropic, free models on
    # paas/v4) — the earlier filter was written against an unverified
    # claim that GLM rejects them.
    from cliver_core.provider.providers.glm import GLMAnthropicProvider, GLMProvider

    options = {
        "frequency_penalty": 0.5,
        "presence_penalty": 0.5,
        "logit_bias": {"1": 1},
        "temperature": 0.7,
    }
    assert GLMProvider(api_key="k", base_url="https://example.test/v1").filter_options(options) == options
    assert (
        GLMAnthropicProvider(api_key="k", base_url="https://example.test/anthropic").filter_options(options) == options
    )


def test_exact_provider_name_is_case_insensitive() -> None:
    from cliver_core.provider.providers.deepseek import DeepSeekOpenAIProvider

    assert resolve_provider_class(provider_name="  DeepSeek ", protocol="openai") is DeepSeekOpenAIProvider


def test_unknown_or_absent_brand_selects_protocol_default_class() -> None:
    # No name, unknown names, and the dropped "zhipu"/"bigmodel" aliases all
    # select the protocol-default class — nothing about a URL changes that.
    from cliver_core.provider.providers.anthropic import AnthropicProvider
    from cliver_core.provider.providers.openai import OpenAIProvider

    assert resolve_provider_class(provider_name="my-deepseek", protocol="openai") is OpenAIProvider
    assert resolve_provider_class(provider_name="openai-compatible", protocol="openai") is OpenAIProvider
    assert resolve_provider_class(provider_name="zhipu", protocol="openai") is OpenAIProvider
    assert resolve_provider_class(provider_name="wat", protocol="openai") is OpenAIProvider
    assert resolve_provider_class(protocol="openai") is OpenAIProvider
    assert resolve_provider_class(protocol="anthropic") is AnthropicProvider


def test_missing_combo_falls_back_to_protocol_default() -> None:
    from cliver_core.provider.providers.anthropic import AnthropicProvider

    # ollama has no anthropic endpoint → the protocol-default class, silently
    # (same swap as before the protocol lock)
    assert resolve_provider_class(provider_name="ollama", protocol="anthropic") is AnthropicProvider


def test_unknown_protocol_raises_instead_of_defaulting() -> None:
    with pytest.raises(ValueError, match="Unknown protocol 'mqtt'"):
        resolve_provider_class(provider_name="deepseek", protocol="mqtt")


def test_leaf_without_protocol_raises_at_class_definition() -> None:
    from cliver_core.provider import Provider

    with pytest.raises(TypeError, match="must define protocol"):

        class _ForgotProtocol(Provider):
            brand = "forgot"


def test_mixin_base_is_an_explicit_opt_out() -> None:
    # declaring protocol = "" is the documented intermediate marker —
    # defining the mixin itself must not raise
    from cliver_core.provider import Provider

    class _AnotherMixin(Provider):
        protocol = ""  # noqa: RUF012 — intermediate opt-out marker

        def filter_options(self, options: dict) -> dict:
            return options


# ── create_provider — defaults only from explicit input ────────


def test_create_provider_requires_base_url_without_an_exact_brand_hit() -> None:
    for kwargs in (
        {},  # no provider_name at all — was: silently api.openai.com
        {"provider_name": "my-deepseek-proxy"},  # substring match is gone
        {"provider_name": "zhipu"},  # alias dropped (exact "glm" only)
        {"provider_name": "minimax", "protocol": "openai"},  # no leaf for the combo
        {"provider_name": "seedance", "protocol": "anthropic"},
        {"provider_name": "ollama", "protocol": "anthropic"},
    ):
        with pytest.raises(ValueError, match="base_url"):
            create_provider(api_key="k", **kwargs)

    # an unknown protocol fails with its own, more specific message
    with pytest.raises(ValueError, match="Unknown protocol 'mqtt'"):
        create_provider(api_key="k", provider_name="deepseek", protocol="mqtt")


def test_no_base_url_error_teaches_the_open_door() -> None:
    # the no-URL error must point at the way out: an unlisted, wire-compatible
    # vendor needs no registry entry — base_url + protocol is all (D25)
    with pytest.raises(ValueError, match="compatible endpoint works; no registry entry needed"):
        create_provider(api_key="k")


@pytest.mark.asyncio
async def test_create_provider_generic_class_with_explicit_base_url() -> None:
    from cliver_core.provider.providers.openai import OpenAIProvider

    provider = create_provider(api_key="k", base_url="https://vllm.internal:8000/v1", provider_name="my-vllm")
    assert isinstance(provider, OpenAIProvider)
    assert provider.base_url == "https://vllm.internal:8000/v1"
    await provider.close()


@pytest.mark.asyncio
async def test_create_provider_resolves_default_urls_and_transport() -> None:
    from cliver_core.provider.providers.minimax import MiniMaxAnthropicProvider

    provider = create_provider(api_key="k", protocol="anthropic", provider_name="minimax")
    assert isinstance(provider, MiniMaxAnthropicProvider)
    assert provider.base_url == "https://api.minimaxi.com/anthropic"
    assert provider.provider_name() == "minimax"
    assert provider.transport.label == "anthropic-sdk"
    await provider.close()


@pytest.mark.asyncio
async def test_create_provider_explicit_class_wins_and_uses_its_default_url() -> None:
    from cliver_core.provider import Provider
    from cliver_core.provider.providers.openai import OpenAIProvider

    class _NoDefault(Provider):
        protocol = "mock"
        brand = "nodefault"

    provider = create_provider(api_key="k", provider_class=OpenAIProvider)
    assert isinstance(provider, OpenAIProvider)
    assert provider.base_url == OpenAIProvider.default_base_url
    await provider.close()

    # an explicit class without a default endpoint still requires base_url
    with pytest.raises(ValueError, match="base_url"):
        create_provider(api_key="k", provider_class=_NoDefault)
