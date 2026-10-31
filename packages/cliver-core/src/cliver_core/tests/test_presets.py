"""Ready-made binding presets — wiring, env-at-call-time, freshness, overrides.

No network: provider classes are asserted as types; the couple of tests that
materialize a provider never call it (real SDK clients construct offline).
API keys here are literal dummies — the suite never reads key environment
variables for real values.
"""

from __future__ import annotations

import pytest

from cliver_core import presets
from cliver_core.bindings import ModelBindings
from cliver_core.provider.providers import API_KEY_ENV
from cliver_core.provider.providers.dashscope import DashScopeProvider
from cliver_core.provider.providers.deepseek import DeepSeekAnthropicProvider, DeepSeekOpenAIProvider
from cliver_core.provider.providers.glm import GLMAnthropicProvider, GLMProvider
from cliver_core.provider.providers.minimax import MiniMaxAnthropicProvider
from cliver_core.provider.providers.qwen import QwenProvider
from cliver_core.provider.providers.qwen_image import QwenImageProvider
from cliver_core.provider.providers.seedance import SeedanceProvider

# factory, purpose, protocol, brand, default model, provider class, default base_url
CASES = [
    (
        presets.deepseek,
        "llm",
        "openai",
        "deepseek",
        "deepseek-v4-flash",
        DeepSeekOpenAIProvider,
        "https://api.deepseek.com/v1",
    ),
    (
        presets.deepseek_anthropic,
        "llm",
        "anthropic",
        "deepseek",
        "deepseek-v4-flash",
        DeepSeekAnthropicProvider,
        "https://api.deepseek.com/anthropic",
    ),
    (
        presets.minimax_anthropic,
        "llm",
        "anthropic",
        "minimax",
        "MiniMax-M3",
        MiniMaxAnthropicProvider,
        "https://api.minimaxi.com/anthropic",
    ),
    (
        presets.glm,
        "llm",
        "openai",
        "glm",
        "glm-5.3",
        GLMProvider,
        "https://open.bigmodel.cn/api/paas/v4",
    ),
    (
        presets.glm_anthropic,
        "llm",
        "anthropic",
        "glm",
        "glm-5.3",
        GLMAnthropicProvider,
        "https://open.bigmodel.cn/api/anthropic",
    ),
    (
        presets.minimax_image,
        "image",
        "anthropic",
        "minimax",
        "image-01",
        MiniMaxAnthropicProvider,
        "https://api.minimaxi.com/anthropic",
    ),
    (
        presets.minimax_tts,
        "tts",
        "anthropic",
        "minimax",
        "speech-2.8-hd",
        MiniMaxAnthropicProvider,
        "https://api.minimaxi.com/anthropic",
    ),
    (
        presets.minimax_asr,
        "asr",
        "anthropic",
        "minimax",
        "asr-1.0",
        MiniMaxAnthropicProvider,
        "https://api.minimaxi.com/anthropic",
    ),
    (
        presets.seedance,
        "video",
        "openai",
        "seedance",
        "doubao-seedance-2-5-260628",
        SeedanceProvider,
        "https://ark.cn-beijing.volces.com/api/v3",
    ),
    (
        presets.qwen,
        "llm",
        "openai",
        "qwen",
        "qwen-plus",
        QwenProvider,
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    ),
    (
        presets.dashscope_image,
        "image",
        "openai",
        "dashscope",
        "wanx2.1-t2i-turbo",
        DashScopeProvider,
        "https://dashscope.aliyuncs.com/api/v1",
    ),
    (
        presets.qwen_image,
        "image",
        "openai",
        "qwen-image",
        "qwen-image-3.0",
        QwenImageProvider,
        "https://dashscope.aliyuncs.com/api/v1",
    ),
]

_IDS = [case[0].__name__ for case in CASES]
_CASE = pytest.mark.parametrize("factory,purpose,protocol,brand,model,provider_cls,base_url", CASES, ids=_IDS)


def _clear_key_env(monkeypatch, brand: str) -> None:
    for env_var in API_KEY_ENV.get(brand, ()):
        monkeypatch.delenv(env_var, raising=False)


@_CASE
def test_factory_wiring(factory, purpose, protocol, brand, model, provider_cls, base_url) -> None:
    binding = factory()
    assert binding.model == model
    assert binding.purpose == purpose
    assert binding.protocol == protocol
    assert binding.provider_name == brand
    # base_url stays empty — the registry class default is the single source of truth
    assert binding.base_url == ""


@_CASE
def test_key_env_read_at_call_time_not_import(
    factory, purpose, protocol, brand, model, provider_cls, base_url, monkeypatch
) -> None:
    _clear_key_env(monkeypatch, brand)
    assert factory().api_key == ""  # unset → empty string, no error yet

    monkeypatch.setenv(API_KEY_ENV[brand][0], "env-key")
    assert factory().api_key == "env-key"  # read NOW — a fresh call sees it


def test_glm_falls_back_to_its_second_env_var(monkeypatch) -> None:
    monkeypatch.delenv(API_KEY_ENV["glm"][0], raising=False)
    monkeypatch.setenv(API_KEY_ENV["glm"][1], "zhipu-key")
    assert presets.glm().api_key == "zhipu-key"


@_CASE
def test_explicit_key_wins_including_empty_string(
    factory, purpose, protocol, brand, model, provider_cls, base_url, monkeypatch
) -> None:
    monkeypatch.setenv(API_KEY_ENV[brand][0], "env-key")
    assert factory(api_key="explicit").api_key == "explicit"
    assert factory(api_key="").api_key == ""  # explicit "" opts out of the env


@_CASE
def test_empty_key_provider_raises_naming_the_env_var(
    factory, purpose, protocol, brand, model, provider_cls, base_url, monkeypatch
) -> None:
    _clear_key_env(monkeypatch, brand)
    with pytest.raises(ValueError, match=API_KEY_ENV[brand][0]):
        factory().get_provider()


@_CASE
@pytest.mark.asyncio
async def test_provider_class_and_registry_default_url(
    factory, purpose, protocol, brand, model, provider_cls, base_url
) -> None:
    provider = factory(api_key="k").get_provider()
    assert type(provider) is provider_cls
    assert provider.provider_name() == brand
    assert provider.protocol == protocol
    assert provider.base_url == base_url
    await provider.close()


@_CASE
def test_overrides_fold_through(factory, purpose, protocol, brand, model, provider_cls, base_url) -> None:
    binding = factory(model="custom-model", base_url="https://proxy.test/v1")
    assert binding.model == "custom-model"
    assert binding.base_url == "https://proxy.test/v1"

    # **params become default call options via the matching per-purpose factory
    assert factory(temperature=0.3).params == {"temperature": 0.3}


def test_params_use_the_purpose_factories_named_knobs() -> None:
    assert presets.minimax_tts(voice="English_Trustworth_Man", response_format="wav").params == {
        "voice": "English_Trustworth_Man",
        "response_format": "wav",
    }
    assert presets.minimax_image(size="1024x1024", n=2).params == {"size": "1024x1024", "n": 2}
    assert presets.minimax_asr(language="zh", response_format="verbose_json").params == {
        "language": "zh",
        "response_format": "verbose_json",
    }


@pytest.mark.asyncio
async def test_each_call_is_a_fresh_binding_with_its_own_provider() -> None:
    a = presets.deepseek(api_key="k")
    b = presets.deepseek(api_key="k")
    assert a is not b

    pa = a.get_provider()
    pb = b.get_provider()
    assert pa is not pb

    # closing one agent's binding must not disturb another's provider
    await a.aclose()
    assert a.get_provider() is not pa  # closed and dropped — rebuilt on demand
    assert b.get_provider() is pb  # b's provider untouched
    await a.aclose()
    await b.aclose()


def test_presets_mount_in_model_bindings() -> None:
    bindings = ModelBindings(llm=presets.glm_anthropic(api_key="k"), tts=presets.minimax_tts(api_key="k"))
    assert bindings.get("llm") is bindings.llm
    assert bindings.get("tts") is bindings.tts

    legacy_audio = ModelBindings(audio=presets.minimax_tts(api_key="k"))
    assert legacy_audio.get("audio") is not None


@pytest.mark.asyncio
async def test_minimax_tts_preset_resolves_the_t2a_endpoint() -> None:
    binding = presets.minimax_tts(api_key="k")
    provider = binding.get_provider()
    assert provider.media_path("tts") == "/t2a_v2"
    assert provider._resolve_media_gen_url("tts") == "https://api.minimaxi.com/v1/t2a_v2"
    await binding.aclose()


@pytest.mark.asyncio
async def test_media_presets_pin_the_endpoint_path_explicitly() -> None:
    """The media presets name their endpoint path on the binding — never left
    to the provider's brand map.  Resolving with the binding's default path
    must land on the same absolute URL as the brand-map fallback."""
    cases = [
        (presets.minimax_image, "image", "/image_generation"),
        (presets.minimax_tts, "tts", "/t2a_v2"),
        (presets.minimax_asr, "asr", "/speech_to_text"),
    ]
    for factory, purpose, path in cases:
        binding = factory(api_key="k")
        assert binding.call_defaults()["path"] == path
        provider = binding.get_provider()
        via_binding = provider._resolve_media_gen_url(purpose, binding.call_defaults()["path"])
        via_brand_map = provider._resolve_media_gen_url(purpose)
        assert via_binding == via_brand_map == f"https://api.minimaxi.com/v1{path}"
        await binding.aclose()

    # path=None defers to the brand map: no path in the binding defaults
    assert "path" not in presets.minimax_tts(api_key="k", path=None).call_defaults()


@pytest.mark.asyncio
async def test_seedance_preset_resolves_the_ark_task_endpoint() -> None:
    """The seedance preset pins the Ark task path; the binding default and the
    brand-map fallback resolve to the same absolute URL.  The poll knobs ride
    ``**params`` like any other default call option."""
    binding = presets.seedance(api_key="k")
    assert binding.call_defaults()["path"] == "/contents/generations/tasks"
    provider = binding.get_provider()
    via_binding = provider._resolve_media_gen_url("video", binding.call_defaults()["path"])
    via_brand_map = provider._resolve_media_gen_url("video")
    assert via_binding == via_brand_map == "https://ark.cn-beijing.volces.com/api/v3/contents/generations/tasks"
    await binding.aclose()

    assert presets.seedance(poll_interval=2, poll_timeout=60).params == {"poll_interval": 2, "poll_timeout": 60}
    assert "path" not in presets.seedance(api_key="k", path=None).call_defaults()


def test_all_presets_are_exported() -> None:
    assert set(presets.__all__) == {
        "bailian_system_one",
        "dashscope_background",
        "dashscope_image",
        "dashscope_video",
        "deepseek",
        "deepseek_anthropic",
        "glm",
        "glm_anthropic",
        "minimax_anthropic",
        "minimax_asr",
        "minimax_image",
        "minimax_tts",
        "qwen",
        "qwen_image",
        "seedance",
        "typesafe_system_one",
    }
    import cliver_core

    assert cliver_core.presets is presets
