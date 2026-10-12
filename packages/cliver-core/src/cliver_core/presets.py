"""Ready-made :class:`~cliver_core.bindings.ModelBinding` factories for well-known providers.

Each factory returns a **fresh** binding on every call — a binding owns the
provider it builds, and :meth:`~cliver_core.bindings.ModelBinding.aclose`
closes it; one shared binding across agents would let one agent close a
provider another is still streaming on.  The API key comes from the brand's
conventional environment variable(s) (:data:`API_KEY_ENV`) when set — read at
**call time**, after dotenv/config load, never at import — and is the empty
string otherwise; the first
:meth:`~cliver_core.bindings.ModelBinding.get_provider` then fails with a
message naming the exact variable to set.

``provider_name`` and ``protocol`` are pinned, and ``base_url`` stays empty so
the provider registry's ``default_base_url`` is the single source of truth for
each brand's endpoint.  Media presets likewise pin the endpoint ``path``
explicitly — ``minimax_image`` / ``minimax_tts`` / ``minimax_asr`` name
``/image_generation`` / ``/t2a_v2`` / ``/speech_to_text`` and ``seedance``
names the Ark task endpoint ``/contents/generations/tasks`` on the binding
instead of falling back to the provider's brand map (pass ``path=None`` to
defer to the brand map, or another path to override).
``**params`` become the binding's default call options
and fold through the matching per-purpose
:class:`~cliver_core.bindings.ModelBinding` factory (e.g. ``minimax_tts``
accepts ``voice=`` / ``response_format=``; ``minimax_asr`` accepts
``language=`` — sent as an HTTP header on MiniMax's speech-to-text — and
``response_format=``).  Every factory also accepts an
explicit ``api_key`` (wins over the environment, including ``""``) and a
``base_url`` override — for MiniMax media that is also how you reach the
other platform (``api.minimax.io``).

Usage::

    import cliver_core.presets as presets

    agent = AgentCore(ModelBindings(llm=presets.deepseek_anthropic()))
    ...
"""

from __future__ import annotations

import os
from typing import Any

from cliver_core.bindings import ModelBinding
from cliver_core.provider.providers import API_KEY_ENV

__all__ = [
    "deepseek",
    "deepseek_anthropic",
    "glm",
    "glm_anthropic",
    "minimax_anthropic",
    "minimax_asr",
    "minimax_image",
    "minimax_tts",
    "seedance",
]


def _api_key(api_key: str | None, brand: str) -> str:
    """Explicit *api_key* wins (including ``""``); else the brand's env vars, read now."""
    if api_key is not None:
        return api_key
    for env_var in API_KEY_ENV.get(brand, ()):
        if os.environ.get(env_var):
            return os.environ[env_var]
    return ""


def deepseek(
    *,
    api_key: str | None = None,
    model: str = "deepseek-v4-flash",
    base_url: str = "",
    **params: Any,
) -> ModelBinding:
    """DeepSeek chat over its OpenAI-compatible endpoint."""
    return ModelBinding.llm(
        model=model,
        api_key=_api_key(api_key, "deepseek"),
        base_url=base_url,
        protocol="openai",
        provider_name="deepseek",
        **params,
    )


def deepseek_anthropic(
    *, api_key: str | None = None, model: str = "deepseek-v4-flash", base_url: str = "", **params: Any
) -> ModelBinding:
    """DeepSeek chat over its Anthropic-compatible endpoint."""
    return ModelBinding.llm(
        model=model,
        api_key=_api_key(api_key, "deepseek"),
        base_url=base_url,
        protocol="anthropic",
        provider_name="deepseek",
        **params,
    )


def minimax_anthropic(
    *, api_key: str | None = None, model: str = "MiniMax-M3", base_url: str = "", **params: Any
) -> ModelBinding:
    """MiniMax chat — the brand's only chat preset (its Anthropic-compatible endpoint)."""
    return ModelBinding.llm(
        model=model,
        api_key=_api_key(api_key, "minimax"),
        base_url=base_url,
        protocol="anthropic",
        provider_name="minimax",
        **params,
    )


def glm(*, api_key: str | None = None, model: str = "glm-5.3", base_url: str = "", **params: Any) -> ModelBinding:
    """GLM chat over Zhipu's OpenAI-compatible endpoint."""
    return ModelBinding.llm(
        model=model,
        api_key=_api_key(api_key, "glm"),
        base_url=base_url,
        protocol="openai",
        provider_name="glm",
        **params,
    )


def glm_anthropic(
    *,
    api_key: str | None = None,
    model: str = "glm-5.3",
    base_url: str = "",
    **params: Any,
) -> ModelBinding:
    """GLM chat over Zhipu's Anthropic-compatible endpoint."""
    return ModelBinding.llm(
        model=model,
        api_key=_api_key(api_key, "glm"),
        base_url=base_url,
        protocol="anthropic",
        provider_name="glm",
        **params,
    )


def minimax_image(
    *,
    api_key: str | None = None,
    model: str = "image-01",
    base_url: str = "",
    path: str | None = "/image_generation",
    **params: Any,
) -> ModelBinding:
    """MiniMax image generation — the endpoint path (``/image_generation``) is
    pinned explicitly on the binding; ``path=None`` defers to the provider's
    brand map.

    Media posts ride the brand's chat leaf (its only provider class); a
    ``base_url`` override maps ``…/anthropic`` to the sibling ``…/v1``
    media root.
    """
    return ModelBinding.image(
        model=model,
        api_key=_api_key(api_key, "minimax"),
        base_url=base_url,
        protocol="anthropic",
        provider_name="minimax",
        path=path,
        **params,
    )


def minimax_tts(
    *,
    api_key: str | None = None,
    model: str = "speech-2.8-hd",
    base_url: str = "",
    path: str | None = "/t2a_v2",
    **params: Any,
) -> ModelBinding:
    """MiniMax speech synthesis — the endpoint path (``/t2a_v2``) is pinned
    explicitly on the binding (``path=None`` defers to the provider's brand
    map); ``voice`` / ``response_format`` are default params.

    Media posts ride the brand's chat leaf (its only provider class); a
    ``base_url`` override maps ``…/anthropic`` to the sibling ``…/v1``
    media root.
    """
    return ModelBinding.tts(
        model=model,
        api_key=_api_key(api_key, "minimax"),
        base_url=base_url,
        protocol="anthropic",
        provider_name="minimax",
        path=path,
        **params,
    )


def minimax_asr(
    *,
    api_key: str | None = None,
    model: str = "asr-1.0",
    base_url: str = "",
    path: str | None = "/speech_to_text",
    **params: Any,
) -> ModelBinding:
    """MiniMax speech-to-text — the endpoint path (``/speech_to_text``) is
    pinned explicitly on the binding (``path=None`` defers to the provider's
    brand map).

    Default params: ``language`` is a BCP-47 hint (``en``, ``zh``, ``yue``,
    …) sent as an HTTP header — omit it for mixed-language recognition;
    ``response_format`` selects ``json`` / ``verbose_json`` (transcript in
    ``.message.content``) or ``srt`` / ``vtt`` (timed text as plain text);
    ``timestamp_level`` (``sentence`` / ``word``) applies to the timed
    formats only.

    Media posts ride the brand's chat leaf (its only provider class); a
    ``base_url`` override maps ``…/anthropic`` to the sibling ``…/v1``
    media root — ``https://api.minimax.cn/anthropic`` reaches the CN
    platform (the brand default is the international ``api.minimaxi.com``).
    """
    return ModelBinding.asr(
        model=model,
        api_key=_api_key(api_key, "minimax"),
        base_url=base_url,
        protocol="anthropic",
        provider_name="minimax",
        path=path,
        **params,
    )


def seedance(
    *,
    api_key: str | None = None,
    model: str = "doubao-seedance-2-5-260628",
    base_url: str = "",
    path: str | None = "/contents/generations/tasks",
    **params: Any,
) -> ModelBinding:
    """Seedance (Volcano Engine Ark) video generation — the endpoint path
    (``/contents/generations/tasks``) is pinned explicitly on the binding
    (``path=None`` defers to the provider's brand map).

    Generation is asynchronous: the create call answers a task id and the
    provider polls until the video is ready.  Default params carry the Ark
    request fields (``duration`` 4–30 s, ``resolution`` 480p/720p/1080p,
    ``ratio``, ``watermark``, ``seed``, …) plus the poll knobs
    ``poll_interval`` (seconds between task reads, default 5) and
    ``poll_timeout`` (seconds to wait for the result, default 600;
    ``None`` waits without a deadline).  A ``base_url`` override retargets
    the endpoint (proxy or another Ark region).
    """
    return ModelBinding.video(
        model=model,
        api_key=_api_key(api_key, "seedance"),
        base_url=base_url,
        protocol="openai",
        provider_name="seedance",
        path=path,
        **params,
    )
