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


def qwen(*, api_key: str | None = None, model: str = "qwen-plus", base_url: str = "", **params: Any) -> ModelBinding:
    """Qwen chat over DashScope's OpenAI-compatible endpoint.

    Keys are region-bound: a key from another region (e.g. Singapore) needs
    an explicit ``base_url`` override.
    """
    return ModelBinding.llm(
        model=model,
        api_key=_api_key(api_key, "qwen"),
        base_url=base_url,
        protocol="openai",
        provider_name="qwen",
        **params,
    )


def qwen_image(
    *,
    api_key: str | None = None,
    model: str = "qwen-image-3.0",
    base_url: str = "",
    path: str | None = "/services/aigc/multimodal-generation/generation",
    **params: Any,
) -> ModelBinding:
    """Qwen-Image generation — reference-aware and synchronous.

    The endpoint path (``/services/aigc/multimodal-generation/generation``)
    is pinned explicitly on the binding (``path=None`` defers to the
    provider's brand map).  The prompt is the instruction and each
    ``media`` image is a reference shown to the model (qwen-image-3.0
    accepts 1–3; the prompt can address them as "image 1", "image 2", …).
    Local bytes (base64 / ``data:``) pass straight through — this wire
    accepts them directly, no upload — and public ``http(s)`` URLs pass
    through.  Options (``size``, ``n``, ``negative_prompt``,
    ``prompt_extend``, ``seed`` …) ride ``parameters``; sizes may be
    written either ``x`` or ``*`` separated.  The result is returned in the
    same request (no task polling).

    Keys are region-bound: a key from another region needs an explicit
    ``base_url`` override.
    """
    return ModelBinding.image(
        model=model,
        api_key=_api_key(api_key, "qwen-image"),
        base_url=base_url,
        protocol="openai",
        provider_name="qwen-image",
        path=path,
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


def dashscope_image(
    *,
    api_key: str | None = None,
    model: str = "wanx2.1-t2i-turbo",
    base_url: str = "",
    path: str | None = "/services/aigc/text2image/image-synthesis",
    **params: Any,
) -> ModelBinding:
    """DashScope text-to-image — the endpoint path
    (``/services/aigc/text2image/image-synthesis``) is pinned explicitly on
    the binding (``path=None`` defers to the provider's brand map).

    Generation is asynchronous: the create call answers a task id and the
    provider polls until the image is ready.  Default params carry DashScope
    request fields (``size`` written either ``x`` or ``*`` separated, ``n``,
    ``negative_prompt``, ``prompt_extend``, ``seed``, …) plus the poll knobs
    ``poll_interval`` (seconds between task reads, default 3) and
    ``poll_timeout`` (seconds to wait for the result, default 600; ``None``
    waits without a deadline).  ``media=[...]`` reference images are ignored
    by plain text-to-image models.
    """
    return ModelBinding.image(
        model=model,
        api_key=_api_key(api_key, "dashscope"),
        base_url=base_url,
        protocol="openai",
        provider_name="dashscope",
        path=path,
        **params,
    )


def dashscope_background(
    *,
    api_key: str | None = None,
    model: str = "wanx-background-generation-v2",
    base_url: str = "",
    path: str | None = "/services/aigc/background-generation/generation",
    model_version: str = "v3",
    n: int | None = 1,
    **params: Any,
) -> ModelBinding:
    """DashScope product background generation — the endpoint path
    (``/services/aigc/background-generation/generation``) is pinned
    explicitly on the binding (``path=None`` defers to the provider's brand
    map, which serves text-to-image).

    The prompt becomes the scene instruction (``input.ref_prompt``); the
    first ``media`` image is the product photo (``input.base_image_url``),
    a second is a style reference (``input.ref_image_url``).  Local bytes
    (base64 / ``data:``) are uploaded to DashScope temporary storage and
    sent as ``oss://`` URLs automatically — public ``http(s)`` URLs pass
    through.  ``model_version="v3"`` and ``n=1`` are the documented
    defaults; more parameters (``ref_prompt_weight``, ``noise_level``, …)
    ride the request's ``parameters``.  Poll knobs as on
    :func:`dashscope_image`.
    """
    return ModelBinding.image(
        model=model,
        api_key=_api_key(api_key, "dashscope"),
        base_url=base_url,
        protocol="openai",
        provider_name="dashscope",
        path=path,
        n=n,
        model_version=model_version,
        **params,
    )


def dashscope_video(
    *,
    api_key: str | None = None,
    model: str = "wanx2.1-t2v-turbo",
    base_url: str = "",
    path: str | None = "/services/aigc/video-generation/video-synthesis",
    duration: int | None = 5,
    size: str | None = "1280*720",
    **params: Any,
) -> ModelBinding:
    """DashScope text/image-to-video — the endpoint path
    (``/services/aigc/video-generation/video-synthesis``) is pinned
    explicitly on the binding (``path=None`` defers to the provider's brand
    map).

    The prompt is the motion/scene instruction (``input.prompt``); the first
    ``media`` image is the start frame (``input.img_url``, local bytes are
    uploaded to ``oss://`` automatically).  Defaults: ``duration=5``
    seconds and ``size="1280*720"`` (``x`` is accepted too).  Poll knobs as
    on :func:`dashscope_image`.
    """
    return ModelBinding.video(
        model=model,
        api_key=_api_key(api_key, "dashscope"),
        base_url=base_url,
        protocol="openai",
        provider_name="dashscope",
        path=path,
        duration=duration,
        size=size,
        **params,
    )
