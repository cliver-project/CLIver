"""Concrete Provider implementations and explicit brand selection.

Protocol providers (``providers.openai.OpenAIProvider`` /
``providers.anthropic.AnthropicProvider``) own one wire protocol each —
transport I/O, request assembly, response extraction.  Brand classes
subclass one protocol provider and add behavior via hook overrides.

Brands are selected, never guessed: an exact ``provider_name`` match
(case-insensitive) against ``_BRAND_PROTOCOL_FACTORIES`` plus the requested
protocol.  A brand's ``default_base_url`` applies only from such an exact
entry or an explicitly passed ``provider_class`` — otherwise ``base_url``
must be explicit.

Brand entries are convenience, never coverage: any endpoint that speaks
either protocol works with **no brand entry at all** — pass *protocol* and
an explicit *base_url* and the protocol-default class serves it.  The
registry is a curated, live-verified shortlist (default endpoints,
convention API-key env vars, verified wire quirks).  New wire protocols
arrive out-of-tree via the ``cliver.provider_sdks`` entry-point group;
private quirks via subclassing a provider and overriding its hooks.

**Brand admission rule.**  A brand entry must earn its place with either
*behavior* — a live-verified wire deviation (request body, headers,
polling, auth, response shape, parameter filtering) — or *identity* a
preset depends on (a default endpoint plus the conventional API-key
variable, so presets, the empty-key diagnostic, and provider events name
the right brand).  A pure endpoint + key pair with neither belongs in a
preset's explicit ``base_url``; aliases are never added.  A third body
shape within one brand family becomes its own brand class (as
``qwen-image`` did), never another ``model.startswith(...)`` branch.
Registry length is not a goal: it stays a curated shortlist, never a
coverage claim.
"""

from __future__ import annotations

import logging

from cliver_core.events import EventHandler
from cliver_core.provider import Provider

# ── Lazy imports for individual providers ────────────────────
# Each provider module defines a protocol-provider subclass
# (OpenAIProvider / AnthropicProvider).
# Using lazy imports here avoids circular dependencies since
# provider modules import from this __init__.py.


def _get_deepseek_openai():
    from cliver_core.provider.providers.deepseek import DeepSeekOpenAIProvider

    return DeepSeekOpenAIProvider


def _get_deepseek_anthropic():
    from cliver_core.provider.providers.deepseek import DeepSeekAnthropicProvider

    return DeepSeekAnthropicProvider


def _get_minimax_anthropic():
    from cliver_core.provider.providers.minimax import MiniMaxAnthropicProvider

    return MiniMaxAnthropicProvider


def _get_openai_provider():
    from cliver_core.provider.providers.openai import OpenAIProvider

    return OpenAIProvider


def _get_anthropic_provider():
    from cliver_core.provider.providers.anthropic import AnthropicProvider

    return AnthropicProvider


def _get_glm_provider():
    from cliver_core.provider.providers.glm import GLMProvider

    return GLMProvider


def _get_glm_anthropic():
    from cliver_core.provider.providers.glm import GLMAnthropicProvider

    return GLMAnthropicProvider


def _get_ollama_provider():
    from cliver_core.provider.providers.ollama import OllamaProvider

    return OllamaProvider


def _get_seedance_provider():
    from cliver_core.provider.providers.seedance import SeedanceProvider

    return SeedanceProvider


def _get_dashscope_provider():
    from cliver_core.provider.providers.dashscope import DashScopeProvider

    return DashScopeProvider


def _get_qwen_provider():
    from cliver_core.provider.providers.qwen import QwenProvider

    return QwenProvider


def _get_qwen_image_provider():
    from cliver_core.provider.providers.qwen_image import QwenImageProvider

    return QwenImageProvider


# brand key → protocol → provider class factory.  Every brand entry is
# protocol-locked; a brand without an entry for the requested protocol
# falls back to the protocol-default class in ``resolve_provider_class``
# and carries no default endpoint.
_BRAND_PROTOCOL_FACTORIES: dict[str, dict[str, "callable"]] = {
    "deepseek": {"openai": _get_deepseek_openai, "anthropic": _get_deepseek_anthropic},
    # MiniMax chat rides its Anthropic-compatible endpoint only (no openai
    # leaf); the openai-protocol combo takes the protocol-default fallback
    # and needs an explicit base_url.
    "minimax": {"anthropic": _get_minimax_anthropic},
    "glm": {"openai": _get_glm_provider, "anthropic": _get_glm_anthropic},
    "ollama": {"openai": _get_ollama_provider},
    "openai": {"openai": _get_openai_provider},
    "anthropic": {"anthropic": _get_anthropic_provider},
    # Seedance (Volcano Engine Ark): OpenAI-compatible chat + async-task
    # video; media beyond video reports unsupported.
    "seedance": {"openai": _get_seedance_provider},
    # DashScope (Alibaba Model Studio / Bailian): native async media tasks
    # on /api/v1 (chat is the separate OpenAI-compatible "qwen" brand).
    "dashscope": {"openai": _get_dashscope_provider},
    # Qwen chat on DashScope's OpenAI-compatible endpoint.
    "qwen": {"openai": _get_qwen_provider},
    # Qwen-Image: synchronous reference-aware image generation on /api/v1.
    "qwen-image": {"openai": _get_qwen_image_provider},
}

# brand key → conventional API-key environment variables (first set wins).
# The single source for both the ready-made binding factories
# (``cliver_core.presets``) and the empty-key guard in ``resolve_brand_sdk``.
API_KEY_ENV: dict[str, tuple[str, ...]] = {
    "deepseek": ("DEEPSEEK_API_KEY",),
    "minimax": ("MINIMAX_API_KEY",),
    "glm": ("GLM_API_KEY", "ZHIPUAI_API_KEY"),
    "openai": ("OPENAI_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "seedance": ("SEEDANCE_API_KEY",),
    "dashscope": ("DASHSCOPE_API_KEY",),
    "qwen": ("DASHSCOPE_API_KEY",),
    "qwen-image": ("DASHSCOPE_API_KEY",),
    # Not a chat/media brand: the System One presets read this variable.
    "typesafe": ("TYPESAFE_API_KEY",),
}

# ── Brand selection ──────────────────────────────────────────

# The wire protocols a provider class can speak.  Anything else is a
# configuration error — never silently normalized to a protocol.
_PROTOCOLS = ("openai", "anthropic")


def _brand_class(brand: str, protocol: str) -> type[Provider] | None:
    """The brand's protocol-locked class — exact (brand, protocol) registry hit only."""
    factory = _BRAND_PROTOCOL_FACTORIES.get(brand, {}).get(protocol)
    return factory() if factory is not None else None


def _protocol_default_class(protocol: str) -> type[Provider]:
    """The protocol's standard class — used when no exact brand entry applies."""
    if protocol not in _PROTOCOLS:
        raise ValueError(f"Unknown protocol '{protocol}'. Supported: {list(_PROTOCOLS)}")
    return _get_anthropic_provider() if protocol == "anthropic" else _get_openai_provider()


def resolve_provider_class(
    provider_name: str | None = None,
    *,
    protocol: str = "openai",
) -> type[Provider]:
    """Resolve the protocol-locked provider class for an explicit *provider_name*.

    The name must exactly match a brand key in ``_BRAND_PROTOCOL_FACTORIES``
    (case-insensitive; no substring matching, no aliases).  Unknown names, no
    name at all, and brands without an entry for *protocol* (e.g. ollama +
    anthropic) select the protocol-default class — the same documented swap
    as before the protocol lock.  Class resolution alone never supplies a
    default endpoint; ``create_provider`` decides that from the caller's
    explicit input.
    """
    brand = (provider_name or "").strip().lower()
    return _brand_class(brand, protocol) or _protocol_default_class(protocol)


def create_provider(
    api_key: str | None = None,
    base_url: str | None = None,
    *,
    protocol: str = "openai",
    provider_class: type[Provider] | None = None,
    provider_name: str | None = None,
    user_agent: str | None = None,
    logger: "logging.Logger | None" = None,
    on_event: EventHandler | None = None,
) -> Provider:
    """Create a Provider instance.

    Brand selection is explicit — nothing is inferred from *base_url*:

    1. An explicit *provider_class* wins; its ``default_base_url`` applies.
    2. Otherwise *provider_name* must exactly match a brand key; an exact
       (brand, protocol) registry hit selects the brand class and supplies
       its ``default_base_url``.
    3. Unknown names and brands without an entry for *protocol* select the
       protocol-default class with **no** default endpoint — *base_url* is
       then required.  This is the open door for unlisted vendors: any
       OpenAI- or Anthropic-compatible endpoint works, no registry entry
       needed — pass *base_url* with the right *protocol*.

    Args:
        api_key: API key for the provider.
        base_url: Endpoint base URL.  Required unless an explicit brand
            selection carried a ``default_base_url``.
        protocol: ``"openai"`` (default) or ``"anthropic"`` — the protocol
            whose class is selected for *provider_name*.  An explicitly
            passed *provider_class* wins; the class itself knows its protocol.
        provider_class: Provider class.  If ``None``, selected from
            *provider_name* and *protocol*.
        provider_name: Exact brand key (e.g. ``"deepseek"``, ``"minimax"``,
            ``"glm"``, ``"ollama"``, ``"openai"``, ``"anthropic"``,
            ``"seedance"``; case-insensitive).  No substrings, no aliases.
            Unlisted vendors — self-hosted, proxies, compatible clones —
            need no name at all: pass *base_url* with the right *protocol*.
        user_agent: Optional User-Agent header.
        logger: Optional logger for this provider.  When provided, all log
            output from the provider is routed through this logger.
        on_event: Optional inference-event subscriber (sync or async) for
            the constructed provider.

    Returns:
        A Provider instance ready to use.

    Raises:
        ValueError: No explicit *base_url* and no applicable default endpoint.
    """
    api_key = api_key or ""
    brand = (provider_name or "").strip().lower()

    if provider_class is not None:
        cls = provider_class
        default_url = getattr(provider_class, "default_base_url", "")
    else:
        hit = _brand_class(brand, protocol)
        cls = hit or _protocol_default_class(protocol)
        # an exact (brand, protocol) hit is the only non-explicit source of a default URL
        default_url = getattr(cls, "default_base_url", "") if hit is not None else ""

    url = base_url or ""
    if not url:
        url = default_url
    if not url:
        known = sorted(_BRAND_PROTOCOL_FACTORIES)
        raise ValueError(
            f"No base_url for protocol '{protocol}'"
            + (f" and provider_name '{provider_name}'" if provider_name else "")
            + " — pass base_url explicitly (any OpenAI- or Anthropic-compatible endpoint"
            + " works; no registry entry needed).  Only these exact provider_name values"
            + f" carry a built-in default endpoint: {known}"
        )

    kwargs: dict[str, object] = {
        "api_key": api_key,
        "base_url": url,
        "user_agent": user_agent,
        "logger": logger,
    }
    if on_event is not None:
        kwargs["on_event"] = on_event
    return cls(**kwargs)
