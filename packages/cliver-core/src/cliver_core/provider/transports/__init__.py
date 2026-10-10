"""BrandSDK resolution — pick the transport for a (brand, protocol) pair.

Preference order:

1. A brand SDK adapter registered by a third party under the
   ``cliver.provider_sdks`` entry-point group (entry point name = brand)
   that supports the requested protocol.
2. The builtin adapter for the protocol — the official OpenAI / Anthropic
   SDK (a hard cliver-core dependency), pointed at the brand's
   ``base_url``.  Every OpenAI- or Anthropic-compatible endpoint works
   through it with zero new code.

Unknown protocols raise ``ValueError``; so do an empty base URL (a protocol
provider always needs an explicit endpoint) and an empty API key — the
latter naming the brand's conventional environment variable(s)
(``API_KEY_ENV``) instead of the SDK's misleading protocol-brand hint.
"""

from __future__ import annotations

import functools
import logging
from importlib.metadata import entry_points

from cliver_core.provider.providers import API_KEY_ENV
from cliver_core.provider.transport import BrandSDK
from cliver_core.provider.transports.anthropic_sdk import AnthropicSDK
from cliver_core.provider.transports.openai_sdk import OpenAISDK

logger = logging.getLogger(__name__)

PROVIDER_SDKS_GROUP = "cliver.provider_sdks"

# Builtin protocol adapters — the official SDKs cliver-core depends on.
_BUILTIN_ADAPTERS: dict[str, type] = {
    "openai": OpenAISDK,
    "anthropic": AnthropicSDK,
}


@functools.lru_cache(maxsize=1)
def registered_adapters() -> dict[str, type]:
    """Brand key → third-party adapter class, from ``cliver.provider_sdks``."""
    adapters: dict[str, type] = {}
    for entry_point in entry_points(group=PROVIDER_SDKS_GROUP):
        try:
            adapters[entry_point.name] = entry_point.load()
        except Exception as exc:  # a broken third-party adapter must not break the rest
            logger.warning("Failed to load provider SDK adapter '%s': %s", entry_point.name, exc)
    return adapters


def resolve_brand_sdk(
    protocol: str,
    *,
    api_key: str,
    base_url: str,
    brand: str | None = None,
    user_agent: str | None = None,
    use_bearer_auth: bool = False,
) -> BrandSDK:
    """Resolve the transport for *brand* speaking *protocol*.

    ``brand`` defaults to *protocol* — i.e. the protocol-default brands
    ``openai`` / ``anthropic``.  A third-party brand SDK adapter wins when
    it supports the protocol; otherwise the protocol's builtin SDK adapter
    serves the brand through its ``base_url``.
    """
    requested = (brand or protocol).lower()

    adapter_cls = registered_adapters().get(requested)
    if adapter_cls is not None:
        if protocol in getattr(adapter_cls, "supported_protocols", ()):
            try:
                return adapter_cls.create(
                    api_key=api_key,
                    base_url=base_url,
                    protocol=protocol,
                    user_agent=user_agent,
                    use_bearer_auth=use_bearer_auth,
                )
            except ImportError as exc:
                logger.warning(
                    "Brand SDK adapter '%s' unavailable (%s); using builtin %s transport",
                    requested,
                    exc,
                    protocol,
                )
        else:
            logger.warning(
                "Brand SDK adapter '%s' does not support protocol '%s'; using builtin %s transport",
                requested,
                protocol,
                protocol,
            )

    builtin_cls = _BUILTIN_ADAPTERS.get(protocol)
    if builtin_cls is None:
        raise ValueError(f"Unknown protocol '{protocol}'. Supported: ['openai', 'anthropic']")
    if not base_url:
        # create_provider always resolves a URL, so this fires for a protocol
        # provider constructed by hand — fail loudly instead of letting the
        # SDK build a URL-less client.
        raise ValueError(f"the '{requested}' provider needs a base_url — pass base_url=...")
    if not api_key:
        # Same failure point as the SDK's own missing-credentials error, but
        # with the brand's variable named — "set OPENAI_API_KEY" would be
        # misleading advice for, say, DeepSeek.
        env_vars = API_KEY_ENV.get(requested, ())
        if env_vars:
            hint = f" or set {env_vars[0]}"
        else:
            hint = " (pass a placeholder value if the endpoint needs no auth)"
        raise ValueError(f"the '{requested}' provider needs an API key — pass api_key=...{hint}")
    return builtin_cls.create(
        api_key=api_key,
        base_url=base_url,
        protocol=protocol,
        user_agent=user_agent,
        use_bearer_auth=use_bearer_auth,
    )
