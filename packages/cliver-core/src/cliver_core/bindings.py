"""Model bindings — purpose → model handle for AgentCore's embedded APIs.

A :class:`ModelBinding` is a self-contained model handle: which model, on which
endpoint, with which credentials, plus default call parameters — and the
:class:`~cliver_core.provider.Provider` built from those settings.  Each purpose
owning its own Provider means image / tts / asr / ... can each point at a
different endpoint or account.

The provider is built on first use, cached, and reused for every later call;
providers a binding built are closed by :meth:`ModelBinding.aclose` (a
caller-supplied provider is never closed by the binding).

Build bindings with the per-purpose factories — each knows the common
parameters of its purpose and folds them into ``params``::

    bindings = ModelBindings(
        llm=ModelBinding.llm(
            model="gpt-4o",
            api_key="sk-...",
            base_url="https://api.openai.com/v1",
            temperature=0.3,        # default chat options → binding.params
        ),
        image=ModelBinding.image(
            model="gpt-image-1",
            api_key="sk-...",
            base_url="https://api.openai.com/v1",
            size="1024x1024",       # purpose params
            n=2,                    # → binding.params
        ),
        tts=ModelBinding.tts(
            model="gpt-4o-mini-tts",
            api_key="sk-...",
            base_url="https://api.openai.com/v1",
            voice="alloy",
        ),
    )
    agent = AgentCore(bindings)         # chat/stream run on bindings.llm

    await agent.image("a red panda")   # → capabilities.generate_image(
                                       #       prompt=..., binding=bindings.image)
                                       #   the binding carries the model name, the
                                       #   cached provider, and the size/n defaults
    await agent.aclose()               # closes providers the bindings built
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cliver_core.errors import CapabilityNotSupported
from cliver_core.provider import Provider
from cliver_core.provider.providers import create_provider
from cliver_core.util import drop_none

#: Purposes AgentCore can bind models for — ``llm`` is the chat/stream model.
PURPOSES = ("llm", "embed", "rerank", "image", "audio", "video", "tts", "asr")

# Re-exported for backward compatibility — the exception now lives in
# :mod:`cliver_core.errors` because the provider layer raises it too.
__all__ = ["CapabilityNotSupported", "ModelBinding", "ModelBindings", "PURPOSES"]


@dataclass(kw_only=True)
class ModelBinding:
    """A model bound to one purpose — definition + provider + default params.

    The *model definition* is ``model`` / ``api_key`` / ``base_url`` /
    ``protocol`` / ``provider_name``, plus ``path`` — the endpoint path
    override.  Like ``base_url``, ``path`` says *where* the request goes
    (it is never sent in the request body); when unset, the per-purpose path
    resolves through the provider's brand map and protocol default.

    Two ways to give the binding its provider:

    * ``provider=`` — a pre-built provider, used as-is and **never** closed by
      the binding (the caller owns its lifecycle);
    * otherwise one is built lazily from the model definition (``api_key`` /
      ``base_url`` / ``protocol`` / ``provider_name``) on first use, cached,
      and reused for every following call.

    ``params`` holds default call parameters (e.g. image ``size`` / tts
    ``voice``) that individual calls may override.  ``provider_name`` is the
    explicit brand selector — an exact registry key (e.g. ``"minimax"``,
    ``"deepseek"``) that also supplies the brand's default endpoint; without
    one, *base_url* must be set.  ``purpose`` records which factory created
    the binding (``None`` for the raw constructor); it is diagnostics/validation
    metadata — routing is by the slot in :class:`ModelBindings`, which rejects
    a binding mounted under the wrong purpose.

    Prefer the per-purpose factories below (:meth:`llm`, :meth:`image`,
    :meth:`video`, :meth:`tts`, :meth:`asr`, :meth:`embed`, :meth:`rerank`) —
    the legacy ``audio`` purpose uses :meth:`tts`.  An ``llm`` binding is the
    chat/stream handle for :class:`~cliver_core.agent.AgentCore`.
    """

    model: str
    provider: Provider | None = None
    api_key: str = ""
    base_url: str = ""
    protocol: str = "openai"
    provider_name: str | None = None
    path: str | None = None
    purpose: str | None = field(default=None, compare=False)
    params: dict[str, Any] = field(default_factory=dict)
    _built: Provider | None = field(default=None, init=False, repr=False, compare=False)

    # ── Per-purpose factories ─────────────────────────────────

    @classmethod
    def llm(
        cls,
        *,
        model: str,
        api_key: str = "",
        base_url: str = "",
        protocol: str = "openai",
        provider_name: str | None = None,
        provider: Provider | None = None,
        path: str | None = None,
        **params: Any,
    ) -> ModelBinding:
        """Chat/stream-purpose binding — the LLM handle for ``AgentCore``.

        ``params`` are default request options (e.g. ``temperature``,
        ``max_tokens``); call-site ``options`` override them.  Unlike the
        media factories there is no endpoint ``path`` — chat requests have no
        per-purpose path override, and a ``path=`` here is rejected.  Endpoints
        speaking the Anthropic protocol need ``protocol="anthropic"``
        explicitly (the default protocol selects the OpenAI-protocol class);
        with no recognized ``provider_name`` there is no default endpoint, so
        ``base_url`` is required.
        """
        if path is not None:
            raise TypeError(
                "ModelBinding.llm() takes no path — chat requests have no endpoint path "
                "override (provider, base_url, and protocol configure the endpoint)."
            )
        return cls(
            model=model,
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            protocol=protocol,
            provider_name=provider_name,
            purpose="llm",
            params=dict(params),
        )

    @classmethod
    def image(
        cls,
        *,
        model: str,
        api_key: str = "",
        base_url: str = "",
        protocol: str = "openai",
        provider_name: str | None = None,
        provider: Provider | None = None,
        path: str | None = None,
        n: int | None = None,
        size: str | None = None,
        style: str | None = None,
        quality: str | None = None,
        **params: Any,
    ) -> ModelBinding:
        """Image-purpose binding — ``path`` is the endpoint override;
        ``n`` / ``size`` / ``style`` / ``quality`` become default params."""
        return cls(
            model=model,
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            protocol=protocol,
            provider_name=provider_name,
            path=path,
            purpose="image",
            params={**drop_none(n=n, size=size, style=style, quality=quality), **params},
        )

    @classmethod
    def video(
        cls,
        *,
        model: str,
        api_key: str = "",
        base_url: str = "",
        protocol: str = "openai",
        provider_name: str | None = None,
        provider: Provider | None = None,
        path: str | None = None,
        duration: int | None = None,
        resolution: str | None = None,
        **params: Any,
    ) -> ModelBinding:
        """Video-purpose binding — ``path`` is the endpoint override;
        ``duration`` / ``resolution`` become default params."""
        return cls(
            model=model,
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            protocol=protocol,
            provider_name=provider_name,
            path=path,
            purpose="video",
            params={**drop_none(duration=duration, resolution=resolution), **params},
        )

    @classmethod
    def tts(
        cls,
        *,
        model: str,
        api_key: str = "",
        base_url: str = "",
        protocol: str = "openai",
        provider_name: str | None = None,
        provider: Provider | None = None,
        voice: str | None = None,
        response_format: str | None = None,
        path: str | None = None,
        **params: Any,
    ) -> ModelBinding:
        """TTS-purpose binding — ``path`` is the endpoint override;
        ``voice`` / ``response_format`` become default params."""
        return cls(
            model=model,
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            protocol=protocol,
            provider_name=provider_name,
            path=path,
            purpose="tts",
            params={**drop_none(voice=voice, response_format=response_format), **params},
        )

    @classmethod
    def asr(
        cls,
        *,
        model: str,
        api_key: str = "",
        base_url: str = "",
        protocol: str = "openai",
        provider_name: str | None = None,
        provider: Provider | None = None,
        language: str | None = None,
        path: str | None = None,
        **params: Any,
    ) -> ModelBinding:
        """ASR-purpose binding — ``path`` is the endpoint override;
        ``language`` becomes a default param."""
        return cls(
            model=model,
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            protocol=protocol,
            provider_name=provider_name,
            path=path,
            purpose="asr",
            params={**drop_none(language=language), **params},
        )

    @classmethod
    def embed(
        cls,
        *,
        model: str,
        api_key: str = "",
        base_url: str = "",
        protocol: str = "openai",
        provider_name: str | None = None,
        provider: Provider | None = None,
        dimensions: int | None = None,
        path: str | None = None,
        **params: Any,
    ) -> ModelBinding:
        """Embedding-purpose binding — ``path`` is the endpoint override;
        ``dimensions`` becomes a default param."""
        return cls(
            model=model,
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            protocol=protocol,
            provider_name=provider_name,
            path=path,
            purpose="embed",
            params={**drop_none(dimensions=dimensions), **params},
        )

    @classmethod
    def rerank(
        cls,
        *,
        model: str,
        api_key: str = "",
        base_url: str = "",
        protocol: str = "openai",
        provider_name: str | None = None,
        provider: Provider | None = None,
        top_k: int | None = None,
        path: str | None = None,
        **params: Any,
    ) -> ModelBinding:
        """Rerank-purpose binding — ``path`` is the endpoint override;
        ``top_k`` becomes a default param."""
        return cls(
            model=model,
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            protocol=protocol,
            provider_name=provider_name,
            path=path,
            purpose="rerank",
            params={**drop_none(top_k=top_k), **params},
        )

    # ── Call parameters ───────────────────────────────────────

    def call_defaults(self) -> dict[str, Any]:
        """Request parameters this binding supplies by default.

        The endpoint ``path`` override (when set) plus ``params`` — the
        default call parameters the per-purpose factories collect.
        Capabilities merge call-site parameters *over* these, so the call
        site wins on conflict and an unset call-site parameter leaves the
        binding default in place.
        """
        return {**drop_none(path=self.path), **self.params}

    # ── Provider handle ───────────────────────────────────────

    def get_provider(self) -> Provider:
        """The provider for this binding — the supplied one, or built once and reused."""
        if self.provider is not None:
            return self.provider
        if self._built is None:
            self._built = create_provider(
                api_key=self.api_key,
                base_url=self.base_url,
                protocol=self.protocol,
                provider_name=self.provider_name,
            )
        return self._built

    async def aclose(self) -> None:
        """Close the provider this binding built — a caller-supplied one stays open."""
        if self._built is not None:
            built, self._built = self._built, None
            await built.close()


@dataclass(kw_only=True)
class ModelBindings:
    """Per-purpose model handles for AgentCore.

    ``llm`` is the chat/stream model and is required by ``AgentCore``; the
    remaining fields are optional — an unbound purpose reports "not supported"
    from AgentCore's embedded model APIs.
    """

    llm: ModelBinding | None = None
    embed: ModelBinding | None = None
    rerank: ModelBinding | None = None
    image: ModelBinding | None = None
    audio: ModelBinding | None = None
    video: ModelBinding | None = None
    tts: ModelBinding | None = None
    asr: ModelBinding | None = None

    def get(self, purpose: str) -> ModelBinding | None:
        """Return the binding for *purpose* (a :data:`PURPOSES` member), or None.

        Fails fast when the mounted binding was created for a different
        purpose (e.g. a ``ModelBinding.image(...)`` in ``bindings.embed``).
        The legacy ``audio`` slot accepts ``tts`` bindings.
        """
        if purpose not in PURPOSES:
            raise ValueError(f"Unknown purpose '{purpose}'. Supported: {list(PURPOSES)}")
        binding = getattr(self, purpose)
        if binding is not None and binding.purpose is not None and binding.purpose != purpose:
            if {binding.purpose, purpose} != {"audio", "tts"}:  # legacy alias pair
                raise ValueError(
                    f"bindings.{purpose} holds a '{binding.purpose}' binding — "
                    f"mount it in bindings.{binding.purpose} instead."
                )
        return binding

    def bound_purposes(self) -> list[str]:
        """Purposes that have a model bound."""
        return [p for p in PURPOSES if getattr(self, p) is not None]

    async def aclose(self) -> None:
        """Close providers built by these bindings — caller-supplied ones stay open."""
        for purpose in self.bound_purposes():
            binding: ModelBinding = getattr(self, purpose)
            await binding.aclose()
