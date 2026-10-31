"""Model bindings — purpose → model handle for AgentCore's embedded APIs.

A :class:`ModelBinding` is a self-contained model handle: which model, on which
endpoint, with which credentials, plus default call parameters — and the
:class:`~cliver_core.provider.Provider` built from those settings.  Each purpose
owning its own Provider means image / tts / asr / ... can each point at a
different endpoint or account.

The provider is built on first use, cached, and reused for every later call;
providers a binding built are closed by :meth:`ModelBinding.arefresh` once
they are retired with no in-flight call, and by :meth:`ModelBinding.aclose`
(a caller-supplied provider is never closed by the binding).

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

A purpose may mount a list of bindings instead of one.  The first element is
the primary — ``bindings.llm`` still returns that single binding, so existing
code is unaffected — and the rest are ordered same-purpose fallbacks AgentCore
uses when a retryable provider failure (rate limit, timeout, 5xx) occurs::

    bindings = ModelBindings(
        llm=[primary_llm, backup_llm],
        image=[primary_image, backup_image],
    )
    bindings.llm               # the primary binding — a single handle
    bindings.get_all("llm")    # [primary_llm, backup_llm] — primary first

Failover never crosses purposes: an ``image`` fallback is never used for chat,
and purpose validation applies to every element of a list.
"""

from __future__ import annotations

import asyncio
import inspect
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Awaitable, Callable, Sequence

from cliver_core.errors import CapabilityNotSupported
from cliver_core.events import EventHandler
from cliver_core.provider import Provider
from cliver_core.provider.providers import create_provider
from cliver_core.provider.providers.system_one import SystemOneProvider
from cliver_core.util import drop_none, validate_client_options

#: Purposes AgentCore can bind models for — ``llm`` is the chat/stream model
#: and ``system_one`` a decision model for structured predictions.
PURPOSES = ("llm", "embed", "rerank", "image", "audio", "video", "tts", "asr", "system_one")

#: Resolves a fresh credential for a binding — sync or async; must return a
#: non-empty string.  Rotation is explicit (``arefresh()``); nothing refreshes
#: automatically on a 401 — the application owns that policy.
CredentialProvider = Callable[[], "str | Awaitable[str]"]

# Re-exported for backward compatibility — the exception now lives in
# :mod:`cliver_core.errors` because the provider layer raises it too.
__all__ = ["CapabilityNotSupported", "CredentialProvider", "ModelBinding", "ModelBindings", "PURPOSES"]


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
    api_key: str = field(default="", repr=False)
    # Excluded from repr: bindings are logged and dumped, and the key is a
    # secret (``repr`` is the leak path into log lines and error reports).
    base_url: str = ""
    protocol: str = "openai"
    provider_name: str | None = None
    provider_class: type[Provider] | None = None
    # Explicit provider class for purposes whose wire is not one of the
    # registry protocols (``system_one``).  When set, ``create_provider``
    # builds this class and a ``default_base_url`` on it applies.
    path: str | None = None
    on_event: EventHandler | None = field(default=None, repr=False, compare=False)
    # Subscriber for the provider this binding builds (sync or async).
    # Applied at construction, so inference events fire before/after every
    # request the binding's provider makes.  With ``provider=`` the supplied
    # instance keeps its own handler — configure it there instead.
    credential_provider: CredentialProvider | None = field(default=None, repr=False)
    # Resolves the credential used by :meth:`arefresh` — sync or async, must
    # return a non-empty string.  Without one, ``arefresh()`` rebuilds with the
    # current ``api_key`` (assign it first to rotate in place).
    client_timeout: float | None = None
    # SDK-client request timeout in seconds.  ``None`` keeps the SDK default
    # (600 s read on both builtin SDKs).  This is a client-construction
    # setting, distinct from a per-request ``timeout`` in ``params``; it
    # applies to every HTTP phase, and a streaming response's read timeout is
    # re-armed per chunk.  Only providers with an SDK transport accept it.
    client_max_retries: int | None = None
    # SDK-client retry count.  ``None`` keeps the SDK default (2 on both
    # builtin SDKs).  This is transport-level retrying, distinct from
    # AgentCore's same-purpose binding failover.
    purpose: str | None = field(default=None, compare=False)
    params: dict[str, Any] = field(default_factory=dict)
    _built: Provider | None = field(default=None, init=False, repr=False, compare=False)
    # The event loop this binding was first used on.  Providers are
    # loop-bound (SDK clients pool connections on their loop), so a binding
    # refuses a second loop instead of failing deep inside the client.
    _loop: asyncio.AbstractEventLoop | None = field(default=None, init=False, repr=False, compare=False)
    # Providers replaced by :meth:`arefresh`.  They outlive the swap only
    # while an in-flight lease still uses them; the next :meth:`arefresh` (or
    # :meth:`aclose`) closes every retired provider with no active lease.
    _retired: list[Provider] = field(default_factory=list, init=False, repr=False, compare=False)
    # In-flight lease count per provider, maintained by :meth:`lease_provider`.
    # A retired provider is never closed while its count is non-zero.
    _leases: dict[Provider, int] = field(default_factory=dict, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.client_timeout, self.client_max_retries = validate_client_options(
            self.client_timeout, self.client_max_retries
        )
        if self.provider is not None and (self.client_timeout is not None or self.client_max_retries is not None):
            raise ValueError(
                "client_timeout/client_max_retries apply only to a provider the binding builds; "
                "configure the supplied provider's SDK client directly."
            )

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
        on_event: EventHandler | None = None,
        credential_provider: CredentialProvider | None = None,
        client_timeout: float | None = None,
        client_max_retries: int | None = None,
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

        ``on_event`` subscribes to inference events from the provider this
        binding builds — the same surface as passing ``on_event=`` to the
        provider constructor.  It applies only to a provider the binding
        builds; a supplied ``provider=`` keeps its own handler.

        ``client_timeout`` and ``client_max_retries`` tune the SDK client this
        binding builds (positive seconds; non-negative retries); ``None``
        keeps the SDK's defaults.  A per-request ``timeout`` belongs in
        ``params`` instead.
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
            on_event=on_event,
            credential_provider=credential_provider,
            client_timeout=client_timeout,
            client_max_retries=client_max_retries,
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
        credential_provider: CredentialProvider | None = None,
        client_timeout: float | None = None,
        client_max_retries: int | None = None,
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
            credential_provider=credential_provider,
            client_timeout=client_timeout,
            client_max_retries=client_max_retries,
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
        credential_provider: CredentialProvider | None = None,
        client_timeout: float | None = None,
        client_max_retries: int | None = None,
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
            credential_provider=credential_provider,
            client_timeout=client_timeout,
            client_max_retries=client_max_retries,
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
        credential_provider: CredentialProvider | None = None,
        client_timeout: float | None = None,
        client_max_retries: int | None = None,
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
            credential_provider=credential_provider,
            client_timeout=client_timeout,
            client_max_retries=client_max_retries,
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
        credential_provider: CredentialProvider | None = None,
        client_timeout: float | None = None,
        client_max_retries: int | None = None,
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
            credential_provider=credential_provider,
            client_timeout=client_timeout,
            client_max_retries=client_max_retries,
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
        credential_provider: CredentialProvider | None = None,
        client_timeout: float | None = None,
        client_max_retries: int | None = None,
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
            credential_provider=credential_provider,
            client_timeout=client_timeout,
            client_max_retries=client_max_retries,
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
        credential_provider: CredentialProvider | None = None,
        client_timeout: float | None = None,
        client_max_retries: int | None = None,
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
            credential_provider=credential_provider,
            client_timeout=client_timeout,
            client_max_retries=client_max_retries,
            path=path,
            purpose="rerank",
            params={**drop_none(top_k=top_k), **params},
        )

    @classmethod
    def system_one(
        cls,
        *,
        model: str,
        api_key: str = "",
        base_url: str = "",
        provider: Provider | None = None,
        path: str | None = None,
        protocol: str = "system-one",
        provider_class: type[Provider] | None = None,
        on_event: EventHandler | None = None,
        credential_provider: CredentialProvider | None = None,
        client_timeout: float | None = None,
        client_max_retries: int | None = None,
        **params: Any,
    ) -> ModelBinding:
        """System One decision-model binding — predictions, never text.

        The built provider speaks the shared ``/v1/systemone`` HTTP wire
        (Jev, Aliyun Bailian, ``laya-serve``), so the binding needs only
        ``base_url`` and ``api_key`` — no brand registry entry.  Bailian's
        base URL ends in ``/compatible-mode`` (the workspace id is part of
        the hostname); the preset factories in :mod:`cliver_core.presets`
        build those URLs for you.  A local ``laya`` backend is mounted with
        ``cliver_core.local.local_system_one``, which supplies
        ``protocol="local"`` and its provider object.

        ``provider_class`` defaults to the shared HTTP wire class for
        ``protocol="system-one"``; other protocols expect a supplied
        ``provider`` and select no class of their own.
        """
        if provider_class is None and protocol == "system-one":
            provider_class = SystemOneProvider
        return cls(
            model=model,
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            protocol=protocol,
            provider_class=provider_class,
            path=path,
            on_event=on_event,
            credential_provider=credential_provider,
            client_timeout=client_timeout,
            client_max_retries=client_max_retries,
            purpose="system_one",
            params=dict(params),
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

    def _claim_loop(self) -> None:
        """Bind this handle to the running loop, rejecting a different one.

        Providers are loop-bound: the SDK clients own connection pools and
        stream tasks on the loop that created them.  The first async use
        claims the loop; any later use from another loop (in practice another
        thread) fails here with an actionable message rather than inside
        ``httpx``.  Calls made outside a running loop are not recorded.
        """
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            return
        if self._loop is None:
            self._loop = running
        elif self._loop is not running:
            raise RuntimeError(
                f"ModelBinding for '{self.model}' is bound to another event loop; "
                "create one binding (and AgentCore) per event loop — one per thread."
            )

    def get_provider(self) -> Provider:
        """The provider for this binding — the supplied one, or built once and reused.

        ``on_event`` is applied when the binding builds the provider.  A
        caller-supplied ``provider=`` is used as-is; its own ``on_event``
        (set at construction) is the subscription surface, so combining
        ``provider=`` with a binding-level ``on_event`` is a configuration
        error rather than a silently ignored handler.

        The first async use claims the running event loop; a later use from
        another loop raises ``RuntimeError`` — the cached provider's clients
        are loop-bound.
        """
        self._claim_loop()
        if self.provider is not None:
            if self.on_event is not None:
                raise ValueError(
                    "ModelBinding.on_event applies only to a provider the binding builds; "
                    "with provider=..., pass on_event to that provider's constructor instead."
                )
            return self.provider
        if self._built is None:
            self._built = create_provider(
                api_key=self.api_key,
                base_url=self.base_url,
                protocol=self.protocol,
                provider_class=self.provider_class,
                provider_name=self.provider_name,
                on_event=self.on_event,
                client_timeout=self.client_timeout,
                client_max_retries=self.client_max_retries,
            )
        return self._built

    @asynccontextmanager
    async def lease_provider(self) -> AsyncIterator[Provider]:
        """Yield the current provider while holding an in-flight lease.

        Every :class:`~cliver_core.agent.AgentCore` model call, capability
        call, and System One prediction holds one of these, so
        :meth:`arefresh` closes a leased-out provider at the first refresh
        after its last in-flight call finishes, instead of keeping it until
        :meth:`aclose`.
        Use this — not :meth:`get_provider` — whenever a provider is used
        across an ``await``::

            async with binding.lease_provider() as provider:
                response = await provider.chat(request)

        Leases are per binding and per event loop, like the provider itself.
        """
        provider = self.get_provider()
        self._leases[provider] = self._leases.get(provider, 0) + 1
        try:
            yield provider
        finally:
            count = self._leases.get(provider, 1) - 1
            if count:
                self._leases[provider] = count
            else:
                self._leases.pop(provider, None)

    async def _prune_retired(self) -> None:
        """Close retired providers whose in-flight leases have all finished."""
        remaining: list[Provider] = []
        for provider in self._retired:
            if self._leases.get(provider, 0):
                remaining.append(provider)
            else:
                await provider.close()
        self._retired = remaining

    async def arefresh(self) -> None:
        """Rotate this binding's credential and rebuild its provider.

        The credential comes from ``credential_provider`` (sync or async); it
        must resolve to a non-empty string or :class:`ValueError` is raised
        before anything is mutated.  Without a credential provider the current
        ``api_key`` is reused — assign it first to rotate in place.

        The replacement provider is built **before** the swap: on build
        failure the binding keeps its old provider and key.  The old provider
        is retired: a call that already holds it through
        :meth:`lease_provider` keeps working undisturbed, and once its last
        lease is released the next refresh closes it.  When nothing is in
        flight the old provider is closed inside this call, so frequent
        rotation on a long-lived server does not accumulate providers.  A
        caller-supplied ``provider=`` is never rebuilt or closed; only
        ``api_key`` updates (media paths read it from the binding).

        Runs on the owning loop; another loop raises ``RuntimeError``.  There
        is no automatic refresh on a 401 — the application decides when to
        rotate (TTL timer, or catch :class:`ProviderError` and call this).
        """
        self._claim_loop()
        if self.credential_provider is None:
            credential = self.api_key
        else:
            credential = self.credential_provider()
            if inspect.isawaitable(credential):
                credential = await credential
            if not isinstance(credential, str) or not credential.strip():
                raise ValueError(
                    f"credential_provider for '{self.model}' must return a non-empty string, got {credential!r}"
                )
        if self.provider is not None or self._built is None:
            # Supplied provider: never rebuild or close it.  Unbuilt binding:
            # the next get_provider() builds with the new key.
            self.api_key = credential
            return
        replacement = create_provider(
            api_key=credential,
            base_url=self.base_url,
            protocol=self.protocol,
            provider_class=self.provider_class,
            provider_name=self.provider_name,
            on_event=self.on_event,
            client_timeout=self.client_timeout,
            client_max_retries=self.client_max_retries,
        )
        previous, self._built = self._built, replacement
        self.api_key = credential
        self._retired.append(previous)
        await self._prune_retired()

    async def aclose(self) -> None:
        """Close the provider this binding built and every retired one.

        A caller-supplied provider stays open.  Must run on the loop the
        binding was used on, and releases the loop claim so the binding can be
        mounted in a new agent afterwards.  In-flight leases are not awaited
        here — the agent drains its work before closing its bindings; a
        provider closed under a still-running call is the caller's contract
        violation, exactly as before.
        """
        self._claim_loop()
        if self._built is not None:
            built, self._built = self._built, None
            await built.close()
        retired, self._retired = self._retired, []
        for provider in retired:
            await provider.close()
        self._leases.clear()
        self._loop = None


@dataclass(kw_only=True)
class ModelBindings:
    """Per-purpose model handles for AgentCore.

    ``llm`` is the chat/stream model and is required by ``AgentCore``; the
    remaining fields are optional — an unbound purpose reports "not supported"
    from AgentCore's embedded model APIs.

    Each purpose accepts one :class:`ModelBinding` (unchanged) or an ordered
    sequence whose first element becomes the primary.  After construction the
    attribute itself always holds the primary ``ModelBinding``; the remaining
    elements are same-purpose fallbacks exposed by :meth:`get_all`.
    """

    llm: ModelBinding | Sequence[ModelBinding] | None = None
    embed: ModelBinding | Sequence[ModelBinding] | None = None
    rerank: ModelBinding | Sequence[ModelBinding] | None = None
    image: ModelBinding | Sequence[ModelBinding] | None = None
    audio: ModelBinding | Sequence[ModelBinding] | None = None
    video: ModelBinding | Sequence[ModelBinding] | None = None
    tts: ModelBinding | Sequence[ModelBinding] | None = None
    asr: ModelBinding | Sequence[ModelBinding] | None = None
    system_one: ModelBinding | Sequence[ModelBinding] | None = None

    #: Same-purpose fallbacks, keyed by purpose (empty for single mounts).
    _fallbacks: dict[str, list[ModelBinding]] = field(default_factory=dict, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Normalize list mounts: primary in the field, extras in ``_fallbacks``."""
        fallbacks: dict[str, list[ModelBinding]] = {}
        for purpose in PURPOSES:
            mount = getattr(self, purpose)
            if mount is None or isinstance(mount, ModelBinding):
                continue
            if not isinstance(mount, (list, tuple)):
                raise TypeError(
                    f"bindings.{purpose} must be a ModelBinding, a list of "
                    f"ModelBindings, or None (got {type(mount).__name__})"
                )
            if not mount:
                raise ValueError(f"bindings.{purpose} needs at least one binding when given a list")
            for binding in mount:
                if not isinstance(binding, ModelBinding):
                    raise TypeError(
                        f"bindings.{purpose} list entries must be ModelBindings (got {type(binding).__name__})"
                    )
            setattr(self, purpose, mount[0])
            if len(mount) > 1:
                fallbacks[purpose] = list(mount[1:])
        self._fallbacks = fallbacks

    def get(self, purpose: str) -> ModelBinding | None:
        """Return the primary binding for *purpose* (a :data:`PURPOSES` member), or None.

        Fails fast when the mounted binding was created for a different
        purpose (e.g. a ``ModelBinding.image(...)`` in ``bindings.embed``).
        The legacy ``audio`` slot accepts ``tts`` bindings.
        """
        if purpose not in PURPOSES:
            raise ValueError(f"Unknown purpose '{purpose}'. Supported: {list(PURPOSES)}")
        binding = getattr(self, purpose)
        if binding is not None:
            self._check_purpose(purpose, binding)
        return binding

    def get_all(self, purpose: str) -> list[ModelBinding]:
        """Every binding mounted for *purpose*, primary first, then fallbacks.

        An unbound purpose returns ``[]``.  The same purpose validation as
        :meth:`get` applies to every element, so a fallback created for a
        different purpose is rejected rather than silently used.
        """
        primary = self.get(purpose)
        if primary is None:
            return []
        group = [primary, *self._fallbacks.get(purpose, [])]
        for binding in group[1:]:
            self._check_purpose(purpose, binding)
        return group

    @staticmethod
    def _check_purpose(purpose: str, binding: ModelBinding) -> None:
        if binding.purpose is not None and binding.purpose != purpose:
            if {binding.purpose, purpose} != {"audio", "tts"}:  # legacy alias pair
                raise ValueError(
                    f"bindings.{purpose} holds a '{binding.purpose}' binding — "
                    f"mount it in bindings.{binding.purpose} instead."
                )

    def bound_purposes(self) -> list[str]:
        """Purposes that have a model bound."""
        return [p for p in PURPOSES if getattr(self, p) is not None]

    async def arefresh(self) -> None:
        """Rotate the credential of every mounted binding — primary and fallbacks.

        Purposes are visited in :data:`PURPOSES` order and bindings in mount
        order.  The first failure (resolution or build) stops the pass with
        its exception; bindings already refreshed keep their new credentials.
        """
        for purpose in self.bound_purposes():
            for binding in self.get_all(purpose):
                await binding.arefresh()

    async def aclose(self) -> None:
        """Close providers built by these bindings — caller-supplied ones stay open."""
        for purpose in self.bound_purposes():
            for binding in self.get_all(purpose):
                await binding.aclose()
