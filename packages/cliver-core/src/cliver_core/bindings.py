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
from dataclasses import dataclass, field, replace
from typing import Any, AsyncIterator, Awaitable, Callable, Sequence

from cliver_core.errors import CapabilityNotSupported, LoopAffinityError
from cliver_core.events import EventHandler
from cliver_core.provider import Provider
from cliver_core.provider.media_shapes import SHAPES
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

    A loop-bound provider claims the event loop it is first used on; a later
    use from another loop raises
    :class:`~cliver_core.errors.LoopAffinityError`.  A caller-supplied
    provider marked ``loop_agnostic`` (the in-process local providers) is
    exempt, and :meth:`for_current_loop` clones the binding spec for use on
    another loop.

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
    wire_shape: str | None = None
    # Wire-shape selector for media bindings — names an entry in the
    # provider's ``MEDIA_SHAPES`` (``cliver_core.provider.media_shapes``).
    # Like ``path`` it says *how* the request is built, and is never sent in
    # the request body.  Only meaningful on image/video/tts bindings; a brand
    # with one shape per purpose needs none (the provider default applies).
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
        if self.wire_shape is not None:
            if self.provider is not None:
                raise ValueError(
                    "wire_shape applies only to a provider the binding builds; "
                    "a supplied provider= keeps its own shapes — configure it there."
                )
            shape = SHAPES.get(self.wire_shape)
            if shape is None:
                raise ValueError(f"unknown wire shape '{self.wire_shape}'; known shapes: {sorted(SHAPES)}")
            if self.purpose is not None and shape.purpose not in ("any", self.purpose):
                raise ValueError(
                    f"wire shape '{self.wire_shape}' is for {shape.purpose} requests, "
                    f"not this binding's purpose ('{self.purpose}')"
                )

    def __repr__(self) -> str:
        """Show the binding's state without secret material.

        ``loop=0x...`` is the id of the event loop the binding first used
        (``None`` before first use), and ``built=`` says whether the binding
        built its provider — a caller-supplied ``provider=`` keeps it
        ``False``.  Never includes ``api_key``, ``on_event``, or
        ``credential_provider``.
        """
        loop = f"0x{id(self._loop):x}" if self._loop is not None else "None"
        return (
            f"ModelBinding(model={self.model!r}, provider={self.provider!r}, "
            f"base_url={self.base_url!r}, protocol={self.protocol!r}, "
            f"provider_name={self.provider_name!r}, provider_class={self.provider_class!r}, "
            f"path={self.path!r}, wire_shape={self.wire_shape!r}, "
            f"client_timeout={self.client_timeout!r}, client_max_retries={self.client_max_retries!r}, "
            f"purpose={self.purpose!r}, params={self.params!r}, "
            f"loop={loop}, built={self._built is not None})"
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
        wire_shape: str | None = None,
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
        if wire_shape is not None:
            raise TypeError(
                "ModelBinding.llm() takes no wire_shape — chat media input is not a media "
                "wire; images ride the message content, not a reference shape."
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
        wire_shape: str | None = None,
        n: int | None = None,
        size: str | None = None,
        style: str | None = None,
        quality: str | None = None,
        **params: Any,
    ) -> ModelBinding:
        """Image-purpose binding — ``path`` is the endpoint override;
        ``wire_shape`` names the reference-media wire;
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
            wire_shape=wire_shape,
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
        wire_shape: str | None = None,
        duration: int | None = None,
        resolution: str | None = None,
        **params: Any,
    ) -> ModelBinding:
        """Video-purpose binding — ``path`` is the endpoint override;
        ``wire_shape`` names the reference-media wire;
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
            wire_shape=wire_shape,
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
        wire_shape: str | None = None,
        **params: Any,
    ) -> ModelBinding:
        """TTS-purpose binding — ``path`` is the endpoint override;
        ``wire_shape`` names the reference-media wire;
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
            wire_shape=wire_shape,
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
        wire_shape: str | None = None,
        **params: Any,
    ) -> ModelBinding:
        """ASR-purpose binding — ``path`` is the endpoint override;
        ``language`` becomes a default param."""
        if wire_shape is not None:
            raise TypeError(
                "ModelBinding.asr() takes no wire_shape — transcription is a multipart "
                "upload of the audio itself; there is no reference-media wire to select."
            )
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
        wire_shape: str | None = None,
        **params: Any,
    ) -> ModelBinding:
        """Embedding-purpose binding — ``path`` is the endpoint override;
        ``dimensions`` becomes a default param."""
        if wire_shape is not None:
            raise TypeError("ModelBinding.embed() takes no wire_shape — embedding requests carry no media.")
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
        wire_shape: str | None = None,
        **params: Any,
    ) -> ModelBinding:
        """Rerank-purpose binding — ``path`` is the endpoint override;
        ``top_k`` becomes a default param."""
        if wire_shape is not None:
            raise TypeError("ModelBinding.rerank() takes no wire_shape — rerank requests carry no media.")
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

    @property
    def owner_loop(self) -> asyncio.AbstractEventLoop | None:
        """The event loop this binding first used, or ``None`` before first use.

        A loop-bound provider claims the running loop on first use — an eager
        :meth:`preflight` / :meth:`build_provider`, or any model or capability
        call.  ``None`` also after :meth:`aclose` released a binding that can
        rebuild its provider, and always for a ``loop_agnostic`` provider.
        """
        return self._loop

    @staticmethod
    def _is_loop_agnostic(provider: Provider) -> bool:
        """Whether *provider* declared it owns no loop-bound resources."""
        return bool(getattr(provider, "loop_agnostic", False))

    def _claim_loop(self) -> None:
        """Bind this handle to the running loop, rejecting a different one.

        Only loop-bound resources claim: an SDK client owns connection pools
        and stream tasks on the loop that created it.  The first async use
        claims the loop; any later use from another loop (in practice another
        thread) fails here with :class:`LoopAffinityError` rather than inside
        ``httpx``.  Providers marked ``loop_agnostic`` own no such resources
        and never reach this check.  Calls made outside a running loop are
        not recorded.
        """
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            return
        if self._loop is None:
            self._loop = running
        elif self._loop is not running:
            raise LoopAffinityError(
                f"ModelBinding for '{self.model}' is bound to another event loop; "
                "create one binding (and AgentCore) per event loop — one per thread — "
                "or call ModelBinding.for_current_loop() for a loop-agnostic provider.",
                model=self.model,
                owner_loop=self._loop,
                current_loop=running,
            )

    def _claim_loop_for_providers(self) -> None:
        """Claim the loop only when this binding touches a loop-bound provider."""
        providers = (self.provider, self._built, *self._retired)
        if any(provider is not None and not self._is_loop_agnostic(provider) for provider in providers):
            self._claim_loop()

    def build_provider(self) -> Provider:
        """Build this binding's provider for the calling loop (idempotent).

        The provider is the supplied one, or one built once and reused:
        calling this again returns the same instance without rebuilding.
        ``on_event`` is applied when the binding builds the provider.  A
        caller-supplied ``provider=`` is used as-is; its own ``on_event``
        (set at construction) is the subscription surface, so combining
        ``provider=`` with a binding-level ``on_event`` is a configuration
        error rather than a silently ignored handler.

        A loop-bound provider claims the running event loop; a later use from
        another loop raises :class:`LoopAffinityError` — the cached provider's
        clients are loop-bound.  A provider marked ``loop_agnostic`` (the
        in-process local providers) claims nothing and may be shared across
        loops.

        Use this to build eagerly at startup (see :meth:`preflight`); ordinary
        calls resolve the provider through :meth:`lease_provider`, which holds
        an in-flight lease across the ``await``.
        """
        if self.provider is not None:
            if self.on_event is not None:
                raise ValueError(
                    "ModelBinding.on_event applies only to a provider the binding builds; "
                    "with provider=..., pass on_event to that provider's constructor instead."
                )
            provider = self.provider
        else:
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
                    wire_shape=self.wire_shape,
                )
            provider = self._built
        if not self._is_loop_agnostic(provider):
            self._claim_loop()
        return provider

    def get_provider(self) -> Provider:
        """Compatibility alias for :meth:`build_provider`.

        Kept for 0.3.x callers; new code should call :meth:`build_provider`,
        which reads as the side-effecting, idempotent build it is.
        """
        return self.build_provider()

    def for_current_loop(self) -> ModelBinding:
        """Return a sibling binding spec for use on another event loop.

        The clone shares the model definition, ``params``, event handler,
        credential provider, and a caller-supplied provider, but starts with
        no loop claim and no cached provider — an application can keep one
        binding spec and use it from multiple loops/threads:

        * a supplied provider marked ``loop_agnostic`` (the in-process local
          providers) is shared as-is;
        * a binding that builds its provider builds a fresh one on the loop
          that first uses the clone — never share a loop-bound SDK client.

        A caller-supplied provider that owns loop-bound resources cannot be
        rebound: sharing it across loops is exactly what the loop guard
        prevents, so this raises :class:`ValueError` instead of letting the
        clone bypass it.  Build that provider inside the target loop, or
        create a new binding there.

        The original binding keeps its own claim and cached provider; each
        clone carries its own lease bookkeeping, so use one clone per thread
        when calls overlap concurrently.
        """
        if self.provider is not None and not self._is_loop_agnostic(self.provider):
            raise ValueError(
                f"ModelBinding for '{self.model}' wraps a caller-supplied provider that owns "
                "loop-bound resources; it cannot be rebound to another loop — build that "
                "provider inside the target loop or create a new binding there."
            )
        return replace(self, params=dict(self.params))

    @asynccontextmanager
    async def lease_provider(self) -> AsyncIterator[Provider]:
        """Yield the current provider while holding an in-flight lease.

        Every :class:`~cliver_core.agent.AgentCore` model call, capability
        call, and System One prediction holds one of these, so
        :meth:`arefresh` closes a leased-out provider at the first refresh
        after its last in-flight call finishes, instead of keeping it until
        :meth:`aclose`.
        Use this — not :meth:`build_provider` — whenever a provider is used
        across an ``await``::

            async with binding.lease_provider() as provider:
                response = await provider.chat(request)

        Leases are per binding and per event loop, like the provider itself.
        """
        provider = self.build_provider()
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

    async def preflight(self) -> None:
        """Build this binding's provider now, for startup checks.

        Idempotent: a provider this binding already built is returned as-is —
        nothing is rebuilt or closed.  A caller-supplied provider is used
        as-is; preflight only claims the calling loop when it is loop-bound.

        A ``credential_provider`` is resolved **once** for a binding that has
        not built its provider yet, and the provider is built with that
        credential.  If the build fails, the previous ``api_key`` is restored,
        so a failed preflight leaves the binding exactly as it was.  A binding
        that already built its provider skips credential resolution — rotate
        explicitly with :meth:`arefresh`.

        Raises the provider-construction error (missing ``base_url``, missing
        credential, invalid client options) or :class:`LoopAffinityError` when
        the binding was already claimed by another loop.  Lazy construction
        remains the default; call this only when startup validation pays off.
        """
        if self.provider is not None or self._built is not None:
            self.build_provider()
            return
        credential = self.api_key
        if self.credential_provider is not None:
            resolved = self.credential_provider()
            if inspect.isawaitable(resolved):
                resolved = await resolved
            if not isinstance(resolved, str) or not resolved.strip():
                raise ValueError(
                    f"credential_provider for '{self.model}' must return a non-empty string, got {resolved!r}"
                )
            credential = resolved
        if credential == self.api_key:
            self.build_provider()
            return
        previous = self.api_key
        self.api_key = credential
        try:
            self.build_provider()
        except BaseException:
            self.api_key = previous
            raise

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

        Runs on the owning loop; another loop raises
        :class:`LoopAffinityError` — unless every provider it holds is
        loop-agnostic, which may be refreshed from any loop.  There is no
        automatic refresh on a 401 — the application decides when to rotate
        (TTL timer, or catch :class:`ProviderError` and call this).
        """
        self._claim_loop_for_providers()
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
            # the next build_provider() builds with the new key.
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
            wire_shape=self.wire_shape,
        )
        previous, self._built = self._built, replacement
        self.api_key = credential
        self._retired.append(previous)
        await self._prune_retired()

    async def aclose(self) -> None:
        """Close the provider this binding built and every retired one.

        A caller-supplied provider stays open.  Must run on the loop the
        binding was used on — unless every provider it holds is loop-agnostic,
        in which case any loop may close it.  Releases the loop claim when the
        binding can safely be reused: its built provider is now closed, so the
        next loop builds a fresh client, and a loop-agnostic supplied provider
        never claimed one.  A caller-supplied loop-bound provider keeps the
        claim — it stays open on its original loop, so reusing the binding on
        another loop would hand a live SDK client to the wrong loop.  In-flight
        leases are not awaited here — the agent drains its work before closing
        its bindings; a provider closed under a still-running call is the
        caller's contract violation, exactly as before.
        """
        self._claim_loop_for_providers()
        if self._built is not None:
            built, self._built = self._built, None
            await built.close()
        retired, self._retired = self._retired, []
        for provider in retired:
            await provider.close()
        self._leases.clear()
        if self.provider is None or self._is_loop_agnostic(self.provider):
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

    async def preflight(self) -> None:
        """Build every configured provider on the calling loop.

        The startup check: every primary and fallback binding builds its
        provider now, resolving a credential source once for bindings that
        have not built yet, so a missing endpoint or credential fails before
        traffic is served instead of on the first request.  Idempotent — a
        binding that already built its provider is not rebuilt or closed.

        Purposes are visited in :data:`PURPOSES` order and bindings in mount
        order; the first failure stops the pass, named with its purpose and
        model.  Bindings already built stay usable.  Lazy construction remains
        the default; call this explicitly at startup (``agent_lifespan`` does).
        """
        for purpose in self.bound_purposes():
            for binding in self.get_all(purpose):
                try:
                    await binding.preflight()
                except LoopAffinityError as exc:
                    raise LoopAffinityError(
                        f"bindings.{purpose} ('{binding.model}') failed preflight: {exc}",
                        model=exc.model,
                        owner_loop=exc.owner_loop,
                        current_loop=exc.current_loop,
                    ) from exc
                except ValueError as exc:
                    raise ValueError(f"bindings.{purpose} ('{binding.model}') failed preflight: {exc}") from exc

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
