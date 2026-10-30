"""Provider interface and request/response models."""

import base64
import inspect
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, AsyncIterator, Awaitable, Callable, Union

from pydantic import BaseModel, Field

from cliver_core.errors import CapabilityNotSupported, ProviderError
from cliver_core.media import (
    MediaContent,
    MediaType,
    extension_from_mime,
    media_type_from_mime,
    resolve_media_mime,
)
from cliver_core.messages import CLIverMessage, CLIverMessageChunk, UsageInfo
from cliver_core.tool import CLIverTool

if TYPE_CHECKING:
    from cliver_core.provider.transport import BrandSDK

logger = logging.getLogger(__name__)

#: HTTP statuses that conventionally mean "try again later".
_RETRYABLE_STATUS = frozenset({408, 409, 425, 429})
#: Class-name markers for SDK failures without a usable status code.
_RETRYABLE_NAME_MARKERS = (
    "ratelimit",
    "timeout",
    "connect",
    "internalserver",
    "serviceunavailable",
    "overloaded",
    "unavailable",
)


def _is_retryable_error(exc: BaseException) -> bool:
    """Best-effort retryability classification for vendor exceptions."""
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(exc, "status", None)
    if isinstance(status, int):
        return status in _RETRYABLE_STATUS or status >= 500
    name = type(exc).__name__.lower()
    return any(marker in name for marker in _RETRYABLE_NAME_MARKERS)


def _stream_requested(value: Any) -> bool:
    """Wire truthiness of a ``stream`` option.

    Vendor samples send the flag as the *string* ``"false"``, so only a real
    ``True`` or an affirmative string counts as a streaming request.
    """
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes", "on")
    return bool(value)


class CLIverRequest(BaseModel):
    """A request to an LLM provider.

    Media (images/audio/video) is embedded into the user message content
    as content blocks by AgentCore before the request is built.
    """

    messages: list[CLIverMessage]
    tools: list[CLIverTool] | None = None
    model: str
    options: dict[str, Any] = Field(default_factory=dict)
    # temperature, top_p, max_tokens, thinking, etc.
    # Passed through — each provider filters what it supports.


class CLIverResponse(BaseModel):
    """A response from an LLM provider."""

    message: CLIverMessage
    media: list[MediaContent] | None = None
    # Generated or returned media files (images, audio, video).
    usage: UsageInfo | None = None
    # The final model call's usage.  AgentCore keeps this meaning unchanged;
    # a Re-Act run's aggregate lives on ``usage_total`` instead.
    usage_total: UsageInfo | None = None
    # Sum of every model call in the run that produced this response —
    # intermediate tool-requesting iterations included.  ``None`` when no
    # call reported usage.
    request_info: dict[str, Any] | None = None
    # Diagnostics echo of the request that produced this response
    # (model, provider, options, system_prompt, messages, tools) —
    # AgentCore collects it into the Re-Act trace.


#: ``generate_media(output_handler=...)`` — a sync or async callable that turns
#: the raw media payload (the parsed JSON body, or bytes when the request asked
#: for a binary response) into the final :class:`CLIverResponse`, replacing the
#: default output handling entirely.
MediaOutputHandler = Callable[[Any], Union[CLIverResponse, Awaitable[CLIverResponse]]]


class RerankItem(BaseModel):
    """One scored document from a rerank call."""

    index: int
    relevance_score: float
    document: str | None = None


class EmbeddingsResult(BaseModel):
    """Vectors from an embeddings call, ordered to match the input texts."""

    vectors: list[list[float]]
    model: str
    usage: UsageInfo | None = None


class Provider(ABC):
    """Interface for LLM inference.

    A Provider wraps a specific brand (DeepSeek, MiniMax, OpenAI, ...) and
    speaks ONE wire protocol.  The protocol providers (``OpenAIProvider`` /
    ``AnthropicProvider``) own the protocol shape — transport I/O, request
    assembly, response extraction; brand classes subclass one of them and
    add behavior through hook overrides.

    Every concrete provider is protocol-locked: it declares ``protocol``
    (the ONE wire protocol it speaks) and ``brand`` (the short brand key
    used for transport resolution and logging).  Brands serving two
    endpoints (e.g. GLM) split into two classes —
    ``GLMProvider`` / ``GLMAnthropicProvider`` — each locked to its
    protocol (quirks shared through a mixin base where useful).  The check runs
    at class-definition time: an intermediate base opts out by declaring
    ``protocol = ""``; a leaf resolving to a falsy protocol raises.
    """

    protocol: str = ""
    brand: str = ""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # A literal ``protocol = ""`` in the class's own __dict__ is the
        # explicit intermediate opt-out (e.g. _MiniMaxBase).
        # Any other class that still resolves to a falsy protocol is a leaf
        # that forgot to lock its protocol — caught at import, not at the
        # first API call.
        if "protocol" in cls.__dict__:
            return  # declared here: "" = explicit intermediate opt-out, else the lock
        if not cls.protocol:
            raise TypeError(
                f"{cls.__name__} must define protocol ('openai' or 'anthropic'); "
                f'declare protocol = "" only on intermediate mixin bases'
            )

    def __init__(self, api_key: str, base_url: str, *, logger: "logging.Logger | None" = None):
        self.api_key = api_key
        self.base_url = base_url
        self.logger = logger or logging.getLogger(__name__)
        #: The brand's I/O pipe — set by the protocol providers; media
        #: generation sends through it (:meth:`_post_media`).
        self.transport: BrandSDK | None = None

    @classmethod
    def provider_name(cls) -> str:
        """Short brand key for logging/registration."""
        return cls.brand

    # ── Hooks (subclasses override for brand-specific behavior) ──

    def on_response(self, response: CLIverResponse) -> CLIverResponse:
        """Post-process a response. Override for brand-specific reshaping;
        clear ``message.native`` when the fields no longer derive from it."""
        return response

    def on_chunk(self, chunk: CLIverMessageChunk) -> CLIverMessageChunk:
        """Post-process a streaming chunk. Override to remap vendor_ext keys."""
        return chunk

    def filter_options(self, options: dict[str, Any]) -> dict[str, Any]:
        """Filter/transform options before they are assembled into the request body."""
        return options

    # ── Public API ─────────────────────────────────────────

    @abstractmethod
    async def chat(self, request: CLIverRequest) -> CLIverResponse: ...

    @abstractmethod
    async def stream(self, request: CLIverRequest) -> AsyncIterator[CLIverMessageChunk]: ...

    async def close(self) -> None:
        """Release provider-held resources.

        Overridden by providers that hold a client (the protocol providers
        close their transport); the default is a no-op.
        """
        return

    # ── Transport I/O ───────────────────────────────────────

    async def _transport_chat(self, body: dict[str, Any]) -> dict[str, Any]:
        """Send one chat request, translating transport failures to ``ProviderError``.

        Vendor exceptions keep the original as ``__cause__`` so callers can
        inspect the SDK error while mapping on ``retryable`` alone.  An
        already-translated :class:`ProviderError` passes through unchanged.
        """
        if self.transport is None:
            raise ProviderError(f"{self.provider_name()} has no transport")
        try:
            return await self.transport.chat(body)
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(str(exc), retryable=_is_retryable_error(exc)) from exc

    async def _transport_stream(self, body: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        """Iterate the transport stream, translating failures to ``ProviderError``.

        Only the transport's own iteration is wrapped — exceptions raised by
        the consumer's loop body (e.g. an event handler) never pass through
        this generator's handler.
        """
        if self.transport is None:
            raise ProviderError(f"{self.provider_name()} has no transport")
        try:
            async for data in self.transport.stream(body):
                yield data
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(str(exc), retryable=_is_retryable_error(exc)) from exc

    # ── Media generation (transport-backed, shared across providers) ──

    # Subclasses override these to configure media generation endpoints.
    # Keys: "image", "tts", "video", ...  Values: full URL or path appended
    # to ``self.base_url`` when the value starts with "/".
    _media_gen_urls: dict[str, str] = {}

    #: When False, media purposes missing from ``_media_gen_urls`` report
    #: unsupported instead of falling back to the protocol's standard media
    #: paths — for brands whose media endpoints deviate from the wire
    #: standard entirely (e.g. MiniMax, which serves ``/image_generation``
    #: on its own domain) or that serve no media at all (DeepSeek, Ollama).
    #: Embed/rerank are not gated: they keep their protocol defaults even on
    #: media-less brands (e.g. Ollama's working ``/embeddings``).
    _standard_media_fallback: bool = True

    #: The purposes ``_standard_media_fallback`` gates.
    _MEDIA_PURPOSES: frozenset[str] = frozenset({"image", "tts", "video", "asr"})

    #: Purposes whose response payload is raw bytes, not JSON (per protocol).
    _BINARY_MEDIA_PURPOSES: frozenset[str] = frozenset()

    #: Option keys a transcription routes to HTTP headers instead of multipart
    #: form fields, per brand (MiniMax takes ``language`` as a BCP-47 header).
    #: Empty (the default) = every option is a form field (OpenAI fashion).
    _TRANSCRIBE_HEADER_OPTIONS: tuple[str, ...] = ()

    #: User-facing labels for purposes whose key is not the plain word.
    _UNSUPPORTED_MEDIA_LABELS: dict[str, str] = {"asr": "audio transcription"}

    #: purpose → (media kind, mime type, file extension) for the shared flow.
    _MEDIA_KINDS: dict[str, tuple[MediaType, str, str]] = {
        "image": (MediaType.IMAGE, "image/png", "png"),
        "tts": (MediaType.AUDIO, "audio/mpeg", "mp3"),
        "audio": (MediaType.AUDIO, "audio/mpeg", "mp3"),  # legacy alias of tts
        "video": (MediaType.VIDEO, "video/mp4", "mp4"),
    }

    #: Audio ``response_format`` → (mime type, file extension); unknown
    #: formats fall back to ``audio/<format>`` / ``<format>``.
    _AUDIO_FORMAT_KINDS: dict[str, tuple[str, str]] = {
        "mp3": ("audio/mpeg", "mp3"),
        "wav": ("audio/wav", "wav"),
        "flac": ("audio/flac", "flac"),
        "pcm": ("audio/pcm", "pcm"),
        "aac": ("audio/aac", "aac"),
        "opus": ("audio/opus", "opus"),
    }

    def _media_kind(
        self, purpose: str, options: dict[str, Any], *, mime: str | None = None
    ) -> tuple[MediaType, str, str]:
        """Resolve (kind, mime, extension) for the shared HTTP media flow.

        An explicit *mime* (``generate_media(mime=...)``) labels the output of
        purposes outside :attr:`_MEDIA_KINDS` and wins over the purpose
        default.  For TTS a requested ``response_format`` wins over the static
        default, so ``response_format="wav"`` writes ``generated_0.wav`` with
        ``audio/wav`` instead of mislabeling the bytes ``.mp3``.
        """
        if mime:
            return (media_type_from_mime(mime), mime, extension_from_mime(mime))
        default = self._MEDIA_KINDS.get(purpose, (MediaType.FILE, "application/octet-stream", "bin"))
        if purpose != "tts":
            return default
        fmt = options.get("response_format")
        if not isinstance(fmt, str) or not fmt:
            return default
        item_mime, ext = self._AUDIO_FORMAT_KINDS.get(fmt.lower(), (f"audio/{fmt.lower()}", fmt.lower()))
        return (MediaType.AUDIO, item_mime, ext)

    # Per-purpose request paths by wire protocol — the standard wire fashion
    # (OpenAI-style endpoint names).  Brand classes override individual
    # purposes via ``_media_gen_urls``; protocols without media endpoints
    # (e.g. ``anthropic``) have no entry and report the purpose unsupported.
    _DEFAULT_MEDIA_PATHS: dict[str, dict[str, str]] = {
        "openai": {
            "image": "/images/generations",
            "tts": "/audio/speech",
            "asr": "/audio/transcriptions",
            "video": "/videos/generations",
            "embed": "/embeddings",
            "rerank": "/rerank",
        },
    }

    def media_path(self, purpose: str, override: str | None = None) -> str | None:
        """Resolve the request path for one media *purpose*.

        Resolution order (first hit wins):

        1. *override* — explicit ``path=`` from the model binding (or a
           provider-call option);
        2. ``_media_gen_urls[purpose]`` — the brand map on the provider class
           (e.g. MiniMax image → ``/image_generation``);
        3. ``_DEFAULT_MEDIA_PATHS[protocol][purpose]`` — the standard
           OpenAI-fashion path for the wire protocol.

        Returns ``None`` when no layer has a path for the purpose — the
        provider/protocol does not support it.  The legacy ``"audio"`` alias
        normalizes to ``"tts"``.  The ``_standard_media_fallback`` opt-out
        only gates media purposes (``image``/``tts``/``video``/``asr``) —
        embed/rerank always keep the protocol's default path.
        """
        purpose = "tts" if purpose == "audio" else purpose
        if override:
            return override
        if purpose in self._media_gen_urls:
            return self._media_gen_urls[purpose]
        if purpose in self._MEDIA_PURPOSES and not self._standard_media_fallback:
            return None
        return self._DEFAULT_MEDIA_PATHS.get(self.protocol, {}).get(purpose)

    def _build_media_gen_body(
        self,
        prompt: str,
        model: str,
        purpose: str,
        *,
        media: list[MediaContent] | None = None,
        **options,
    ) -> dict:
        """Build the request body for media generation.

        Override to customise field names or add provider-specific params
        (*media* carries reference files for image editing / variation).
        The default uses ``{"model": ..., "prompt": ...}`` which works for
        most OpenAI-compatible image/audio/video APIs.
        """
        return {"model": model, "prompt": prompt, **options}

    def _build_media_gen_headers(self) -> dict[str, str]:
        """Build HTTP headers for media generation requests."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _check_media_response_error(self, data: dict, purpose: str) -> str | None:
        """Check for provider-specific error envelopes in the JSON body.

        Some providers (e.g. MiniMax) return HTTP 200 with an error code
        inside the response body.  Override this to extract and return
        the error message, or ``None`` if the response is genuinely
        successful.
        """
        return None

    def _extract_media_items(self, data: dict, purpose: str) -> list[dict]:
        """Extract raw media items from a provider's JSON response.

        Each returned dict should have at least one of ``url`` or
        ``b64_json``.  Override this to handle provider-specific response
        shapes (nested wrappers, different field names, etc.).

        The default handles ``{"data": [{"url": "...", ...}]}`` — the
        OpenAI / MiniMax / compatible format.
        """
        raw = data.get("data", data)
        if isinstance(raw, dict):
            raw = [raw]
        if not isinstance(raw, list):
            return []
        return [item for item in raw if isinstance(item, dict)]

    async def _await_media_result(self, raw: Any, purpose: str, url: str, options: dict[str, Any]) -> Any:
        """Resolve an async vendor's job object into its finished payload.

        Some vendors answer the create POST with a *task id* — the result
        only exists once the task finishes (Seedance video).  This is the
        seam where such a vendor waits; the default is **inert** (returns
        *raw* unchanged), so one-shot vendors are byte-identical.

        *url* is the create endpoint exactly as POSTed (a poller derives the
        task URL from it, honoring a call-site ``path=``), and *options* the
        call's request options: the body was already built from a copy, so a
        poller may consume its own keys (e.g. ``poll_interval``) here without
        touching the wire.  Whatever is returned reaches output handling —
        *output_handler*, or the envelope check + :meth:`_extract_media_items`.
        """
        return raw

    async def _prepare_media_body(self, body: dict[str, Any], purpose: str, *, model: str) -> dict[str, Any]:
        """Rewrite one media request body before it is POSTed (default: unchanged).

        Async so a brand can upload local bytes to vendor storage first
        (DashScope accepts reference images only as public or ``oss://``
        URLs, never as data URIs).  *body* is the freshly built request dict;
        return the dict to send.  The default is **inert** — the body is
        returned unchanged — so one-shot vendors stay byte-identical.
        """
        return body

    def _media_request_headers(self, purpose: str, body: dict[str, Any]) -> dict[str, str]:
        """Extra per-request headers for one media create POST (default: none).

        A brand hook for vendors that take a media request option as an HTTP
        header (DashScope's ``X-DashScope-Async``).  *body* is the body that
        will be POSTed — after :meth:`_prepare_media_body` rewrote it — so a
        brand can key headers off its final content (e.g. the OSS resolver
        flag only when an ``oss://`` URL is present).  Headers returned here
        reach the transport through :meth:`_post_media`, on top of its own
        defaults; the default adds nothing, keeping other vendors unchanged.
        """
        return {}

    def _resolve_media_gen_url(self, purpose: str, override: str | None = None) -> str | None:
        """Resolve the URL for a media generation endpoint.

        Path lookup goes through :meth:`media_path` (*override* → brand map →
        standard protocol default), then the result is resolved against
        ``self.base_url`` (the model's configured ``api_url``):

        - ``"https://..."`` → as-is (absolute URL, bypasses base_url)
        - ``"/path"``       → ``base_url + "/path"``
        - ``""``            → ``base_url`` (model api_url IS the full endpoint)
        - no path anywhere  → ``None`` (media type not supported)
        """
        path = self.media_path(purpose, override)
        if path is None:
            return None
        if not path:
            return self.base_url or None
        # Absolute URL (e.g. different domain) — use as-is, ignore base_url
        if path.startswith(("http://", "https://")):
            return path
        if path.startswith("/") and self.base_url:
            return self.base_url.rstrip("/") + path
        if self.base_url:
            return self.base_url.rstrip("/") + "/" + path.lstrip("/")
        return path

    async def _post_media(
        self,
        url: str,
        body: dict,
        *,
        files: dict | None = None,
        binary: bool = False,
        headers: dict | None = None,
    ):
        """POST one media request through the transport — the single sender seam.

        Media traffic is transport I/O exactly like chat: the URL is the
        absolute one :meth:`_resolve_media_gen_url` resolved (both bundled
        SDK transports pass absolute URLs through verbatim), and brand
        differences live in the body/response hooks — never in a second
        HTTP stack.

        *headers* carries per-request extras for brands that take an option
        as an HTTP header (e.g. MiniMax's transcription ``language``); it is
        passed to the transport only when non-empty, so transports on the
        older four-argument ``post`` signature keep working.
        """
        if self.transport is None:
            raise RuntimeError(
                f"{self.provider_name()} has no transport — media generation requires a protocol provider"
            )
        if headers:
            return await self.transport.post(url, body, files=files, binary=binary, headers=headers)
        return await self.transport.post(url, body, files=files, binary=binary)

    async def generate_media(
        self,
        prompt: str,
        *,
        model: str,
        purpose: str = "image",
        media: list[MediaContent] | None = None,
        output_dir: str | None = None,
        path: str | None = None,
        mime: str | None = None,
        binary: bool | None = None,
        output_handler: MediaOutputHandler | None = None,
        **options: Any,
    ) -> CLIverResponse:
        """Generate any media kind — the public, extensible media pipeline.

        Resolves the endpoint URL (*purpose* → *path* → brand map → protocol
        default), POSTs through the provider's transport, and turns the
        payload into :class:`MediaContent` items (saved to *output_dir*).  The
        four built-in kinds are this same seam behind their shorthand
        methods — ``provider.image(prompt, model)`` is exactly
        ``provider.generate_media(prompt, model=model, purpose="image")``
        (likewise ``video()`` / ``tts()`` / ``asr()``).

        *options* pass through into the request body verbatim — there is no
        fixed option list; whatever the vendor's API documents is what you
        pass.  The response is expected in the OpenAI fashion,
        ``{"data": [{"url": ... | "b64_json": ...}]}``; a vendor with a
        different shape is supported by overriding the hooks in a provider
        subclass (:meth:`_build_media_gen_body`, :meth:`_extract_media_items`,
        :meth:`_check_media_response_error`, :meth:`_await_media_result`,
        ``_media_gen_urls``) — or by taking over output handling completely
        with *output_handler*.

        A purpose with no resolvable endpoint raises
        :class:`CapabilityNotSupported` before anything reaches the wire;
        runtime failures return a friendly :class:`CLIverResponse`.

        Common media kinds::

            # image — built-in purposes need no path= at all
            await provider.image("a cat on a mat", "image-01",
                                 size="1024x1024", n=2, output_dir="out/")
            # image editing — the reference rides media=
            await provider.image("make it watercolor", "image-01",
                                 media=[load_media_file("cat.png")])

            # video
            await provider.generate_media("a sunset over the sea", model="video-01",
                                          purpose="video", duration=6, resolution="1080p")

            # text-to-speech
            await provider.tts("hello world", "speech-2.5",
                               voice="English_expressive_narrator", output_dir="out/")

            # speech-to-text — a reserved purpose: multipart in, transcript out
            await provider.asr(load_media_file("talk.mp3"), "whisper-1", language="en")

            # music — a kind the core does not model: pick a purpose name,
            # point path= at the endpoint, pass the vendor's parameters
            await provider.generate_media("calm piano loop", model="music-1",
                                          purpose="music", path="/v1/music/generate",
                                          duration=30, output_format="mp3", mime="audio/mpeg")

            # sound effects
            await provider.generate_media("thunder crack", model="sfx-1",
                                          purpose="sfx", path="/v1/sfx", duration=3)

            # a 3-D mesh — a bytes-out endpoint: binary=True brings the raw
            # payload, mime= labels it
            await provider.generate_media("a low-poly teapot", model="mesh-1",
                                          purpose="3d", path="/v1/meshes", format="glb",
                                          mime="model/gltf-binary", binary=True, output_dir="out/")

            # full output handover — only a successful POST (resolved through
            # the async-vendor hook) reaches the handler; it receives the raw
            # payload (parsed JSON, or bytes with binary=True) and owns
            # everything after that
            def pick_track(body: dict) -> CLIverResponse:
                track = body["tracks"][0]
                return CLIverResponse(
                    message=CLIverMessage(role="assistant", content=track["title"]),
                    media=[MediaContent(type=MediaType.AUDIO, data=track["data"], mime_type="audio/mpeg")],
                )

            await provider.generate_media("calm piano loop", model="music-1",
                                          purpose="music", path="/v1/music/generate",
                                          output_handler=pick_track)

        Output knobs: *mime* is the label of last resort — payload magic bytes
        or the URL's own format still win (:func:`resolve_media_mime`).
        *binary* asks the transport for raw bytes (default: the protocol's
        ``_BINARY_MEDIA_PURPOSES`` policy).  *output_handler* must return a
        :class:`CLIverResponse` (sync or async); nothing else touches the
        payload then — no error envelope check, no extraction, no mime
        resolution, no saving.  All three are generation-only:
        ``purpose="asr"`` rejects them with ``ValueError``.

        ``timeout`` is consumed for call-site compatibility (the transport's
        client owns request timeouts) so it never leaks into the body.
        """
        # The legacy alias, normalized before path resolution — media_path()
        # does the same, and everything downstream keys on "tts".
        purpose = "tts" if purpose == "audio" else purpose
        options.pop("timeout", None)
        url = self._resolve_media_gen_url(purpose, path)
        if not url:
            label = self._UNSUPPORTED_MEDIA_LABELS.get(purpose, purpose)
            raise CapabilityNotSupported(
                f"{self.provider_name()} does not support {label}. "
                f"Pass path= to the call or set it on the model binding "
                f"(ModelBinding.<purpose>(path=...)), add a '{purpose}' entry to "
                f"_media_gen_urls on the provider class, or set api_url on the model config."
            )

        if purpose == "asr":
            # Raised here, before the generation try/except, so the guard is
            # never swallowed into a friendly failure response.
            knobs = {"mime": mime, "binary": binary, "output_handler": output_handler}
            generation_only = [name for name, value in knobs.items() if value is not None]
            if generation_only:
                raise ValueError(
                    f"{', '.join(generation_only)}= is generation-only and does not apply to "
                    "transcription (purpose='asr') — use response_format for the transcript shape"
                )
            if not isinstance(media, MediaContent):
                raise TypeError("asr() expects a MediaContent instance")
            return await self._transcribe(media, model, url, options)

        try:
            return await self._generate_media_items(
                prompt,
                model=model,
                purpose=purpose,
                media=media,
                output_dir=output_dir,
                url=url,
                options=options,
                mime=mime,
                binary=binary,
                output_handler=output_handler,
            )
        except Exception as e:
            self.logger.warning("%s %s generation failed: %s", self.provider_name(), purpose, e)
            return self._media_failure_response(purpose, e)

    async def _generate_media_items(
        self,
        prompt: str,
        *,
        model: str,
        purpose: str,
        media: list[MediaContent] | None,
        output_dir: str | None,
        url: str,
        options: dict[str, Any],
        mime: str | None = None,
        binary: bool | None = None,
        output_handler: MediaOutputHandler | None = None,
    ) -> CLIverResponse:
        """One non-ASR media request: build the body, POST it, wrap the payload."""
        from pathlib import Path

        resolved_binary = purpose in self._BINARY_MEDIA_PURPOSES if binary is None else binary
        body = self._build_media_gen_body(prompt, model, purpose, media=media, **options)
        # Async brands may need pre-POST work (e.g. uploading local reference
        # bytes to vendor storage); the default hook returns the body as-is.
        body = await self._prepare_media_body(body, purpose, model=model)
        # Per-request headers ride the transport on top of its own defaults;
        # the default hook returns none, so other vendors POST unchanged.
        headers = self._media_request_headers(purpose, body)

        raw = await self._post_media(url, body, binary=resolved_binary, headers=headers or None)
        # Async vendors (create → task id → poll) resolve their job object
        # here; the default hook is a passthrough, so one-shot vendors are
        # byte-identical.  Safe to consume options: the body above got a copy.
        raw = await self._await_media_result(raw, purpose, url, options)

        # A caller-supplied handler sees the raw payload — parsed JSON, or the
        # bytes we just asked the transport for — and owns everything after
        # the POST.  It runs inside the caller's try/except, so a raising
        # handler surfaces as the friendly failure response, like any other
        # pipeline error.
        if output_handler is not None:
            result = output_handler(raw)
            if inspect.isawaitable(result):
                result = await result
            if not isinstance(result, CLIverResponse):
                raise TypeError(f"output_handler must return a CLIverResponse, got {type(result).__name__}")
            return result

        # Resolved after the POST: the body builder above received a fresh dict
        # (``**options``), so reading the options here never mutates the request.
        kind, item_default_mime, ext = self._media_kind(purpose, options, mime=mime)

        if resolved_binary:
            raw_items: list[dict] = [{"b64_json": base64.b64encode(raw).decode()}] if raw else []
        else:
            # Check for provider-specific error envelopes in the JSON body
            # (e.g. MiniMax returns HTTP 200 with base_resp.status_code ≠ 0).
            err = self._check_media_response_error(raw, purpose)
            if err:
                self.logger.warning("%s %s API error: %s", self.provider_name(), purpose, err)
                return self._media_failure_response(purpose, err)
            # Let the provider extract raw items from its JSON response.
            raw_items = self._extract_media_items(raw, purpose)
            if not raw_items:
                self.logger.warning(
                    "%s %s returned no media items. Raw response: %s",
                    self.provider_name(),
                    purpose,
                    raw,
                )

        # Build MediaContent — same pattern for ALL providers.  A vendor
        # generates what it generates (MiniMax image-01 returns JPEG, not the
        # flow's PNG default), so each item's real format beats the purpose
        # default: magic bytes when the payload is inline, else the format
        # the URL names for itself.
        media_items: list[MediaContent] = []
        for item in raw_items:
            item_url = item.get("url")
            data = item_url or item.get("b64_json") or ""
            if not data:
                continue
            payload = None
            if not item_url and kind is MediaType.IMAGE:
                try:
                    payload = base64.b64decode(data)
                except (ValueError, TypeError):
                    payload = None
            item_mime = resolve_media_mime(item_default_mime, url=item_url, payload=payload)
            media_items.append(MediaContent(type=kind, data=data, mime_type=item_mime))

        # Save to disk — same pattern for ALL providers.
        if output_dir and media_items:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            for i, mc in enumerate(media_items):
                if mc.mime_type != item_default_mime:  # the item's real format won over the default
                    item_ext = mc.get_file_extension().removeprefix(".")
                else:
                    item_ext = ext
                mc.save(out / f"generated_{i}.{item_ext}")

        content = f"Generated {len(media_items)} {purpose}(s) via {self.provider_name()}."
        saved = [mc.saved_path for mc in media_items if mc.saved_path]
        if saved:
            content += "\n" + "\n".join(f"- {p}" for p in saved)
        return CLIverResponse(
            message=CLIverMessage(role="assistant", content=content),
            media=media_items if media_items else None,
        )

    async def _transcribe(
        self,
        media: MediaContent,
        model: str,
        url: str,
        options: dict[str, Any],
    ) -> CLIverResponse:
        """One multipart transcription request — text out, friendly on failure.

        ``stream`` is consumed, not forwarded: SSE transcripts are not part
        of the one-shot pipeline, so a truthy value is rejected before the
        wire (a falsy one, including the vendor samples' ``"false"`` string,
        just keeps the vendor's non-streaming default).  Options listed in
        ``_TRANSCRIBE_HEADER_OPTIONS`` ride HTTP headers instead of the
        multipart body (MiniMax's ``language``); everything else, including
        the OpenAI-fashion ``language`` form field, rides the body.  A
        ``response_format`` of ``text`` / ``srt`` / ``vtt`` returns a plain
        text body — requested as raw bytes and decoded.
        """
        if _stream_requested(options.pop("stream", None)):
            raise ValueError(
                "streaming transcription is not supported — remove stream=True "
                "(the pipeline reads the final transcript in one shot)"
            )

        headers: dict[str, str] = {}
        for key in self._TRANSCRIBE_HEADER_OPTIONS:
            value = options.pop(key, None)
            if value:  # Provider.asr() always passes language=None down
                headers[key] = value

        language = options.pop("language", None)
        filename = media.filename or f"audio{media.get_file_extension()}"
        response_format = options.get("response_format")
        plain_text = isinstance(response_format, str) and response_format.lower() in ("text", "srt", "vtt")
        body: dict[str, Any] = {"model": model, **options}
        if language:
            body["language"] = language
        files = {"file": (filename, media.to_bytes(), media.mime_type)}

        try:
            result = await self._post_media(url, body, files=files, binary=plain_text, headers=headers)
        except Exception as e:
            self.logger.warning("%s asr API error: %s", self.provider_name(), e)
            return self._media_failure_response("asr", e)

        if isinstance(result, (bytes, bytearray)):
            text = bytes(result).decode("utf-8", errors="replace")
        elif isinstance(result, dict):
            text = result.get("text", "")
        else:
            text = str(result)
        return CLIverResponse(
            message=CLIverMessage(role="assistant", content=text),
        )

    def _media_failure_response(self, purpose: str, error: Any) -> CLIverResponse:
        """Friendly failure response for a media call that reached the wire."""
        label = self._UNSUPPORTED_MEDIA_LABELS.get(purpose, purpose)
        return CLIverResponse(
            message=CLIverMessage(
                role="assistant",
                content=f"{self.provider_name()} {label} generation failed: {error}",
            ),
        )

    async def image(
        self,
        prompt: str,
        model: str,
        *,
        media: list[MediaContent] | None = None,
        output_dir: str | None = None,
        **options,
    ) -> CLIverResponse:
        """Generate images — POST to the resolved image endpoint.

        Example::

            result = await provider.image("a cat on a mat", "image-01",
                                          size="1024x1024", n=2, output_dir="out/")
            # image editing — the reference rides media=
            result = await provider.image("make it watercolor", "image-01",
                                          media=[load_media_file("cat.png")])

        The endpoint path resolves via ``path=`` (call-site option, model
        binding, or brand map) → protocol default; a purpose with no path
        raises :class:`CapabilityNotSupported`.  All other *options* pass
        into the request body verbatim.  See :meth:`generate_media` for the
        full pipeline (custom kinds, ``mime=`` / ``binary=`` /
        ``output_handler=``).
        """
        return await self.generate_media(
            prompt, model=model, purpose="image", media=media, output_dir=output_dir, **options
        )

    async def video(
        self,
        prompt: str,
        model: str,
        *,
        media: list[MediaContent] | None = None,
        output_dir: str | None = None,
        **options,
    ) -> CLIverResponse:
        """Generate video — POST to the resolved video endpoint.

        Example::

            result = await provider.video("a sunset over the sea", "video-01",
                                          duration=6, resolution="1080p", output_dir="out/")

        *media* is accepted for a uniform capability-call shape; the wire
        video endpoint takes the prompt only.  See :meth:`generate_media`.
        """
        return await self.generate_media(
            prompt, model=model, purpose="video", media=media, output_dir=output_dir, **options
        )

    async def tts(
        self,
        text: str,
        model: str,
        *,
        output_dir: str | None = None,
        **options,
    ) -> CLIverResponse:
        """Synthesize speech — POST to the resolved TTS endpoint.

        Example::

            result = await provider.tts("hello world", "speech-2.5",
                                        voice="English_expressive_narrator", output_dir="out/")
            # response_format drives the label: generated_0.wav, audio/wav
            result = await provider.tts("hello world", "speech-2.5", response_format="wav")

        The audio lands as one :class:`MediaContent` item (default MIME
        ``audio/mpeg``).  See :meth:`generate_media`.
        """
        return await self.generate_media(text, model=model, purpose="tts", output_dir=output_dir, **options)

    async def asr(
        self,
        media: MediaContent,
        model: str,
        *,
        language: str | None = None,
        **options,
    ) -> CLIverResponse:
        """Transcribe audio — multipart POST to the resolved transcription endpoint.

        Example::

            result = await provider.asr(load_media_file("talk.mp3"), "whisper-1", language="en")
            print(result.message.content)

        ``language`` rides the multipart body in the OpenAI fashion, or an
        HTTP header on providers that take it there (e.g. MiniMax); a
        ``response_format`` of ``srt`` / ``vtt`` puts the timed text in
        ``.message.content``.  ``stream=True`` raises ``ValueError`` —
        streaming transcription is not implemented.  This is the reserved
        ``purpose="asr"`` face of :meth:`generate_media`, so the
        generation-only knobs (``mime=`` / ``binary=`` / ``output_handler=``)
        do not apply here.
        """
        return await self.generate_media(media, model=model, purpose="asr", media=media, language=language, **options)

    # ── Rerank / embeddings (generic HTTP, shared across providers) ────

    async def _post_json_endpoint(self, path: str, body: dict, *, label: str, timeout: float = 120.0) -> dict:
        """POST JSON to ``base_url + path`` and return the parsed response.

        Raises RuntimeError on transport errors or non-2xx responses.
        """
        if not self.base_url:
            raise RuntimeError(f"{self.provider_name()} has no base_url configured for {label}")
        url = self.base_url.rstrip("/") + path
        import httpx

        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, json=body, headers=self._build_media_gen_headers())
        if not resp.is_success:
            raise RuntimeError(
                f"{self.provider_name()} {label} request failed (HTTP {resp.status_code}): {resp.text[:1000]}"
            )
        return resp.json()

    async def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        model: str,
        top_k: int | None = None,
        **options,
    ) -> list[RerankItem]:
        """Score *documents* against *query* via the provider's rerank endpoint.

        ``POST {base_url}{path}`` speaking the common Cohere/Jina/GLM-compatible
        shape.  The path resolves via :meth:`media_path` (``path=`` override →
        brand map → standard protocol default, ``/rerank`` for OpenAI-style
        wire).  Override for brands that differ.  Raises
        :class:`CapabilityNotSupported` when the protocol has no rerank path;
        RuntimeError on transport or API errors.
        """
        path = self.media_path("rerank", options.pop("path", None))
        if path is None:
            raise CapabilityNotSupported(
                f"{self.provider_name()} does not support rerank for protocol '{self.protocol}' — "
                f"pass path=..., or add a 'rerank' entry to _media_gen_urls on the provider class."
            )
        body: dict[str, Any] = {"model": model, "query": query, "documents": documents, **options}
        if top_k is not None:
            body["top_k"] = top_k
        data = await self._post_json_endpoint(path, body, label="rerank")

        raw = data.get("results") or data.get("data") or []
        items: list[RerankItem] = []
        for i, item in enumerate(raw):
            if not isinstance(item, dict):
                continue
            index = item.get("index", i)
            document = item.get("document")
            if isinstance(document, dict):
                document = document.get("text")
            if not isinstance(document, str) and isinstance(index, int) and 0 <= index < len(documents):
                document = documents[index]
            score = item.get("relevance_score", item.get("score", 0.0))
            items.append(RerankItem(index=index, relevance_score=float(score or 0.0), document=document))
        if top_k is not None:
            items = items[:top_k]
        return items

    async def embed(self, texts: list[str], *, model: str, **options) -> EmbeddingsResult:
        """Embed *texts* via the provider's embeddings endpoint.

        ``POST {base_url}{path}`` speaking the standard OpenAI shape.  The path
        resolves via :meth:`media_path` (``path=`` override → brand map →
        standard protocol default, ``/embeddings`` for OpenAI-style wire).
        Override for brands that differ.  Raises
        :class:`CapabilityNotSupported` when the protocol has no embeddings
        path; RuntimeError on transport or API errors.
        """
        path = self.media_path("embed", options.pop("path", None))
        if path is None:
            raise CapabilityNotSupported(
                f"{self.provider_name()} does not support embeddings for protocol '{self.protocol}' — "
                f"pass path=..., or add an 'embed' entry to _media_gen_urls on the provider class."
            )
        body: dict[str, Any] = {"model": model, "input": texts, **options}
        data = await self._post_json_endpoint(path, body, label="embeddings")

        vectors: list[list[float]] = [[] for _ in texts]
        for i, item in enumerate(data.get("data") or []):
            if not isinstance(item, dict):
                continue
            index = item.get("index", i)
            if isinstance(index, int) and 0 <= index < len(vectors):
                vectors[index] = list(item.get("embedding") or [])

        usage = None
        raw_usage = data.get("usage")
        if raw_usage:
            usage = UsageInfo(
                input_tokens=raw_usage.get("prompt_tokens", 0),
                output_tokens=0,
            )

        return EmbeddingsResult(vectors=vectors, model=data.get("model") or model, usage=usage)
