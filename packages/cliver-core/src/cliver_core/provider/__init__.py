"""Provider interface and request/response models."""

import base64
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, AsyncIterator

from pydantic import BaseModel, Field

from cliver_core.errors import CapabilityNotSupported
from cliver_core.media import MediaContent, MediaType, resolve_media_mime
from cliver_core.messages import CLIverMessage, CLIverMessageChunk, UsageInfo
from cliver_core.tool import CLIverTool

if TYPE_CHECKING:
    from cliver_core.provider.transport import BrandSDK

logger = logging.getLogger(__name__)


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
    request_info: dict[str, Any] | None = None
    # Diagnostics echo of the request that produced this response
    # (model, provider, options, system_prompt, messages, tools) —
    # AgentCore collects it into the Re-Act trace.


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

    def _media_kind(self, purpose: str, options: dict[str, Any]) -> tuple[MediaType, str, str]:
        """Resolve (kind, mime, extension) for the shared HTTP media flow.

        For TTS a requested ``response_format`` wins over the static default,
        so ``response_format="wav"`` writes ``generated_0.wav`` with
        ``audio/wav`` instead of mislabeling the bytes ``.mp3``.
        """
        default = self._MEDIA_KINDS.get(purpose, (MediaType.IMAGE, "application/octet-stream", "bin"))
        if purpose != "tts":
            return default
        fmt = options.get("response_format")
        if not isinstance(fmt, str) or not fmt:
            return default
        mime, ext = self._AUDIO_FORMAT_KINDS.get(fmt.lower(), (f"audio/{fmt.lower()}", fmt.lower()))
        return (MediaType.AUDIO, mime, ext)

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

    async def _post_media(self, url: str, body: dict, *, files: dict | None = None, binary: bool = False):
        """POST one media request through the transport — the single sender seam.

        Media traffic is transport I/O exactly like chat: the URL is the
        absolute one :meth:`_resolve_media_gen_url` resolved (both bundled
        SDK transports pass absolute URLs through verbatim), and brand
        differences live in the body/response hooks — never in a second
        HTTP stack.
        """
        if self.transport is None:
            raise RuntimeError(
                f"{self.provider_name()} has no transport — media generation requires a protocol provider"
            )
        return await self.transport.post(url, body, files=files, binary=binary)

    async def _generate_media(
        self,
        prompt: str,
        *,
        model: str,
        purpose: str = "image",
        media: list[MediaContent] | None = None,
        output_dir: str | None = None,
        **options: Any,
    ) -> CLIverResponse:
        """Generate media — the single pipeline every provider shares.

        Resolves the endpoint URL (*purpose* → ``path=`` override / brand map
        / protocol default), POSTs through the transport, and turns the
        payload into :class:`MediaContent` items (saved to *output_dir*).  A
        purpose with no resolvable endpoint raises
        :class:`CapabilityNotSupported` before anything reaches the wire;
        runtime failures return a friendly :class:`CLIverResponse`.

        ``timeout`` is consumed for call-site compatibility (the transport's
        client owns request timeouts) so it never leaks into the body.
        """
        options.pop("timeout", None)
        url = self._resolve_media_gen_url(purpose, options.pop("path", None))
        if not url:
            label = self._UNSUPPORTED_MEDIA_LABELS.get(purpose, purpose)
            raise CapabilityNotSupported(
                f"{self.provider_name()} does not support {label}. "
                f"Pass path= to the call or set it on the model binding "
                f"(ModelBinding.<purpose>(path=...)), add a '{purpose}' entry to "
                f"_media_gen_urls on the provider class, or set api_url on the model config."
            )

        if purpose == "asr":
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
    ) -> CLIverResponse:
        """One non-ASR media request: build the body, POST it, wrap the payload."""
        from pathlib import Path

        kind, mime, ext = self._media_kind(purpose, options)
        binary = purpose in self._BINARY_MEDIA_PURPOSES
        body = self._build_media_gen_body(prompt, model, purpose, media=media, **options)

        raw = await self._post_media(url, body, binary=binary)

        if binary:
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
            item_mime = resolve_media_mime(mime, url=item_url, payload=payload)
            media_items.append(MediaContent(type=kind, data=data, mime_type=item_mime))

        # Save to disk — same pattern for ALL providers.
        if output_dir and media_items:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            for i, mc in enumerate(media_items):
                if mc.mime_type != mime:  # the item's real format won over the default
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
        """One multipart transcription request — text out, friendly on failure."""
        language = options.pop("language", None)
        filename = media.filename or f"audio{media.get_file_extension()}"
        body: dict[str, Any] = {"model": model, **options}
        if language:
            body["language"] = language
        files = {"file": (filename, media.to_bytes(), media.mime_type)}

        try:
            result = await self._post_media(url, body, files=files)
        except Exception as e:
            self.logger.warning("%s asr API error: %s", self.provider_name(), e)
            return self._media_failure_response("asr", e)

        text = result.get("text", "") if isinstance(result, dict) else str(result)
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

        The endpoint path resolves via ``options["path"]`` (set by the call
        site, model binding, or provider) → brand map → protocol default; a
        purpose with no path raises :class:`CapabilityNotSupported`.
        """
        return await self._generate_media(
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

        *media* is accepted for a uniform capability-call shape; the wire
        video endpoint takes the prompt only.
        """
        return await self._generate_media(
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
        """Synthesize speech — POST to the resolved TTS endpoint."""
        return await self._generate_media(text, model=model, purpose="tts", output_dir=output_dir, **options)

    async def asr(
        self,
        media: MediaContent,
        model: str,
        *,
        language: str | None = None,
        **options,
    ) -> CLIverResponse:
        """Transcribe audio — multipart POST to the resolved transcription endpoint."""
        return await self._generate_media(media, model=model, purpose="asr", media=media, language=language, **options)

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
