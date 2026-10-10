"""Per-purpose media request paths — resolution order and wiring.

Order (first hit wins): call-site / model-binding ``path=`` override →
brand map ``_media_gen_urls`` on the provider class → protocol default
``_DEFAULT_MEDIA_PATHS`` (standard OpenAI-fashion endpoint names).

No network: every provider gets a stubbed transport (one sender seam for all
media traffic).
"""

from __future__ import annotations

import base64
from pathlib import Path

import pytest

from cliver_core import capabilities
from cliver_core.bindings import ModelBinding
from cliver_core.capabilities._base import coerce_media
from cliver_core.errors import CapabilityNotSupported
from cliver_core.media import MediaType
from cliver_core.messages import CLIverMessage
from cliver_core.provider import CLIverRequest, CLIverResponse, EmbeddingsResult, Provider
from cliver_core.provider.providers.anthropic import AnthropicProvider
from cliver_core.provider.providers.minimax import MiniMaxAnthropicProvider
from cliver_core.provider.providers.openai import OpenAIProvider


def _response(text: str) -> CLIverResponse:
    return CLIverResponse(message=CLIverMessage(role="assistant", content=text))


class _BareOpenAI(Provider):
    """Minimal concrete OpenAI-protocol provider; records the JSON endpoints it posts to."""

    protocol = "openai"
    brand = "bare"

    def __init__(self, base_url: str = "https://api.example/v1") -> None:
        super().__init__(api_key="k", base_url=base_url)
        self.endpoints: list[str] = []

    async def chat(self, request: CLIverRequest) -> CLIverResponse:  # pragma: no cover
        raise NotImplementedError

    async def stream(self, request: CLIverRequest):  # pragma: no cover
        if False:
            yield None

    async def _post_json_endpoint(self, path: str, body: dict, *, label: str, timeout: float = 120.0) -> dict:
        self.endpoints.append(path)
        if label == "embeddings":
            return {"data": [{"index": 0, "embedding": [0.5]}]}
        return {"results": [{"index": 0, "relevance_score": 0.9}]}


class _BareAnthropic(_BareOpenAI):
    """Same bare provider, protocol-locked to Anthropic (no media endpoints)."""

    protocol = "anthropic"


# ── Resolution order on the provider ───────────────────────────


def test_protocol_defaults_are_openai_fashion() -> None:
    provider = _BareOpenAI()
    assert provider.media_path("image") == "/images/generations"
    assert provider.media_path("tts") == "/audio/speech"
    assert provider.media_path("audio") == "/audio/speech"  # legacy alias
    assert provider.media_path("asr") == "/audio/transcriptions"
    assert provider.media_path("video") == "/videos/generations"
    assert provider.media_path("embed") == "/embeddings"
    assert provider.media_path("rerank") == "/rerank"
    assert provider.media_path("unknown-purpose") is None


def test_override_wins_over_brand_and_protocol() -> None:
    provider = _BareOpenAI()
    assert provider.media_path("image", override="/v1/gen/images") == "/v1/gen/images"


def test_brand_map_wins_over_protocol_default() -> None:
    class _BrandProvider(_BareOpenAI):
        _media_gen_urls = {"image": "/image_generation", "video": "/v1/video/generations"}

    provider = _BrandProvider()
    assert provider.media_path("image") == "/image_generation"
    assert provider.media_path("video") == "/v1/video/generations"
    # purposes the brand doesn't touch keep the protocol default
    assert provider.media_path("tts") == "/audio/speech"

    # an explicit override still beats the brand entry
    assert provider.media_path("image", override="/v2/images") == "/v2/images"


def test_anthropic_protocol_has_no_media_paths() -> None:
    provider = _BareAnthropic()
    assert provider.media_path("image") is None
    assert provider.media_path("embed") is None


@pytest.mark.asyncio
async def test_embed_and_rerank_use_resolved_path() -> None:
    provider = _BareOpenAI()

    await provider.embed(["a", "b"], model="m")
    assert provider.endpoints[-1] == "/embeddings"
    await provider.embed(["a"], model="m", path="/v1/embed/custom")
    assert provider.endpoints[-1] == "/v1/embed/custom"

    await provider.rerank("q", ["a"], model="m")
    assert provider.endpoints[-1] == "/rerank"
    await provider.rerank("q", ["a"], model="m", path="/v1/rerankers")
    assert provider.endpoints[-1] == "/v1/rerankers"


@pytest.mark.asyncio
async def test_embed_on_protocol_without_path_raises() -> None:
    provider = _BareAnthropic()
    with pytest.raises(CapabilityNotSupported, match="does not support embeddings"):
        await provider.embed(["a"], model="m")
    with pytest.raises(CapabilityNotSupported, match="does not support rerank"):
        await provider.rerank("q", ["a"], model="m")


@pytest.mark.asyncio
async def test_media_methods_on_protocol_without_path_raise() -> None:
    provider = _BareAnthropic()  # no media endpoints
    with pytest.raises(CapabilityNotSupported, match="does not support image"):
        await provider.image("a cat", "m")
    with pytest.raises(CapabilityNotSupported, match="does not support tts"):
        await provider.tts("hi", "m")
    with pytest.raises(CapabilityNotSupported, match="does not support video"):
        await provider.video("a sunset", "m")
    with pytest.raises(CapabilityNotSupported, match="does not support audio transcription"):
        await provider.asr(coerce_media(b"RIFF"), "whisper-1")


@pytest.mark.asyncio
async def test_path_override_is_not_sent_in_the_body(monkeypatch) -> None:
    provider = _BareOpenAI()
    seen: dict = {}

    async def fake_post(path: str, body: dict, *, label: str, timeout: float = 120.0) -> dict:
        seen["path"], seen["body"] = path, body
        return {"data": [{"index": 0, "embedding": [0.5]}]}

    monkeypatch.setattr(provider, "_post_json_endpoint", fake_post)
    await provider.embed(["a"], model="m", path="/v1/embed/custom")
    assert seen["path"] == "/v1/embed/custom"
    assert "path" not in seen["body"]


# ── Provider media flows — defaults and overrides ──────────────


class _RecordingTransport:
    def __init__(self) -> None:
        self.posts: list[dict] = []

    async def post(self, path: str, body: dict, *, files=None, binary: bool = False):
        self.posts.append({"path": path, "body": body, "files": files, "binary": binary})
        if binary:
            return b"AUDIO"
        if files:
            return {"text": "transcribed"}
        return {"data": [{"url": "https://cdn.example/out"}]}


@pytest.mark.asyncio
async def test_provider_posts_to_standard_paths_and_honors_override() -> None:
    transport = _RecordingTransport()
    provider = OpenAIProvider(api_key="k", base_url="https://api.example/v1", transport=transport)

    await provider.image("p", "m")
    assert transport.posts[-1]["path"] == "https://api.example/v1/images/generations"
    await provider.image("p", "m", path="/custom/images")
    assert transport.posts[-1]["path"] == "https://api.example/v1/custom/images"
    assert "path" not in transport.posts[-1]["body"]

    await provider.tts("say", "m")
    assert transport.posts[-1]["path"] == "https://api.example/v1/audio/speech"
    assert transport.posts[-1]["binary"] is True  # /audio/speech returns raw bytes

    await provider.asr(coerce_media(b"RIFF"), "whisper-1")
    assert transport.posts[-1]["path"] == "https://api.example/v1/audio/transcriptions"

    await provider.video("p", "m")
    assert transport.posts[-1]["path"] == "https://api.example/v1/videos/generations"


class _FixedResultTransport:
    """Transport stub whose POST always returns one canned JSON result."""

    def __init__(self, result: dict) -> None:
        self.result = result

    async def post(self, path: str, body: dict, *, files=None, binary: bool = False):
        return self.result


@pytest.mark.asyncio
async def test_openai_image_items_are_labeled_by_the_vendors_bytes(tmp_path) -> None:
    """Image outputs carry their real format — never the endpoint's png default."""
    payload = b"\xff\xd8\xff\xe0" + b"jpeg-bytes"
    provider = OpenAIProvider(
        api_key="k",
        base_url="https://api.example/v1",
        transport=_FixedResultTransport({"data": [{"b64_json": base64.b64encode(payload).decode()}]}),
    )
    response = await provider.image("p", "m", output_dir=str(tmp_path))

    (mc,) = response.media
    assert mc.mime_type == "image/jpeg"
    saved = Path(mc.saved_path)
    assert saved.name == "generated_0.jpg"
    assert saved.read_bytes() == payload

    # a URL item is labeled by the format the URL names for itself
    provider = OpenAIProvider(
        api_key="k",
        base_url="https://api.example/v1",
        transport=_FixedResultTransport({"data": [{"url": "https://cdn.example/a.webp"}]}),
    )
    (mc,) = (await provider.image("p", "m")).media
    assert mc.mime_type == "image/webp" and mc.get_file_extension() == ".webp"


def test_resolve_media_mime_follows_the_payload_then_the_url() -> None:
    from cliver_core.media import resolve_media_mime, sniff_image_mime

    # magic bytes win when the payload is already in hand
    assert sniff_image_mime(b"\x89PNG\r\n\x1a\n....") == "image/png"
    assert sniff_image_mime(b"\xff\xd8\xff\xe0jpeg") == "image/jpeg"
    assert sniff_image_mime(b"RIFF\x00\x00\x00\x00WEBPVP8 ") == "image/webp"
    assert sniff_image_mime(b"not-an-image") is None
    # …then the URL names its own format (query strings stripped by mimetypes)
    assert resolve_media_mime("image/png", url="https://cdn.example/x.jpeg?Expires=1") == "image/jpeg"
    assert resolve_media_mime("image/png", payload=b"\xff\xd8\xff\xe0", url="https://cdn.example/x.png") == "image/jpeg"
    # …and an unknown format keeps the caller's default
    assert resolve_media_mime("image/png", url="https://cdn.example/out") == "image/png"


class _BrandOpenAIProvider(OpenAIProvider):
    brand = "brandopenai"
    _media_gen_urls = {"video": "/v1/video/generations"}


@pytest.mark.asyncio
async def test_brand_provider_resolves_path_and_posts_it() -> None:
    transport = _RecordingTransport()
    provider = _BrandOpenAIProvider(api_key="k", base_url="https://api.example/v1", transport=transport)

    await provider.image("p", "m")  # protocol default
    assert transport.posts[-1]["path"] == "https://api.example/v1/images/generations"

    await provider.video("p", "m")  # brand map over protocol default
    assert transport.posts[-1]["path"] == "https://api.example/v1/v1/video/generations"

    await provider.image("p", "m", path="/override")  # call-site wins
    assert transport.posts[-1]["path"] == "https://api.example/v1/override"
    assert "path" not in transport.posts[-1]["body"]

    await provider.tts("say", "m")  # legacy alias normalizes to tts at media_path
    assert transport.posts[-1]["path"] == "https://api.example/v1/audio/speech"

    await provider.asr(coerce_media(b"RIFF"), "whisper-1")
    assert transport.posts[-1]["path"] == "https://api.example/v1/audio/transcriptions"


@pytest.mark.asyncio
async def test_anthropic_provider_raises_when_purpose_has_no_path() -> None:
    transport = _RecordingTransport()
    provider = AnthropicProvider(api_key="k", base_url="https://api.example/v1", transport=transport)

    with pytest.raises(CapabilityNotSupported, match="does not support image"):
        await provider.image("a cat", "m")
    assert transport.posts == []  # fail-fast: nothing reaches the wire


@pytest.mark.asyncio
async def test_media_less_brands_raise_before_the_wire_but_keep_embeddings() -> None:
    """Brands with no media endpoints declare it (``_standard_media_fallback``):
    every media purpose fails fast before anything reaches the wire, while
    embed/rerank keep the protocol defaults — the opt-out is media-scoped
    (Ollama's ``/v1/embeddings`` stays untouched)."""
    from cliver_core.provider.providers.deepseek import DeepSeekOpenAIProvider
    from cliver_core.provider.providers.ollama import OllamaProvider

    for cls, base in (
        (DeepSeekOpenAIProvider, "https://api.deepseek.com/v1"),
        (OllamaProvider, "http://localhost:11434/v1"),
    ):
        transport = _RecordingTransport()
        provider = cls(api_key="k", base_url=base, transport=transport)

        with pytest.raises(CapabilityNotSupported, match="does not support image"):
            await provider.image("a cat", "m")
        with pytest.raises(CapabilityNotSupported, match="does not support tts"):
            await provider.tts("hi", "m")
        with pytest.raises(CapabilityNotSupported, match="does not support video"):
            await provider.video("a sunset", "m")
        with pytest.raises(CapabilityNotSupported, match="does not support audio transcription"):
            await provider.asr(coerce_media(b"RIFF"), "m")
        assert transport.posts == []  # fail-fast: nothing reaches the wire

        # a binding-level path= still opts a purpose in …
        assert provider.media_path("image", override="/v1/gen") == "/v1/gen"
        # … a call-site path= opts in too, and reaches the wire
        await provider.image("a cat", "m", path="/images/custom")
        assert transport.posts[-1]["path"] == f"{base}/images/custom"
        assert "path" not in transport.posts[-1]["body"]
        # … and embeddings/rerank are not gated by the media opt-out
        assert provider.media_path("embed") == "/embeddings"
        assert provider.media_path("rerank") == "/rerank"


# ── MiniMax brand fashion ──────────────────────────────────────


def test_minimax_brand_paths_and_url_resolution() -> None:
    provider = MiniMaxAnthropicProvider(api_key="k", base_url="https://api.minimaxi.com/anthropic")
    assert provider.media_path("image") == "/image_generation"
    assert provider._resolve_media_gen_url("image") == "https://api.minimaxi.com/v1/image_generation"
    assert provider.media_path("tts") == "/t2a_v2"
    assert provider._resolve_media_gen_url("tts") == "https://api.minimaxi.com/v1/t2a_v2"
    # a relative override stays on the brand's media domain
    assert (
        provider._resolve_media_gen_url("image", override="/image_generation_v2")
        == "https://api.minimaxi.com/v1/image_generation_v2"
    )
    # an absolute override bypasses the brand domain entirely
    assert (
        provider._resolve_media_gen_url("image", override="https://edge.example/v9/gen")
        == "https://edge.example/v9/gen"
    )
    # no wire-standard fallback: everything but image/tts is unsupported
    assert provider.media_path("video") is None
    assert provider.media_path("asr") is None


@pytest.mark.asyncio
async def test_minimax_unsupported_purposes_raise() -> None:
    transport = _RecordingTransport()
    provider = MiniMaxAnthropicProvider(api_key="k", base_url="https://api.minimaxi.com/anthropic", transport=transport)

    with pytest.raises(CapabilityNotSupported, match="does not support video"):
        await provider.video("a sunset", "video-01")
    with pytest.raises(CapabilityNotSupported, match="does not support audio transcription"):
        await provider.asr(coerce_media(b"RIFF"), "whisper-1")
    assert transport.posts == []  # nothing reaches the wire


# ── MiniMax media — brand URLs/bodies on the shared transport flow ──

_TTS_HEX = b"ID3\x00minimax-tts-bytes".hex()


class _QueuedTransport:
    """Transport stub: records posts, replays canned payloads in order."""

    def __init__(self, *payloads: dict) -> None:
        self.posts: list[dict] = []
        self.queued = list(payloads)

    async def post(self, path: str, body: dict, *, files=None, binary: bool = False):
        self.posts.append({"path": path, "body": body, "files": files, "binary": binary})
        return self.queued.pop(0)


def _tts_ok(audio: str = _TTS_HEX) -> dict:
    return {"base_resp": {"status_code": 0}, "data": {"audio": audio}}


@pytest.mark.asyncio
async def test_minimax_tts_posts_the_t2a_wire_shape() -> None:
    transport = _QueuedTransport(_tts_ok())
    provider = MiniMaxAnthropicProvider(api_key="k", base_url="https://api.minimaxi.com/anthropic", transport=transport)

    response = await provider.tts("say hi", "speech-2.8-hd")

    call = transport.posts[-1]
    assert call["path"] == "https://api.minimaxi.com/v1/t2a_v2"  # brand URL, verbatim
    assert call["binary"] is False  # T2A answers JSON (hex audio), never raw bytes
    body = call["body"]
    assert body["model"] == "speech-2.8-hd"
    assert body["text"] == "say hi" and "prompt" not in body
    assert body["output_format"] == "hex"
    # voice is required by the vendor — always sent, default or explicit
    assert body["voice_setting"] == {"voice_id": "English_expressive_narrator"}
    assert body["audio_setting"] == {"format": "mp3"}

    # hex → bytes → base64: the MediaContent b64 contract, byte-exact
    (mc,) = response.media
    assert mc.type == MediaType.AUDIO
    assert mc.mime_type == "audio/mpeg"
    assert base64.b64decode(mc.data) == bytes.fromhex(_TTS_HEX)


@pytest.mark.asyncio
async def test_minimax_tts_voice_and_tuning_options_fold_into_settings() -> None:
    transport = _QueuedTransport(_tts_ok())
    provider = MiniMaxAnthropicProvider(api_key="k", base_url="https://api.minimaxi.com/anthropic", transport=transport)

    await provider.tts(
        "hi",
        "m",
        voice="English_Trustworth_Man",
        response_format="wav",
        speed=1.2,
        vol=0.5,
        pitch=2,
        sample_rate=32000,
    )

    body = transport.posts[-1]["body"]
    assert body["voice_setting"] == {"voice_id": "English_Trustworth_Man", "speed": 1.2, "vol": 0.5, "pitch": 2}
    assert body["audio_setting"] == {"format": "wav", "sample_rate": 32000}


@pytest.mark.asyncio
async def test_minimax_tts_saves_with_the_requested_format(tmp_path) -> None:
    transport = _QueuedTransport(_tts_ok(), _tts_ok())
    provider = MiniMaxAnthropicProvider(api_key="k", base_url="https://api.minimaxi.com/anthropic", transport=transport)

    response = await provider.tts("hi", "m", output_dir=str(tmp_path))
    (mc,) = response.media
    assert mc.mime_type == "audio/mpeg"
    saved = Path(mc.saved_path)
    assert saved.name == "generated_0.mp3"
    assert saved.read_bytes() == bytes.fromhex(_TTS_HEX)

    # non-mp3 formats keep their own mime/extension — never mislabeled .mp3
    response = await provider.tts("hi", "m", response_format="wav", output_dir=str(tmp_path))
    (mc,) = response.media
    assert mc.mime_type == "audio/wav"
    assert Path(mc.saved_path).name == "generated_0.wav"


@pytest.mark.asyncio
async def test_minimax_tts_error_envelopes_return_message_not_media() -> None:
    transport = _QueuedTransport(
        {"base_resp": {"status_code": 2013, "status_msg": "invalid params"}, "data": None},
        {"base_resp": {"status_code": 0}, "data": None},
        {"base_resp": {"status_code": 0}, "data": {"audio": "not-hex-zz"}},
    )
    provider = MiniMaxAnthropicProvider(api_key="k", base_url="https://api.minimaxi.com/anthropic", transport=transport)

    r1 = await provider.tts("hi", "m")
    assert "invalid params" in r1.message.content and r1.media is None

    r2 = await provider.tts("hi", "m")
    assert "no audio payload" in r2.message.content and r2.media is None

    r3 = await provider.tts("hi", "m")
    assert "not valid hex" in r3.message.content and r3.media is None

    assert len(transport.posts) == 3  # every attempt reached the transport — none crashed


@pytest.mark.asyncio
async def test_minimax_media_base_follows_the_chat_leaf_base_url() -> None:
    """A base_url override (proxy / the other MiniMax platform) retargets media too."""
    transport = _QueuedTransport(
        {"base_resp": {"status_code": 0}, "data": {"image_urls": ["https://cdn.example/x.png"]}},
        _tts_ok(),
    )
    provider = MiniMaxAnthropicProvider(api_key="k", base_url="https://api.minimax.io/anthropic", transport=transport)

    await provider.image("p", "image-01")
    assert transport.posts[-1]["path"] == "https://api.minimax.io/v1/image_generation"

    await provider.tts("hi", "m")
    assert transport.posts[-1]["path"] == "https://api.minimax.io/v1/t2a_v2"


def test_minimax_media_base_honors_an_explicit_v1_base() -> None:
    # a base that is already the media root (proxy or the other platform's
    # /v1) is used as-is — only the chat-shaped /anthropic suffix remaps
    provider = MiniMaxAnthropicProvider(api_key="k", base_url="https://proxy.example/minimax/v1")
    assert provider._resolve_media_gen_url("image") == "https://proxy.example/minimax/v1/image_generation"
    assert provider._resolve_media_gen_url("tts") == "https://proxy.example/minimax/v1/t2a_v2"


@pytest.mark.asyncio
async def test_minimax_image_items_are_labeled_by_the_urls_own_format(tmp_path, monkeypatch) -> None:
    """image-01 serves JPEG — the item must say .jpg, never the flow's png default."""
    transport = _QueuedTransport(
        {"base_resp": {"status_code": 0}, "data": {"image_urls": ["https://cdn.example/out_aigc.jpeg?Expires=1"]}},
    )
    jpeg = b"\xff\xd8\xff\xe0minimax-jpeg-bytes"
    monkeypatch.setattr("cliver_core.media._download_url", lambda url: jpeg)
    provider = MiniMaxAnthropicProvider(api_key="k", base_url="https://api.minimaxi.com/anthropic", transport=transport)

    response = await provider.image("p", "image-01", output_dir=str(tmp_path))

    (mc,) = response.media
    assert mc.mime_type == "image/jpeg" and mc.get_file_extension() == ".jpg"
    saved = Path(mc.saved_path)
    assert saved.name == "generated_0.jpg"
    assert saved.read_bytes() == jpeg


# ── Capability functions forward path= ─────────────────────────


class _CapabilityRecorder(Provider):
    protocol = "mock"
    brand = "mock"

    def __init__(self) -> None:
        super().__init__(api_key="k", base_url="https://example/v1")
        self.calls: dict[str, list[dict]] = {"image": [], "video": [], "tts": [], "asr": [], "embed": [], "rerank": []}

    async def close(self) -> None:  # pragma: no cover
        pass

    async def chat(self, request) -> CLIverResponse:  # pragma: no cover
        raise NotImplementedError

    async def stream(self, request):  # pragma: no cover
        if False:
            yield None

    async def image(self, prompt, model, *, media=None, output_dir=None, **options):
        self.calls["image"].append({"prompt": prompt, "model": model, "output_dir": output_dir, **options})
        return _response("ok")

    async def video(self, prompt, model, *, media=None, output_dir=None, **options):
        self.calls["video"].append({"prompt": prompt, "model": model, "output_dir": output_dir, **options})
        return _response("ok")

    async def tts(self, text, model, *, output_dir=None, **options):
        self.calls["tts"].append({"text": text, "model": model, "output_dir": output_dir, **options})
        return _response("ok")

    async def asr(self, media, model, *, language=None, **options):
        self.calls["asr"].append({"media": media, "model": model, "language": language, **options})
        return _response("ok")

    async def embed(self, texts, *, model, **options):
        self.calls["embed"].append({"model": model, **options})
        return EmbeddingsResult(vectors=[[0.1]], model=model)

    async def rerank(self, query, documents, *, model, top_k=None, **options):
        self.calls["rerank"].append({"top_k": top_k, **options})
        return []


@pytest.mark.asyncio
async def test_capabilities_forward_the_binding_path_to_the_provider() -> None:
    rec = _CapabilityRecorder()

    await capabilities.generate_image(
        prompt="p", binding=ModelBinding.image(model="m", provider=rec, path="/v1/gen/images")
    )
    assert rec.calls["image"][-1]["path"] == "/v1/gen/images"
    await capabilities.generate_video(
        prompt="p", binding=ModelBinding.video(model="m", provider=rec, path="/v1/gen/videos")
    )
    assert rec.calls["video"][-1]["path"] == "/v1/gen/videos"
    await capabilities.text_to_speech(
        text="hi", binding=ModelBinding.tts(model="m", provider=rec, path="/v1/gen/speech")
    )
    assert rec.calls["tts"][-1]["path"] == "/v1/gen/speech"
    await capabilities.transcribe_audio(
        audio=b"RIFF", binding=ModelBinding.asr(model="m", provider=rec, path="/v1/asr")
    )
    assert rec.calls["asr"][-1]["path"] == "/v1/asr"
    await capabilities.embed(texts="hello", binding=ModelBinding.embed(model="m", provider=rec, path="/v1/embed"))
    assert rec.calls["embed"][-1]["path"] == "/v1/embed"
    await capabilities.rerank(
        query="q", documents=["d"], binding=ModelBinding.rerank(model="m", provider=rec, path="/v1/rerank")
    )
    assert rec.calls["rerank"][-1]["path"] == "/v1/rerank"

    # no path on the binding → nothing sent; the provider resolves brand/protocol defaults
    await capabilities.generate_image(prompt="p", binding=ModelBinding.image(model="m", provider=rec))
    assert "path" not in rec.calls["image"][-1]
    await capabilities.embed(texts="hello", binding=ModelBinding.embed(model="m", provider=rec))
    assert "path" not in rec.calls["embed"][-1]

    # a call-site path overrides the binding's default — allowed by design
    await capabilities.generate_image(prompt="p", binding=ModelBinding.image(model="m", provider=rec), path="/call/gen")
    assert rec.calls["image"][-1]["path"] == "/call/gen"


def test_binding_factories_set_path_field_not_params() -> None:
    image = ModelBinding.image(model="m", path="/v1/gen", size="512x512")
    assert image.path == "/v1/gen"  # endpoint config — a field, like base_url
    assert image.params == {"size": "512x512"}  # generation params stay in params

    assert ModelBinding.tts(model="m", path="/v1/speech").path == "/v1/speech"
    assert ModelBinding.asr(model="m", path="/v1/asr", language="en").path == "/v1/asr"
    assert ModelBinding.embed(model="m", path="/v1/embed", dimensions=256).path == "/v1/embed"
    assert ModelBinding.rerank(model="m", path="/v1/rerank").path == "/v1/rerank"
    assert ModelBinding.video(model="m", path="/v1/video").path == "/v1/video"

    # unset → None, and no stray key anywhere
    bare = ModelBinding.image(model="m", size="512x512")
    assert bare.path is None
    assert "path" not in bare.params


def test_llm_factory_rejects_path() -> None:
    """Chat has no endpoint path override — a ``path=`` must fail loudly
    (it would otherwise ride ``**params`` into the request body)."""
    with pytest.raises(TypeError, match="takes no path"):
        ModelBinding.llm(model="m", path="/v1/chat")


def test_binding_factories_record_their_purpose() -> None:
    assert ModelBinding.llm(model="m").purpose == "llm"
    assert ModelBinding.image(model="m").purpose == "image"
    assert ModelBinding.video(model="m").purpose == "video"
    assert ModelBinding.tts(model="m").purpose == "tts"
    assert ModelBinding.asr(model="m").purpose == "asr"
    assert ModelBinding.embed(model="m").purpose == "embed"
    assert ModelBinding.rerank(model="m").purpose == "rerank"
    # the raw constructor binds no purpose (no validation, routing by slot only)
    assert ModelBinding(model="m").purpose is None
    # purpose is derived metadata — it does not affect equality
    assert ModelBinding.image(model="m") == ModelBinding(model="m")


def test_mounting_a_binding_under_the_wrong_purpose_fails_fast() -> None:
    from cliver_core.bindings import ModelBindings

    wrong = ModelBindings(embed=ModelBinding.image(model="m", size="512x512"))
    with pytest.raises(ValueError, match=r"bindings\.embed holds a 'image' binding"):
        wrong.get("embed")

    right = ModelBindings(image=ModelBinding.image(model="m"), tts=ModelBinding.tts(model="m"))
    assert right.get("image") is not None
    assert right.get("tts") is not None
    # the legacy audio slot accepts tts bindings
    legacy = ModelBindings(audio=ModelBinding.tts(model="m"))
    assert legacy.get("audio") is not None
    # raw-constructor bindings skip the check
    assert ModelBindings(embed=ModelBinding(model="m")).get("embed") is not None
