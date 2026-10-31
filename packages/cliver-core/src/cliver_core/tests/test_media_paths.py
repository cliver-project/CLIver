"""Per-purpose media request paths — resolution order and wiring.

Order (first hit wins): call-site / model-binding ``path=`` override →
brand map ``_media_gen_urls`` on the provider class → protocol default
``_DEFAULT_MEDIA_PATHS`` (standard OpenAI-fashion endpoint names).

No network: every provider gets a stubbed transport (one sender seam for all
media traffic).
"""

from __future__ import annotations

import base64
import inspect
import logging
from pathlib import Path

import pytest

from cliver_core import capabilities
from cliver_core.bindings import ModelBinding
from cliver_core.capabilities._base import coerce_media
from cliver_core.errors import CapabilityNotSupported, ProviderError
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


# ── ASR — multipart fashion, text formats, stream guard ────────


@pytest.mark.asyncio
async def test_asr_plain_text_formats_come_back_as_text() -> None:
    """srt/vtt (and the OpenAI-wire ``text``) bodies are plain text, not
    JSON — requested as raw bytes and decoded into the transcript slot."""
    srt = b"1\n00:00:00,100 --> 00:00:01,660\nHello everyone.\n"

    class _TextTransport:
        def __init__(self) -> None:
            self.posts: list[dict] = []

        async def post(self, path, body, *, files=None, binary=False):
            self.posts.append({"binary": binary, "body": body})
            return srt if binary else {"text": "hello world"}

    transport = _TextTransport()
    provider = OpenAIProvider(api_key="k", base_url="https://api.example/v1", transport=transport)

    response = await provider.asr(coerce_media(b"RIFF"), "whisper-1", response_format="srt")
    assert transport.posts[-1]["binary"] is True
    assert response.message.content == srt.decode()

    # the default json fashion stays JSON-typed
    response = await provider.asr(coerce_media(b"RIFF"), "whisper-1")
    assert transport.posts[-1]["binary"] is False
    assert response.message.content == "hello world"


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [True, "true", "1", "on"])
async def test_asr_stream_true_is_rejected_before_the_wire(value) -> None:
    transport = _RecordingTransport()
    provider = OpenAIProvider(api_key="k", base_url="https://api.example/v1", transport=transport)

    with pytest.raises(ValueError, match="streaming transcription is not supported"):
        await provider.asr(coerce_media(b"RIFF"), "whisper-1", stream=value)
    assert transport.posts == []


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [False, None, "false", "0", ""])
async def test_asr_falsy_stream_values_are_consumed_not_forwarded(value) -> None:
    """The vendor samples send ``"false"`` — a falsy flag keeps the vendor's
    non-streaming default and never leaks into the multipart body."""
    transport = _RecordingTransport()
    provider = OpenAIProvider(api_key="k", base_url="https://api.example/v1", transport=transport)

    response = await provider.asr(coerce_media(b"RIFF"), "whisper-1", stream=value)
    assert response.message.content == "transcribed"
    assert "stream" not in transport.posts[-1]["body"]


# ── MiniMax brand fashion ──────────────────────────────────────


def test_minimax_brand_paths_and_url_resolution() -> None:
    provider = MiniMaxAnthropicProvider(api_key="k", base_url="https://api.minimaxi.com/anthropic")
    assert provider.media_path("image") == "/image_generation"
    assert provider._resolve_media_gen_url("image") == "https://api.minimaxi.com/v1/image_generation"
    assert provider.media_path("tts") == "/t2a_v2"
    assert provider._resolve_media_gen_url("tts") == "https://api.minimaxi.com/v1/t2a_v2"
    assert provider.media_path("asr") == "/speech_to_text"
    assert provider._resolve_media_gen_url("asr") == "https://api.minimaxi.com/v1/speech_to_text"
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
    # no wire-standard fallback: everything but image/tts/asr is unsupported
    assert provider.media_path("video") is None


@pytest.mark.asyncio
async def test_minimax_unsupported_purposes_raise() -> None:
    transport = _RecordingTransport()
    provider = MiniMaxAnthropicProvider(api_key="k", base_url="https://api.minimaxi.com/anthropic", transport=transport)

    with pytest.raises(CapabilityNotSupported, match="does not support video"):
        await provider.video("a sunset", "video-01")
    assert transport.posts == []  # nothing reaches the wire


# ── MiniMax media — brand URLs/bodies on the shared transport flow ──

_TTS_HEX = b"ID3\x00minimax-tts-bytes".hex()


class _QueuedTransport:
    """Transport stub: records posts, replays canned payloads in order."""

    def __init__(self, *payloads: dict) -> None:
        self.posts: list[dict] = []
        self.queued = list(payloads)

    async def post(self, path: str, body: dict, *, files=None, binary: bool = False, headers=None):
        self.posts.append({"path": path, "body": body, "files": files, "binary": binary, "headers": headers})
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


# ── MiniMax ASR — speech_to_text wire shape ────────────────────


@pytest.mark.asyncio
async def test_minimax_asr_posts_the_speech_to_text_wire_shape() -> None:
    transport = _QueuedTransport({"text": "hello world", "duration": 1.5, "trace_id": "t1"})
    provider = MiniMaxAnthropicProvider(api_key="k", base_url="https://api.minimaxi.com/anthropic", transport=transport)

    response = await provider.asr(
        coerce_media(b"RIFF"), "asr-1.0", language="en", response_format="json", timestamp_level="word"
    )

    call = transport.posts[-1]
    assert call["path"] == "https://api.minimaxi.com/v1/speech_to_text"  # brand URL, verbatim
    assert call["binary"] is False  # json/verbose_json answer JSON
    # language rides the BCP-47 HTTP header — never the multipart body
    assert call["headers"] == {"language": "en"}
    assert call["body"] == {"model": "asr-1.0", "response_format": "json", "timestamp_level": "word"}
    assert call["files"] == {"file": ("audio.wav", b"RIFF", "audio/wav")}
    assert response.message.content == "hello world"


@pytest.mark.asyncio
async def test_minimax_asr_without_language_keeps_the_legacy_transport_shape() -> None:
    """No header options → the transport call is exactly the four-argument
    shape every pre-existing (including third-party) transport implements."""
    posts: list[dict] = []

    class _LegacyTransport:  # the pre-headers BrandSDK.post signature, verbatim
        async def post(self, path: str, body: dict, *, files=None, binary: bool = False):
            posts.append({"path": path, "body": body, "files": files, "binary": binary})
            return {"text": "mixed language", "duration": 0.5, "trace_id": "t2"}

    provider = MiniMaxAnthropicProvider(
        api_key="k", base_url="https://api.minimaxi.com/anthropic", transport=_LegacyTransport()
    )
    response = await provider.asr(coerce_media(b"RIFF"), "asr-1.0")

    assert posts[-1]["path"] == "https://api.minimaxi.com/v1/speech_to_text"
    assert posts[-1]["body"] == {"model": "asr-1.0"}
    assert response.message.content == "mixed language"


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


# ── Public generate_media seam — custom kinds and output knobs ──


class _FailingTransport:
    """Transport stub whose POST fails — the pipeline must fail friendly, not crash."""

    async def post(self, path: str, body: dict, *, files=None, binary: bool = False):
        raise RuntimeError("connection reset")


def _identity_handler(payload) -> CLIverResponse:
    return _response("handled")


def test_public_seam_signature_is_pinned() -> None:
    import cliver_core
    from cliver_core.provider import MediaOutputHandler

    assert MediaOutputHandler is not None  # importable from cliver_core.provider …
    assert "MediaOutputHandler" not in cliver_core.__all__  # … but not a core export

    params = inspect.signature(Provider.generate_media).parameters
    assert list(params) == [
        "self",
        "prompt",
        "model",
        "purpose",
        "media",
        "output_dir",
        "path",
        "mime",
        "binary",
        "output_handler",
        "options",
    ]
    assert params["prompt"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert params["model"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["options"].kind is inspect.Parameter.VAR_KEYWORD
    # tri-state: None keeps the protocol policy (tts is bytes-out, others JSON)
    assert params["binary"].default is None
    assert params["mime"].default is None
    assert params["output_handler"].default is None
    assert params["path"].default is None


@pytest.mark.asyncio
async def test_unknown_purpose_without_path_raises_before_the_wire() -> None:
    transport = _RecordingTransport()
    provider = OpenAIProvider(api_key="k", base_url="https://api.example/v1", transport=transport)

    with pytest.raises(CapabilityNotSupported, match="does not support music"):
        await provider.generate_media("p", model="m", purpose="music")
    assert transport.posts == []


@pytest.mark.asyncio
async def test_custom_purpose_rides_the_whole_pipeline(tmp_path) -> None:
    """A kind the core does not model: explicit path=, vendor options verbatim
    into the body, OpenAI-fashion extraction, generated_i.<ext> saving."""
    audio = base64.b64encode(b"music-bytes").decode()
    transport = _QueuedTransport({"data": [{"b64_json": audio}]})
    provider = OpenAIProvider(api_key="k", base_url="https://api.example/v1", transport=transport)

    response = await provider.generate_media(
        "calm piano loop",
        model="music-1",
        purpose="music",
        path="/v1/music/generate",
        mime="audio/mpeg",
        duration=30,
        output_format="mp3",
        output_dir=str(tmp_path),
    )

    call = transport.posts[-1]
    assert call["path"] == "https://api.example/v1/v1/music/generate"
    # the pipeline knobs are consumed, never part of the request body
    assert call["body"] == {"model": "music-1", "prompt": "calm piano loop", "duration": 30, "output_format": "mp3"}
    assert call["binary"] is False

    assert response.message.content.startswith("Generated 1 music(s)")
    (mc,) = response.media
    assert mc.type == MediaType.AUDIO and mc.mime_type == "audio/mpeg"
    assert base64.b64decode(mc.data) == b"music-bytes"
    saved = Path(mc.saved_path)
    assert saved.name == "generated_0.mp3"
    assert saved.read_bytes() == b"music-bytes"


def test_media_kind_derives_kind_and_extension_from_an_explicit_mime() -> None:
    provider = _BareOpenAI()

    assert provider._media_kind("music", {}, mime="audio/mpeg") == (MediaType.AUDIO, "audio/mpeg", "mp3")
    assert provider._media_kind("3d", {}, mime="model/gltf-binary") == (MediaType.FILE, "model/gltf-binary", "glb")
    # an explicit mime wins over the purpose default and the tts format label
    assert provider._media_kind("image", {}, mime="image/webp") == (MediaType.IMAGE, "image/webp", "webp")
    assert provider._media_kind("tts", {"response_format": "wav"}, mime="audio/mpeg") == (
        MediaType.AUDIO,
        "audio/mpeg",
        "mp3",
    )
    # without one, unknown purposes are plain files — never mislabeled images
    assert provider._media_kind("music", {}) == (MediaType.FILE, "application/octet-stream", "bin")
    assert provider._media_kind("image", {}) == (MediaType.IMAGE, "image/png", "png")


def test_mime_helpers_classify_and_name_custom_formats() -> None:
    from cliver_core.media import extension_from_mime, media_type_from_mime

    assert media_type_from_mime("image/png") is MediaType.IMAGE
    assert media_type_from_mime("audio/mpeg; charset=x") is MediaType.AUDIO
    assert media_type_from_mime("video/mp4") is MediaType.VIDEO
    assert media_type_from_mime("text/plain") is MediaType.TEXT
    # everything unclassifiable is a plain file — never a mislabeled image
    assert media_type_from_mime("model/gltf+json") is MediaType.FILE
    assert media_type_from_mime("application/octet-stream") is MediaType.FILE

    assert extension_from_mime("image/jpeg") == "jpg"  # the module's table decides
    assert extension_from_mime("audio/mpeg; mode=x") == "mp3"  # parameters stripped
    assert extension_from_mime("model/gltf+json") == "gltf"
    assert extension_from_mime("model/gltf-binary") == "glb"
    assert extension_from_mime("model/obj") == "obj"  # else the subtype itself
    assert extension_from_mime("application/octet-stream") == "bin"
    assert extension_from_mime("") == "bin"


@pytest.mark.asyncio
async def test_url_evidence_beats_the_explicit_mime_label() -> None:
    """mime= is the label of last resort — the vendor's real format still wins."""
    transport = _QueuedTransport({"data": [{"url": "https://cdn.example/a.webp"}]})
    provider = OpenAIProvider(api_key="k", base_url="https://api.example/v1", transport=transport)

    response = await provider.generate_media("p", model="m", purpose="art", path="/v1/art", mime="image/png")

    (mc,) = response.media
    assert mc.mime_type == "image/webp"


@pytest.mark.asyncio
async def test_binary_override_decides_bytes_out_per_call() -> None:
    transport = _QueuedTransport(b"RIFF", {"data": [{"url": "https://cdn.example/out.mp3"}]})
    provider = OpenAIProvider(api_key="k", base_url="https://api.example/v1", transport=transport)

    # policy default for tts: the transport is asked for raw bytes
    response = await provider.generate_media("hi", model="m", purpose="tts")
    assert transport.posts[-1]["binary"] is True
    (mc,) = response.media
    assert mc.type == MediaType.AUDIO and mc.mime_type == "audio/mpeg"
    assert base64.b64decode(mc.data) == b"RIFF"

    # an explicit binary=False wins over the policy: JSON fashion
    response = await provider.generate_media("hi", model="m", purpose="tts", binary=False)
    assert transport.posts[-1]["binary"] is False
    (mc,) = response.media
    assert mc.mime_type == "audio/mpeg"

    # an explicit binary=True brings bytes for a purpose the policy sends as JSON
    mesh_transport = _QueuedTransport(b"mesh-bytes")
    mesh_provider = OpenAIProvider(api_key="k", base_url="https://api.example/v1", transport=mesh_transport)
    response = await mesh_provider.generate_media(
        "a low-poly teapot", model="m", purpose="3d", path="/v1/meshes", mime="model/gltf-binary", binary=True
    )
    assert mesh_transport.posts[-1]["binary"] is True
    (mc,) = response.media
    assert mc.type == MediaType.FILE and mc.mime_type == "model/gltf-binary"
    assert base64.b64decode(mc.data) == b"mesh-bytes"


@pytest.mark.asyncio
async def test_legacy_audio_purpose_normalizes_to_tts_on_the_public_seam() -> None:
    """purpose="audio" must resolve the tts endpoint AND keep the bytes-out
    policy — else the transport JSON-parses raw audio into a guaranteed failure."""
    transport = _QueuedTransport(b"AUDIO")
    provider = OpenAIProvider(api_key="k", base_url="https://api.example/v1", transport=transport)

    response = await provider.generate_media("hi", model="m", purpose="audio")

    assert transport.posts[-1]["path"] == "https://api.example/v1/audio/speech"
    assert transport.posts[-1]["binary"] is True
    (mc,) = response.media
    assert mc.type == MediaType.AUDIO
    assert base64.b64decode(mc.data) == b"AUDIO"


@pytest.mark.asyncio
async def test_output_handler_replaces_the_default_output_handling(tmp_path) -> None:
    seen: list[dict] = []

    def handle(payload: dict) -> CLIverResponse:
        seen.append(payload)
        return _response(f"picked {payload['data']['image_urls'][0]}")

    envelope = {"base_resp": {"status_code": 2013, "status_msg": "invalid params"}, "data": {"image_urls": ["u1"]}}
    transport = _QueuedTransport(envelope, envelope)
    provider = MiniMaxAnthropicProvider(api_key="k", base_url="https://api.minimaxi.com/anthropic", transport=transport)

    response = await provider.generate_media(
        "p", model="image-01", purpose="image", output_dir=str(tmp_path), output_handler=handle
    )

    assert seen == [envelope]  # the raw payload, vendor error envelope and all
    assert response.message.content == "picked u1"
    assert response.media is None
    assert list(tmp_path.iterdir()) == []  # output_dir is ignored under a handler

    # control: the same payload without a handler is the vendor's error
    failure = await provider.generate_media("p", model="image-01", purpose="image")
    assert failure.media is None
    assert "image generation failed" in failure.message.content


@pytest.mark.asyncio
async def test_output_handler_may_be_async() -> None:
    async def handle(payload: dict) -> CLIverResponse:
        return _response("async ok")

    transport = _RecordingTransport()
    provider = OpenAIProvider(api_key="k", base_url="https://api.example/v1", transport=transport)

    response = await provider.generate_media("p", model="m", purpose="image", output_handler=handle)

    assert response.message.content == "async ok"
    assert transport.posts  # the POST happened first, then the handler ran


@pytest.mark.asyncio
async def test_output_handler_sees_raw_bytes_when_the_request_asked_for_them() -> None:
    seen: list[type] = []

    def handle(payload) -> CLIverResponse:
        seen.append(type(payload))
        return _response("bytes ok")

    transport = _QueuedTransport(b"\x00\x01binary-mesh")
    provider = OpenAIProvider(api_key="k", base_url="https://api.example/v1", transport=transport)

    response = await provider.generate_media(
        "a low-poly teapot", model="m", purpose="3d", path="/v1/meshes", binary=True, output_handler=handle
    )

    assert transport.posts[-1]["binary"] is True
    assert seen == [bytes]
    assert response.message.content == "bytes ok"


@pytest.mark.asyncio
async def test_raising_output_handler_becomes_a_friendly_failure(caplog) -> None:
    def handle(payload) -> CLIverResponse:
        raise ValueError("bad payload shape")

    transport = _RecordingTransport()
    provider = OpenAIProvider(api_key="k", base_url="https://api.example/v1", transport=transport)

    with caplog.at_level(logging.WARNING):
        response = await provider.generate_media("p", model="m", purpose="image", output_handler=handle)

    assert response.media is None
    assert "image generation failed" in response.message.content
    assert "bad payload shape" in response.message.content
    assert any("generation failed" in record.getMessage() for record in caplog.records)


@pytest.mark.asyncio
async def test_output_handler_must_return_a_cliver_response() -> None:
    def handle(payload) -> CLIverResponse:
        return "not a response"  # type: ignore[return-value]

    transport = _RecordingTransport()
    provider = OpenAIProvider(api_key="k", base_url="https://api.example/v1", transport=transport)

    response = await provider.generate_media("p", model="m", purpose="image", output_handler=handle)

    assert response.media is None
    assert "output_handler must return a CLIverResponse, got str" in response.message.content


@pytest.mark.asyncio
async def test_output_handler_is_never_called_without_a_successful_post() -> None:
    calls: list = []

    def handle(payload) -> CLIverResponse:
        calls.append(payload)
        return _response("never")

    # a transport failure never reaches the handler …
    provider = OpenAIProvider(api_key="k", base_url="https://api.example/v1", transport=_FailingTransport())
    response = await provider.generate_media("p", model="m", purpose="image", output_handler=handle)
    assert "connection reset" in response.message.content
    assert calls == []

    # … and a purpose with no resolvable endpoint fails before the wire
    provider = AnthropicProvider(api_key="k", base_url="https://api.example/v1", transport=_RecordingTransport())
    with pytest.raises(CapabilityNotSupported, match="does not support image"):
        await provider.generate_media("p", model="m", purpose="image", output_handler=handle)
    assert calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "knob",
    [
        {"mime": "audio/mpeg"},
        {"mime": ""},  # even an empty label is a generation intent
        {"binary": True},
        {"binary": False},
        {"output_handler": _identity_handler},
    ],
)
async def test_asr_purpose_rejects_generation_only_knobs(knob) -> None:
    transport = _RecordingTransport()
    provider = OpenAIProvider(api_key="k", base_url="https://api.example/v1", transport=transport)

    with pytest.raises(ValueError, match="generation-only"):
        await provider.generate_media(
            coerce_media(b"RIFF"), model="whisper-1", purpose="asr", media=coerce_media(b"RIFF"), **knob
        )
    assert transport.posts == []


@pytest.mark.asyncio
async def test_asr_purpose_still_takes_path_and_language_on_the_public_seam() -> None:
    transport = _RecordingTransport()
    provider = OpenAIProvider(api_key="k", base_url="https://api.example/v1", transport=transport)

    response = await provider.generate_media(
        coerce_media(b"RIFF"),
        model="whisper-1",
        purpose="asr",
        media=coerce_media(b"RIFF"),
        path="/v1/asr/custom",
        language="en",
    )

    call = transport.posts[-1]
    assert call["path"] == "https://api.example/v1/v1/asr/custom"
    assert call["body"] == {"model": "whisper-1", "language": "en"}
    assert response.message.content == "transcribed"


def test_file_media_never_becomes_a_message_part() -> None:
    from cliver_core.media import MediaContent, add_media_content_to_message_parts

    parts: list[dict] = []
    glb = base64.b64encode(b"glb-bytes").decode()
    mc = MediaContent(type=MediaType.FILE, data=glb, mime_type="model/gltf-binary")
    add_media_content_to_message_parts(parts, [mc])
    assert parts == []


@pytest.mark.asyncio
async def test_capability_calls_forward_media_output_knobs() -> None:
    """The ``generate_media`` output knobs ride ``**options`` like ``path=``
    does: binding defaults first, a call-site value wins."""
    rec = _CapabilityRecorder()

    def handler(payload) -> CLIverResponse:
        return _response("handled")  # pragma: no cover - the recorder never invokes it

    binding = ModelBinding.image(model="m", provider=rec, output_handler=handler, mime="image/webp")
    await capabilities.generate_image(prompt="p", binding=binding)
    call = rec.calls["image"][-1]
    assert call["output_handler"] is handler
    assert call["mime"] == "image/webp"

    await capabilities.generate_image(prompt="p", binding=binding, mime="image/png")
    assert rec.calls["image"][-1]["mime"] == "image/png"


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


# ── The async-vendor hook — default-inert ──────────────────────


@pytest.mark.asyncio
async def test_await_media_result_default_is_an_identity_passthrough() -> None:
    provider = _BareOpenAI()
    assert list(inspect.signature(Provider._await_media_result).parameters) == [
        "self",
        "raw",
        "purpose",
        "url",
        "options",
    ]

    raw = {"data": [{"url": "https://cdn.example/out"}]}
    assert await provider._await_media_result(raw, "image", "https://api.example/v1/images/generations", {}) is raw
    assert await provider._await_media_result(b"BYTES", "tts", "https://api.example/v1/audio/speech", {}) == b"BYTES"


@pytest.mark.asyncio
async def test_pipeline_calls_the_await_hook_once_before_output_handling() -> None:
    calls: list[dict] = []

    class _AsyncResultOpenAI(OpenAIProvider):
        async def _await_media_result(self, raw, purpose, url, options):
            calls.append({"raw": raw, "purpose": purpose, "url": url, "options": dict(options)})
            return {"data": [{"url": "https://cdn.example/awaited"}]}

    transport = _RecordingTransport()
    provider = _AsyncResultOpenAI(api_key="k", base_url="https://api.example/v1", transport=transport)

    response = await provider.image("p", "m")

    assert len(calls) == 1
    assert calls[0]["purpose"] == "image"
    assert calls[0]["url"] == "https://api.example/v1/images/generations"
    assert calls[0]["raw"] == {"data": [{"url": "https://cdn.example/out"}]}  # the transport payload
    (mc,) = response.media
    assert mc.data == "https://cdn.example/awaited"  # the hook's payload is what gets extracted

    # the handler sits after the hook: it sees the resolved payload
    seen: dict = {}

    def handler(payload) -> CLIverResponse:
        seen["payload"] = payload
        return _response("handled")

    response = await provider.image("p", "m", output_handler=handler)
    assert seen["payload"] == {"data": [{"url": "https://cdn.example/awaited"}]}
    assert response.message.content == "handled"


@pytest.mark.asyncio
async def test_await_hook_failure_lands_in_the_friendly_response() -> None:
    class _BoomOpenAI(OpenAIProvider):
        async def _await_media_result(self, raw, purpose, url, options):
            raise RuntimeError("task exploded")

    transport = _RecordingTransport()
    provider = _BoomOpenAI(api_key="k", base_url="https://api.example/v1", transport=transport)

    response = await provider.image("p", "m")

    assert response.media is None
    assert "openai image generation failed: task exploded" in response.message.content


@pytest.mark.asyncio
async def test_failure_response_carries_the_error_retryability() -> None:
    """The friendly failure keeps the swallowed error's retryable flag, so
    AgentCore can fail over to the next same-purpose binding."""

    class _RateLimitedOpenAI(OpenAIProvider):
        async def _await_media_result(self, raw, purpose, url, options):
            raise ProviderError("rate limited", retryable=True)

    class _RejectedOpenAI(OpenAIProvider):
        async def _await_media_result(self, raw, purpose, url, options):
            raise RuntimeError("bad prompt")

    retryable = await _RateLimitedOpenAI(
        api_key="k", base_url="https://api.example/v1", transport=_RecordingTransport()
    ).image("p", "m")
    assert retryable.media is None
    assert retryable.retryable is True

    rejected = await _RejectedOpenAI(
        api_key="k", base_url="https://api.example/v1", transport=_RecordingTransport()
    ).image("p", "m")
    assert rejected.media is None
    assert rejected.retryable is False


@pytest.mark.asyncio
async def test_failure_response_classifies_vendor_sdk_errors() -> None:
    """Raw SDK exceptions (not our ProviderError) use the same retry classifier
    as the chat transport boundary — a vendor 429 must still fail over."""

    class _VendorRateLimit(Exception):
        status_code = 429

    class _Vendor429OpenAI(OpenAIProvider):
        async def _post_media(self, url, body, *, files=None, binary=False, headers=None):
            raise _VendorRateLimit("rate limited")

    response = await _Vendor429OpenAI(
        api_key="k", base_url="https://api.example/v1", transport=_RecordingTransport()
    ).image("p", "m")

    assert response.media is None
    assert response.retryable is True
