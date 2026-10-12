"""Standalone capability calls — usable out of the box, no AgentCore required.

Every capability call takes a :class:`~cliver_core.bindings.ModelBinding` —
the one handle carrying the model definition, its (cached) provider, the
endpoint path override, and the default parameters.  The capability functions
forward the constructed parameters to the provider; model configuration is
never passed as a loose argument.
"""

from __future__ import annotations

import base64
import inspect

import pytest

from cliver_core.bindings import ModelBinding
from cliver_core.capabilities import (
    _base,
    embed,
    generate_image,
    generate_video,
    rerank,
    text_to_speech,
    transcribe_audio,
)
from cliver_core.media import MediaContent, MediaType
from cliver_core.messages import CLIverMessage
from cliver_core.provider import CLIverRequest, CLIverResponse, EmbeddingsResult, Provider, RerankItem


class _RecordingProvider(Provider):
    """Records what each capability call forwards to the provider."""

    protocol = "mock"
    brand = "mock"

    def __init__(self) -> None:
        super().__init__(api_key="k", base_url="https://example/v1")
        self.image_calls: list[dict] = []
        self.video_calls: list[dict] = []
        self.tts_calls: list[dict] = []
        self.asr_calls: list[dict] = []
        self.rerank_calls: list[dict] = []
        self.embed_calls: list[dict] = []
        self.closed = False

    async def close(self) -> None:
        self.closed = True

    async def chat(self, request: CLIverRequest) -> CLIverResponse:  # pragma: no cover
        raise NotImplementedError

    async def stream(self, request: CLIverRequest):  # pragma: no cover
        if False:
            yield None

    async def image(self, prompt, model, *, media=None, output_dir=None, **options):
        self.image_calls.append({"prompt": prompt, "model": model, "media": media, "output_dir": output_dir, **options})
        return CLIverResponse(message=CLIverMessage(role="assistant", content="generated image"))

    async def video(self, prompt, model, *, media=None, output_dir=None, **options):
        self.video_calls.append({"prompt": prompt, "model": model, "media": media, "output_dir": output_dir, **options})
        return CLIverResponse(message=CLIverMessage(role="assistant", content="generated video"))

    async def tts(self, text, model, *, output_dir=None, **options):
        self.tts_calls.append({"text": text, "model": model, "output_dir": output_dir, **options})
        return CLIverResponse(message=CLIverMessage(role="assistant", content="synthesized speech"))

    async def asr(self, media, model, *, language=None, **options):
        self.asr_calls.append({"media": media, "model": model, "language": language, **options})
        return CLIverResponse(message=CLIverMessage(role="assistant", content="transcribed"))

    async def rerank(self, query, documents, *, model, top_k=None, **options):
        self.rerank_calls.append({"query": query, "documents": documents, "model": model, "top_k": top_k})
        return [RerankItem(index=0, relevance_score=0.9, document=documents[0])]

    async def embed(self, texts, *, model, **options):
        self.embed_calls.append({"texts": texts, "model": model, **options})
        return EmbeddingsResult(vectors=[[0.1] * 3 for _ in texts], model=model)


# ── The binding is the only model handle ───────────────────────


def test_capability_signatures_take_the_binding_not_model_config() -> None:
    loose = {"model", "api_key", "base_url", "protocol", "provider_name", "provider"}
    funcs = (generate_image, generate_video, text_to_speech, transcribe_audio, embed, rerank)
    for fn in funcs:
        params = inspect.signature(fn).parameters
        assert not set(params) & loose, f"{fn.__name__} still takes {set(params) & loose}"
        assert params["binding"].default is inspect.Parameter.empty  # binding is required


@pytest.mark.asyncio
async def test_capability_requires_a_binding() -> None:
    with pytest.raises(TypeError, match="binding"):
        await generate_image(prompt="p")


@pytest.mark.asyncio
async def test_call_site_path_overrides_the_binding_path() -> None:
    provider = _RecordingProvider()
    binding = ModelBinding.image(model="m", provider=provider, path="/binding/gen")

    await generate_image(prompt="p", binding=binding, path="/call/gen")
    assert provider.image_calls[-1]["path"] == "/call/gen"  # call-site wins

    await generate_image(prompt="p", binding=binding)
    assert provider.image_calls[-1]["path"] == "/binding/gen"  # binding default otherwise

    # a call-site path also works with no binding path at all
    await generate_image(prompt="p", binding=ModelBinding.image(model="m", provider=provider), path="/call/gen")
    assert provider.image_calls[-1]["path"] == "/call/gen"


# ── Provider construction & lifecycle — owned by the binding ───


@pytest.mark.asyncio
async def test_binding_builds_its_provider_from_the_model_definition(monkeypatch) -> None:
    provider = _RecordingProvider()
    seen: dict = {}

    def fake_create_provider(api_key=None, base_url=None, *, protocol="openai", provider_name=None, **kwargs):
        seen.update(api_key=api_key, base_url=base_url, protocol=protocol, provider_name=provider_name)
        return provider

    monkeypatch.setattr("cliver_core.bindings.create_provider", fake_create_provider)

    binding = ModelBinding.video(
        model="video-01",
        api_key="vk",
        base_url="https://v.example/v1",
        provider_name="minimax",
    )
    response = await generate_video(prompt="a drone shot", binding=binding)

    assert response.message.content == "generated video"
    assert seen == {
        "api_key": "vk",
        "base_url": "https://v.example/v1",
        "protocol": "openai",
        "provider_name": "minimax",
    }
    assert provider.video_calls[0]["model"] == "video-01"
    assert provider.closed is False  # the binding owns the provider; the call never closes it
    assert binding.get_provider() is provider  # built once, then reused

    await binding.aclose()
    assert provider.closed is True  # only the binding closes what it built


@pytest.mark.asyncio
async def test_caller_supplied_provider_is_used_as_is_and_never_closed() -> None:
    provider = _RecordingProvider()
    binding = ModelBinding.image(model="m", provider=provider)

    await generate_image(prompt="p", binding=binding)
    assert provider.image_calls[0]["model"] == "m"
    assert provider.closed is False

    await binding.aclose()
    assert provider.closed is False  # caller-owned providers are never closed by the binding


# ── Parameter construction ─────────────────────────────────────


@pytest.mark.asyncio
async def test_generate_image_forwards_params_and_omits_unset_ones() -> None:
    provider = _RecordingProvider()
    binding = ModelBinding.image(model="gpt-image-1", provider=provider)

    await generate_image(
        prompt="a red panda",
        binding=binding,
        n=2,
        size="512x512",
        style="vivid",
        seed=7,  # passthrough option
    )
    call = provider.image_calls[0]
    assert call["prompt"] == "a red panda"
    assert call["model"] == "gpt-image-1"  # model name comes from the binding
    assert call["n"] == 2
    assert call["size"] == "512x512"
    assert call["style"] == "vivid"
    assert call["seed"] == 7

    await generate_image(prompt="p", binding=binding)
    call = provider.image_calls[1]
    assert "size" not in call
    assert "n" not in call
    assert "style" not in call


@pytest.mark.asyncio
async def test_text_to_speech_forwards_voice_and_format() -> None:
    provider = _RecordingProvider()
    binding = ModelBinding.tts(model="tts-1", provider=provider)

    await text_to_speech(text="hello", binding=binding, voice="alloy", response_format="wav")
    call = provider.tts_calls[0]
    assert call["text"] == "hello"
    assert call["model"] == "tts-1"
    assert call["voice"] == "alloy"
    assert call["response_format"] == "wav"

    await text_to_speech(text="hi", binding=binding)
    assert "voice" not in provider.tts_calls[1]


@pytest.mark.asyncio
async def test_transcribe_audio_normalizes_source_forms(tmp_path) -> None:
    provider = _RecordingProvider()
    binding = ModelBinding.asr(model="whisper-1", provider=provider)

    # raw bytes → wrapped as audio/wav MediaContent
    await transcribe_audio(audio=b"RIFF....", binding=binding, language="en")
    call = provider.asr_calls[0]
    assert call["model"] == "whisper-1"
    assert call["language"] == "en"
    assert call["media"].mime_type == "audio/wav"
    assert base64.b64decode(call["media"].data) == b"RIFF...."

    # MediaContent passes through untouched
    mc = MediaContent(type=MediaType.AUDIO, data="AAAA", mime_type="audio/mpeg")
    await transcribe_audio(audio=mc, binding=binding)
    assert provider.asr_calls[1]["media"] is mc

    # path → loaded from disk (mimetypes may report audio/wav or audio/x-wav)
    wav = tmp_path / "clip.wav"
    wav.write_bytes(b"RIFF....")
    await transcribe_audio(audio=str(wav), binding=binding)
    assert provider.asr_calls[2]["media"].mime_type.startswith("audio/")


@pytest.mark.asyncio
async def test_embed_wraps_bare_string_and_rerank_forwards_top_k() -> None:
    provider = _RecordingProvider()

    result = await embed(texts="hello", binding=ModelBinding.embed(model="text-embedding-3-small", provider=provider))
    assert provider.embed_calls[0] == {"texts": ["hello"], "model": "text-embedding-3-small"}
    assert len(result.vectors) == 1

    items = await rerank(
        query="q",
        documents=["doc a", "doc b"],
        binding=ModelBinding.rerank(model="rerank-v1", provider=provider),
        top_k=1,
    )
    assert provider.rerank_calls[0]["query"] == "q"
    assert provider.rerank_calls[0]["top_k"] == 1
    assert items[0].document == "doc a"


# ── Binding defaults — model config stays in the ModelBinding ──


def test_binding_call_defaults_carry_path_and_params() -> None:
    binding = ModelBinding.image(model="m", path="/brand/gen", size="512x512", n=2)
    assert binding.call_defaults() == {"path": "/brand/gen", "size": "512x512", "n": 2}
    # an unset path is omitted entirely, never sent as None
    assert "path" not in ModelBinding.image(model="m").call_defaults()


@pytest.mark.asyncio
async def test_binding_supplies_model_provider_and_defaults() -> None:
    provider = _RecordingProvider()
    binding = ModelBinding.image(model="bind-model", provider=provider, path="/brand/gen", size="512x512", n=2)

    await generate_image(prompt="p", binding=binding)
    call = provider.image_calls[-1]
    assert call["model"] == "bind-model"  # model name comes from the binding
    assert call["path"] == "/brand/gen"  # binding endpoint override applies
    assert call["size"] == "512x512"
    assert call["n"] == 2
    assert call["prompt"] == "p"
    assert provider.closed is False  # the binding's provider stays open

    # unset call-site params leave the binding defaults in place …
    await generate_image(prompt="p", binding=binding)
    call = provider.image_calls[-1]
    assert call["size"] == "512x512"
    assert call["path"] == "/brand/gen"

    # … while call-site values win on conflict (endpoint config stays with the binding)
    await generate_image(prompt="p", binding=binding, size="1024x1024")
    call = provider.image_calls[-1]
    assert call["size"] == "1024x1024"
    assert call["path"] == "/brand/gen"


@pytest.mark.asyncio
async def test_binding_covers_every_capability_shape() -> None:
    provider = _RecordingProvider()

    await generate_video(prompt="p", binding=ModelBinding.video(model="vm", provider=provider, duration=5))
    assert provider.video_calls[-1]["model"] == "vm"
    assert provider.video_calls[-1]["duration"] == 5

    await text_to_speech(text="hi", binding=ModelBinding.tts(model="tm", provider=provider, voice="alloy"))
    assert provider.tts_calls[-1]["model"] == "tm"
    assert provider.tts_calls[-1]["voice"] == "alloy"

    await transcribe_audio(audio=b"RIFF", binding=ModelBinding.asr(model="am", provider=provider, language="zh"))
    assert provider.asr_calls[-1]["model"] == "am"
    assert provider.asr_calls[-1]["language"] == "zh"

    await embed(texts="hello", binding=ModelBinding.embed(model="em", provider=provider, dimensions=256))
    assert provider.embed_calls[-1]["model"] == "em"
    assert provider.embed_calls[-1]["dimensions"] == 256

    await rerank(query="q", documents=["d"], binding=ModelBinding.rerank(model="rm", provider=provider, top_k=3))
    assert provider.rerank_calls[-1]["model"] == "rm"
    assert provider.rerank_calls[-1]["top_k"] == 3


# ── Helpers ────────────────────────────────────────────────────


def test_drop_none_omits_unset_values() -> None:
    assert _base.drop_none(a=1, b=None, c="x") == {"a": 1, "c": "x"}


def test_coerce_media_passthrough_and_bytes() -> None:
    mc = MediaContent(type=MediaType.AUDIO, data="AAAA", mime_type="audio/mpeg")
    assert _base.coerce_media(mc) is mc

    wrapped = _base.coerce_media(b"RIFF")
    assert wrapped.mime_type == "audio/wav"
    assert base64.b64decode(wrapped.data) == b"RIFF"


# ── Out of the box — real brand providers through a binding ────


@pytest.mark.asyncio
async def test_generate_image_and_tts_work_out_of_the_box_on_a_brand_provider(monkeypatch, tmp_path) -> None:
    """The standalone calls against the real MiniMax provider class: binding →
    provider → absolute brand URL on the shared transport → media out.
    No AgentCore anywhere in this test."""
    from cliver_core.provider.providers import anthropic as anthropic_module

    jpeg = b"\xff\xd8\xff\xe0out-of-box-bytes"
    tts_hex = b"ID3\x00speech".hex()

    class _StubTransport:
        def __init__(self) -> None:
            self.posts: list[dict] = []
            self.queued = [
                {
                    "base_resp": {"status_code": 0},
                    "data": {"image_urls": ["https://cdn.example/out_aigc.jpeg?Expires=1"]},
                },
                {"base_resp": {"status_code": 0}, "data": {"audio": tts_hex}},
            ]

        async def post(self, path, body, *, files=None, binary=False):
            self.posts.append({"path": path, "body": body, "files": files, "binary": binary})
            return self.queued.pop(0)

    transport = _StubTransport()
    monkeypatch.setattr(anthropic_module, "resolve_brand_sdk", lambda *args, **kwargs: transport)
    monkeypatch.setattr("cliver_core.media._download_url", lambda url: jpeg)

    image_binding = ModelBinding.image(
        model="image-01",
        api_key="k",
        base_url="https://api.minimaxi.com/anthropic",
        protocol="anthropic",
        provider_name="minimax",
    )
    response = await generate_image(prompt="a red circle", binding=image_binding, output_dir=str(tmp_path))

    call = transport.posts[-1]
    assert call["path"] == "https://api.minimaxi.com/v1/image_generation"
    assert call["body"] == {"model": "image-01", "prompt": "a red circle"}
    (mc,) = response.media
    assert mc.mime_type == "image/jpeg"  # labeled by the URL, not the png default
    saved = tmp_path / "generated_0.jpg"
    assert mc.saved_path == str(saved) and saved.read_bytes() == jpeg

    tts_binding = ModelBinding.tts(
        model="speech-2.8-hd",
        api_key="k",
        base_url="https://api.minimaxi.com/anthropic",
        protocol="anthropic",
        provider_name="minimax",
    )
    response = await text_to_speech(text="hello", binding=tts_binding)

    call = transport.posts[-1]
    assert call["path"] == "https://api.minimaxi.com/v1/t2a_v2"
    assert call["body"]["text"] == "hello"
    (mc,) = response.media
    assert mc.mime_type == "audio/mpeg"
    assert base64.b64decode(mc.data) == bytes.fromhex(tts_hex)


@pytest.mark.asyncio
async def test_transcribe_audio_works_out_of_the_box_on_the_openai_protocol(monkeypatch) -> None:
    from cliver_core.provider.providers import openai as openai_module

    class _StubTransport:
        def __init__(self) -> None:
            self.posts: list[dict] = []

        async def post(self, path, body, *, files=None, binary=False):
            self.posts.append({"path": path, "body": body, "files": files, "binary": binary})
            return {"text": "hello world"}

    transport = _StubTransport()
    monkeypatch.setattr(openai_module, "resolve_brand_sdk", lambda *args, **kwargs: transport)

    binding = ModelBinding.asr(model="whisper-1", api_key="k", base_url="https://api.example/v1")
    response = await transcribe_audio(audio=b"RIFF....", binding=binding, language="en")

    call = transport.posts[-1]
    assert call["path"] == "https://api.example/v1/audio/transcriptions"
    assert call["body"] == {"model": "whisper-1", "language": "en"}
    assert call["files"] == {"file": ("audio.wav", b"RIFF....", "audio/wav")}
    assert response.message.content == "hello world"


@pytest.mark.asyncio
async def test_transcribe_audio_works_out_of_the_box_on_the_minimax_brand(monkeypatch) -> None:
    """The standalone call against the real MiniMax provider class, through
    the preset: binding → provider → absolute brand URL on the shared
    transport — with the language routed to a BCP-47 HTTP header."""
    from cliver_core import presets
    from cliver_core.provider.providers import anthropic as anthropic_module

    class _StubTransport:
        def __init__(self) -> None:
            self.posts: list[dict] = []

        async def post(self, path, body, *, files=None, binary=False, headers=None):
            self.posts.append({"path": path, "body": body, "files": files, "binary": binary, "headers": headers})
            return {"text": "the quick brown fox", "duration": 3.2, "trace_id": "t1"}

        async def close(self) -> None:
            pass

    transport = _StubTransport()
    monkeypatch.setattr(anthropic_module, "resolve_brand_sdk", lambda *args, **kwargs: transport)

    binding = presets.minimax_asr(api_key="k")
    response = await transcribe_audio(audio=b"RIFF....", binding=binding, language="en")

    call = transport.posts[-1]
    assert call["path"] == "https://api.minimaxi.com/v1/speech_to_text"
    assert call["body"] == {"model": "asr-1.0"}
    assert call["headers"] == {"language": "en"}
    assert call["files"] == {"file": ("audio.wav", b"RIFF....", "audio/wav")}
    assert response.message.content == "the quick brown fox"
    await binding.aclose()
