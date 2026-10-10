"""AgentCore embedded model APIs — bindings gating, binding pass-through, MediaJob.

No network: the standalone capability functions are replaced by recorders, and
the assertions check that AgentCore passes the bound ``ModelBinding`` through
unchanged (model config stays encapsulated in it) and forwards only call-site
parameters.  Provider construction (bindings' lazy providers) is real but
inert — no request is ever sent.
"""

from __future__ import annotations

import asyncio

import pytest

from cliver_core import capabilities
from cliver_core.agent import AgentCore
from cliver_core.bindings import CapabilityNotSupported, ModelBinding, ModelBindings
from cliver_core.media_jobs import MediaJob
from cliver_core.messages import CLIverMessage
from cliver_core.provider import (
    CLIverRequest,
    CLIverResponse,
    EmbeddingsResult,
    Provider,
    RerankItem,
)


class _StubProvider(Provider):
    """Chat construction / provider-identity checks — chat is never exercised."""

    protocol = "mock"
    brand = "mock"

    def __init__(self) -> None:
        super().__init__(api_key="", base_url="")
        self.closed = False

    async def close(self) -> None:
        self.closed = True

    async def chat(self, request: CLIverRequest) -> CLIverResponse:  # pragma: no cover
        raise NotImplementedError

    async def stream(self, request: CLIverRequest):  # pragma: no cover
        if False:
            yield None


def _response(text: str) -> CLIverResponse:
    return CLIverResponse(message=CLIverMessage(role="assistant", content=text))


def _record(monkeypatch, **results) -> dict[str, list[dict]]:
    """Replace capability functions with recorders returning *results*."""
    calls: dict[str, list[dict]] = {name: [] for name in results}

    for name, result in results.items():

        def make(name=name, result=result):
            async def fake(**kwargs):
                calls[name].append(kwargs)
                return result

            return fake

        monkeypatch.setattr(capabilities, name, make())
    return calls


def _agent(llm: ModelBinding | None = None, **bindings: ModelBinding) -> AgentCore:
    """AgentCore over stub providers — ``llm`` included, so no test can
    accidentally reach ``create_provider`` (and the network)."""
    return AgentCore(
        ModelBindings(llm=llm or ModelBinding.llm(model="main-model", provider=_StubProvider()), **bindings)
    )


# ── Gating ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_unbound_purpose_raises_not_supported() -> None:
    agent = _agent()
    with pytest.raises(CapabilityNotSupported, match="image"):
        await agent.image("a cat")
    with pytest.raises(CapabilityNotSupported, match="asr"):
        await agent.asr(b"RIFF")
    with pytest.raises(CapabilityNotSupported, match="embed"):
        await agent.embed("hello")


def test_submit_unbound_purpose_fails_fast_synchronously() -> None:
    agent = _agent()
    with pytest.raises(CapabilityNotSupported, match="image"):
        agent.submit_image("a cat")


def test_submit_outside_event_loop_raises() -> None:
    agent = _agent(image=ModelBinding.image(model="img-model"))
    with pytest.raises(RuntimeError, match="running event loop"):
        agent.submit_image("a cat")


# ── Parameter construction / routing ───────────────────────────


def test_llm_binding_is_the_chat_handle() -> None:
    provider = _StubProvider()
    bindings = ModelBindings(llm=ModelBinding.llm(model="main-model", provider=provider))
    agent = AgentCore(bindings, builtin_tools=[])

    assert agent.bindings is bindings
    assert agent.llm is bindings.llm  # the binding itself — chat is not re-wrapped
    assert agent.llm.model == "main-model"
    assert agent.llm.provider is provider


def test_missing_llm_binding_fails_fast() -> None:
    with pytest.raises(ValueError, match=r"bindings\.llm"):
        AgentCore(ModelBindings())
    with pytest.raises(ValueError, match=r"bindings\.llm"):
        AgentCore(ModelBindings(image=ModelBinding.image(model="img-model")))
    with pytest.raises(ValueError, match=r"bindings\.llm"):
        AgentCore(None)  # type: ignore[arg-type]


def test_wrongly_mounted_llm_binding_fails_fast() -> None:
    with pytest.raises(ValueError, match=r"mount it in bindings\.image"):
        AgentCore(ModelBindings(llm=ModelBinding.image(model="img-model")))


@pytest.mark.asyncio
async def test_llm_provider_is_built_lazily_and_closed_by_aclose(monkeypatch) -> None:
    built: list[_StubProvider] = []

    def fake_create_provider(**kwargs):
        built.append(_StubProvider())
        return built[-1]

    monkeypatch.setattr("cliver_core.bindings.create_provider", fake_create_provider)

    binding = ModelBinding.llm(model="main-model", api_key="k", base_url="https://llm.example/v1")
    agent = _agent(llm=binding)
    assert binding.provider is None  # lazy by design — nothing built at construction
    assert built == []

    first = agent.llm.get_provider()
    assert first is built[0]  # built once, then cached
    assert binding.get_provider() is first

    await agent.aclose()
    assert first.closed is True  # aclose() closes the chat provider the binding built
    rebuilt = binding.get_provider()
    assert rebuilt is not first  # next use builds a fresh one
    await agent.aclose()


@pytest.mark.asyncio
async def test_caller_supplied_llm_provider_is_never_closed() -> None:
    stub = _StubProvider()
    agent = _agent(llm=ModelBinding.llm(model="main-model", provider=stub))

    await agent.aclose()
    assert stub.closed is False  # whoever built it owns it


def test_purpose_factories_fold_purpose_params() -> None:
    llm = ModelBinding.llm(model="m", temperature=0.3, top_p=0.9)
    assert llm.params == {"temperature": 0.3, "top_p": 0.9}
    assert llm.purpose == "llm"

    image = ModelBinding.image(model="m", size="1024x1024", n=2, style="vivid", seed=7)
    assert image.params == {"size": "1024x1024", "n": 2, "style": "vivid", "seed": 7}

    video = ModelBinding.video(model="m", duration=5)
    assert video.params == {"duration": 5}

    tts = ModelBinding.tts(model="m")  # unset optionals are omitted
    assert tts.params == {}

    asr = ModelBinding.asr(model="m", language="en")
    assert asr.params == {"language": "en"}

    embedded = ModelBinding.embed(model="m", dimensions=256)
    assert embedded.params == {"dimensions": 256}

    rerank = ModelBinding.rerank(model="m", top_k=3)
    assert rerank.params == {"top_k": 3}


@pytest.mark.asyncio
async def test_image_passes_the_binding_and_call_site_params(monkeypatch) -> None:
    calls = _record(monkeypatch, generate_image=_response("generated image"))
    binding = ModelBinding.image(
        model="image-model",
        api_key="img-key",
        base_url="https://img.example/v1",
        protocol="anthropic",  # minimax media rides the brand's only leaf
        provider_name="minimax",
        size="1024x1024",  # binding default
        style="vivid",  # binding default
    )
    agent = _agent(image=binding)

    response = await agent.image("a red panda", output_dir="/tmp/out", n=2, size="512x512")
    assert response.message.content == "generated image"

    call = calls["generate_image"][0]
    # the binding itself is the model handle — config is never re-passed
    assert call["binding"] is binding
    assert "model" not in call and "provider" not in call and "api_key" not in call and "base_url" not in call
    assert call["prompt"] == "a red panda"
    assert call["output_dir"] == "/tmp/out"
    # only call-site parameters are forwarded; binding defaults stay in the binding
    assert call["n"] == 2
    assert call["size"] == "512x512"
    assert binding.params["style"] == "vivid"

    # the binding still owns a real, lazily built provider
    provider = binding.get_provider()
    assert isinstance(provider, Provider)
    assert provider.api_key == "img-key"
    assert provider.base_url == "https://img.example/v1"
    assert provider.provider_name() == "minimax"

    await agent.aclose()


@pytest.mark.asyncio
async def test_binding_is_passed_through_unchanged_and_provider_built_once(monkeypatch) -> None:
    calls = _record(monkeypatch, generate_image=_response("img"))
    binding = ModelBinding.image(model="image-model", api_key="k", base_url="https://img.example/v1")
    agent = _agent(image=binding)

    await agent.image("one")
    await agent.image("two")

    assert calls["generate_image"][0]["binding"] is binding  # same handle on every call
    assert calls["generate_image"][1]["binding"] is binding
    first = binding.get_provider()
    assert binding.get_provider() is first  # built once, cached — no per-call construction

    await agent.aclose()
    rebuilt = binding.get_provider()
    assert rebuilt is not first  # closed on aclose(); next use builds a fresh one
    await agent.aclose()


@pytest.mark.asyncio
async def test_caller_supplied_provider_used_as_is_and_not_closed(monkeypatch) -> None:
    calls = _record(monkeypatch, generate_image=_response("img"))
    stub = _StubProvider()
    binding = ModelBinding.image(model="image-model", provider=stub)
    agent = _agent(image=binding)

    await agent.image("one")
    assert calls["generate_image"][0]["binding"].provider is stub

    await agent.aclose()
    assert stub.closed is False  # caller-owned providers are never closed by aclose()


@pytest.mark.asyncio
async def test_tts_and_legacy_audio_use_their_own_bindings(monkeypatch) -> None:
    calls = _record(monkeypatch, text_to_speech=_response("spoken"))
    tts_binding = ModelBinding.tts(
        model="tts-model", api_key="tts-key", base_url="https://tts.example/v1", voice="alloy"
    )
    audio_binding = ModelBinding.tts(model="audio-model", api_key="audio-key", base_url="https://audio.example/v1")
    agent = _agent(tts=tts_binding, audio=audio_binding)

    await agent.tts("say hi", voice="nova")  # call-site voice forwarded
    first = calls["text_to_speech"][0]
    assert first["binding"] is tts_binding
    assert first["text"] == "say hi"
    assert first["voice"] == "nova"

    await agent.tts("again")  # no call-site voice — the binding default stays in the binding
    second = calls["text_to_speech"][1]
    assert "voice" not in second
    assert tts_binding.params["voice"] == "alloy"

    await agent.audio("say hi")  # legacy alias → audio binding
    third = calls["text_to_speech"][2]
    assert third["binding"] is audio_binding
    assert third["binding"].model == "audio-model"

    await agent.aclose()


@pytest.mark.asyncio
async def test_video_passes_binding_and_call_params(monkeypatch) -> None:
    calls = _record(monkeypatch, generate_video=_response("generated video"))
    binding = ModelBinding.video(
        model="video-model",
        provider=_StubProvider(),
        path="/videos/generations",
        resolution="1080p",
    )
    agent = _agent(video=binding)

    await agent.video("a sunset", duration=5)

    call = calls["generate_video"][0]
    assert call["binding"] is binding
    assert call["prompt"] == "a sunset"
    assert call["duration"] == 5  # call-site params only
    assert binding.path == "/videos/generations"  # endpoint config lives on the binding
    assert binding.params["resolution"] == "1080p"

    await agent.aclose()


@pytest.mark.asyncio
async def test_binding_path_stays_inside_the_binding(monkeypatch) -> None:
    calls = _record(monkeypatch, generate_image=_response("img"))
    agent = _agent(image=ModelBinding.image(model="image-model", provider=_StubProvider(), path="/brand/default"))

    await agent.image("one")  # endpoint config travels inside the binding
    call = calls["generate_image"][0]
    assert "path" not in call
    assert call["binding"].path == "/brand/default"

    await agent.aclose()


def test_mounting_a_binding_under_the_wrong_purpose_fails_fast() -> None:
    agent = _agent(image=ModelBinding.video(model="video-model"))  # video binding in the image slot
    with pytest.raises(ValueError, match=r"mount it in bindings\.video"):
        agent.submit_image("a cat")


@pytest.mark.asyncio
async def test_asr_forwards_source_language_and_binding(monkeypatch) -> None:
    calls = _record(monkeypatch, transcribe_audio=_response("transcript"))
    binding = ModelBinding.asr(model="whisper-1", api_key="k", base_url="https://asr.example/v1", language="zh")
    agent = _agent(asr=binding)

    payload = b"RIFF...."
    await agent.asr(payload)  # no call-site language — binding default applies downstream
    call = calls["transcribe_audio"][0]
    assert call["binding"] is binding
    # source normalization (bytes → MediaContent) happens inside the capability
    assert call["audio"] is payload
    assert call["language"] is None
    assert binding.params["language"] == "zh"

    await agent.asr(payload, language="en")  # call-site overrides
    assert calls["transcribe_audio"][1]["language"] == "en"

    await agent.aclose()


@pytest.mark.asyncio
async def test_embed_and_rerank_route_to_bound_models(monkeypatch) -> None:
    calls = _record(
        monkeypatch,
        embed=EmbeddingsResult(vectors=[[0.1, 0.2]], model="embed-model"),
        rerank=[RerankItem(index=0, relevance_score=0.9, document="doc a")],
    )
    embed_binding = ModelBinding.embed(
        model="embed-model", api_key="ek", base_url="https://e.example/v1", dimensions=256
    )
    rerank_binding = ModelBinding.rerank(model="rerank-model", api_key="rk", base_url="https://r.example/v1", top_k=5)
    agent = _agent(embed=embed_binding, rerank=rerank_binding)

    result = await agent.embed("hello")
    embed_call = calls["embed"][0]
    assert embed_call["binding"] is embed_binding
    assert embed_call["texts"] == "hello"
    assert "dimensions" not in embed_call  # binding default lives in the binding
    assert embed_binding.params["dimensions"] == 256
    assert len(result.vectors) == 1

    items = await agent.rerank("q", ["doc a", "doc b"])  # no call-site top_k
    assert calls["rerank"][0]["binding"] is rerank_binding
    assert calls["rerank"][0]["top_k"] is None
    assert rerank_binding.params["top_k"] == 5
    await agent.rerank("q", ["doc a", "doc b"], top_k=1)  # call-site wins
    assert calls["rerank"][1]["top_k"] == 1
    assert items[0].document == "doc a"

    await agent.aclose()


# ── MediaJob futures ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_submit_returns_job_with_result(monkeypatch) -> None:
    _record(monkeypatch, generate_image=_response("generated image"))
    agent = _agent(image=ModelBinding.image(model="image-model", provider=_StubProvider()))

    job = agent.submit_image("a red panda")
    assert isinstance(job, MediaJob)
    assert job.purpose == "image"
    assert job.status in ("running", "done")

    response = await job.result()
    assert response.message.content == "generated image"
    assert job.done()
    assert job.status == "done"
    assert repr(job).startswith("MediaJob(")


@pytest.mark.asyncio
async def test_failed_job_reports_failed_status(monkeypatch) -> None:
    async def boom(**kwargs):
        raise RuntimeError("generation exploded")

    monkeypatch.setattr(capabilities, "generate_image", boom)
    agent = _agent(image=ModelBinding.image(model="image-model", provider=_StubProvider()))

    job = agent.submit_image("boom")
    with pytest.raises(RuntimeError, match="exploded"):
        await job.result()
    assert job.status == "failed"


@pytest.mark.asyncio
async def test_result_timeout_does_not_cancel_job(monkeypatch) -> None:
    _record(monkeypatch, generate_image=_response("generated image"))
    agent = _agent(image=ModelBinding.image(model="image-model", provider=_StubProvider()))

    job = agent.submit_image("slow")
    try:
        await job.result(timeout=0)  # already-past deadline
    except (asyncio.TimeoutError, TimeoutError):
        pass
    else:  # a very fast machine may finish before the timeout fires
        assert job.done()

    response = await job.result()  # job kept running — still collectable
    assert response.message.content == "generated image"


@pytest.mark.asyncio
async def test_cancel_marks_job_cancelled(monkeypatch) -> None:
    started = asyncio.Event()

    async def slow(**kwargs):
        started.set()
        await asyncio.sleep(30)
        return _response("late")

    monkeypatch.setattr(capabilities, "generate_image", slow)
    agent = _agent(image=ModelBinding.image(model="image-model", provider=_StubProvider()))

    job = agent.submit_image("slow")
    await asyncio.wait_for(started.wait(), timeout=5)  # fail fast, don't hang the suite
    assert job.cancel() is True
    await asyncio.sleep(0)
    assert job.status == "cancelled"


def test_media_job_status_before_completion_is_running() -> None:
    async def _scenario() -> None:
        async def _never() -> str:
            await asyncio.sleep(30)
            return "x"

        job = MediaJob("image", asyncio.get_running_loop().create_task(_never()))
        assert job.status == "running"
        job.cancel()

    asyncio.run(_scenario())
