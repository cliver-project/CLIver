"""Seedance (Volcano Engine Ark) — the async video-task wire.

The create POST goes through an injected transport stub; the poll read
(``_fetch_task``), the poll cadence (``_sleep``), and the media download
(``cliver_core.media._download_url``) are monkeypatched — no network.  API
keys here are literal dummies; the suite never reads key environment
variables for real values.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from cliver_core.capabilities import generate_video
from cliver_core.media import MediaContent, MediaType
from cliver_core.messages import CLIverMessage
from cliver_core.provider import CLIverResponse
from cliver_core.provider.providers import seedance as seedance_module
from cliver_core.provider.providers.seedance import SeedanceProvider

_ARK_BASE = "https://ark.cn-beijing.volces.com/api/v3"
_TASK_URL = _ARK_BASE + "/contents/generations/tasks"
_MODEL = "doubao-seedance-2-5-260628"


class _CreateTransport:
    """Records the create POST; answers one canned task-submission payload."""

    def __init__(self, payload: Any = None) -> None:
        self.posts: list[dict] = []
        self.payload = {"id": "cgt-1"} if payload is None else payload

    async def post(self, path: str, body: dict, *, files=None, binary: bool = False, headers=None):
        self.posts.append({"path": path, "body": body, "binary": binary})
        return self.payload

    async def close(self) -> None:
        pass


class _PollStub:
    """Async-callable standing in for ``_fetch_task`` — replays canned payloads."""

    def __init__(self, *payloads: dict) -> None:
        self.payloads = list(payloads)
        self.calls: list[dict] = []

    async def __call__(self, task_id: str, *, url: str) -> dict:
        self.calls.append({"task_id": task_id, "url": url})
        return self.payloads.pop(0)


def _provider(transport: _CreateTransport) -> SeedanceProvider:
    return SeedanceProvider(api_key="k", base_url=_ARK_BASE, transport=transport)


def _stub_poll(monkeypatch, provider: SeedanceProvider, *payloads: dict) -> _PollStub:
    poller = _PollStub(*payloads)
    monkeypatch.setattr(provider, "_fetch_task", poller)
    return poller


def _succeeded(video_url: str = "https://tos.example/out.mp4?X-Tos=1") -> dict:
    return {"id": "cgt-1", "status": "succeeded", "content": {"video_url": video_url}}


# ── The Ark surface — descriptors ──────────────────────────────


def test_seedance_locks_the_ark_surface() -> None:
    provider = _provider(_CreateTransport())
    assert provider.provider_name() == "seedance"
    assert provider.base_url == _ARK_BASE
    assert provider._resolve_media_gen_url("video") == _TASK_URL
    # Ark has no OpenAI-fashion media endpoints: image/tts/asr fail fast...
    assert provider.media_path("image") is None
    assert provider.media_path("tts") is None
    assert provider.media_path("asr") is None
    # ...but the media-scoped opt-out leaves the protocol's embed/rerank alone
    assert provider.media_path("embed") is not None


# ── The create body — Ark's content array ──────────────────────


@pytest.mark.asyncio
async def test_create_body_is_the_ark_content_array(monkeypatch) -> None:
    transport = _CreateTransport()
    provider = _provider(transport)
    _stub_poll(monkeypatch, provider, _succeeded())

    await provider.video(
        "a panda eating bamboo",
        _MODEL,
        duration=5,
        resolution="720p",
        ratio="16:9",
        watermark=False,
        poll_interval=0,
    )

    call = transport.posts[-1]
    assert call["path"] == _TASK_URL
    assert call["binary"] is False  # the create answer is JSON
    assert call["body"] == {
        "model": _MODEL,
        "content": [{"type": "text", "text": "a panda eating bamboo"}],
        "duration": 5,
        "resolution": "720p",
        "ratio": "16:9",
        "watermark": False,
    }


@pytest.mark.asyncio
async def test_create_body_carries_reference_images_as_data_urls(monkeypatch) -> None:
    transport = _CreateTransport()
    provider = _provider(transport)
    _stub_poll(monkeypatch, provider, _succeeded())

    raw = MediaContent(type=MediaType.IMAGE, data="UE5H", mime_type="image/png")
    wrapped = MediaContent(type=MediaType.IMAGE, data="data:image/jpeg;base64,/9j/2Q==", mime_type="image/jpeg")
    await provider.video("make it move", _MODEL, media=[raw, wrapped], poll_interval=0)

    assert transport.posts[-1]["body"]["content"] == [
        {"type": "text", "text": "make it move"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,UE5H"}},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/2Q=="}},
    ]


@pytest.mark.asyncio
async def test_poll_knobs_never_reach_the_create_body(monkeypatch) -> None:
    transport = _CreateTransport()
    provider = _provider(transport)
    _stub_poll(monkeypatch, provider, _succeeded())

    await provider.video("p", _MODEL, poll_interval=0, poll_timeout=30)

    body = transport.posts[-1]["body"]
    assert "poll_interval" not in body
    assert "poll_timeout" not in body


# ── The poll loop — create → task → result ─────────────────────


@pytest.mark.asyncio
async def test_create_then_poll_to_succeeded_saves_the_mp4(monkeypatch, tmp_path) -> None:
    transport = _CreateTransport()
    provider = _provider(transport)
    poller = _stub_poll(
        monkeypatch,
        provider,
        {"id": "cgt-1", "status": "queued"},
        {"id": "cgt-1", "status": "running"},
        _succeeded(),
    )
    video = b"\x00\x00\x00 ftypisom" + b"\x00" * 40
    monkeypatch.setattr("cliver_core.media._download_url", lambda url: video)

    response = await provider.video("a panda", _MODEL, poll_interval=0, output_dir=str(tmp_path))

    assert [call["task_id"] for call in poller.calls] == ["cgt-1", "cgt-1", "cgt-1"]
    assert all(call["url"] == _TASK_URL for call in poller.calls)
    (mc,) = response.media
    assert mc.type == MediaType.VIDEO
    assert mc.mime_type == "video/mp4"
    saved = Path(mc.saved_path)
    assert saved.name == "generated_0.mp4"
    assert saved.read_bytes() == video
    assert response.message.content.startswith("Generated 1 video(s) via seedance.")


@pytest.mark.asyncio
async def test_missing_status_on_the_create_response_keeps_polling(monkeypatch) -> None:
    transport = _CreateTransport()
    provider = _provider(transport)
    poller = _stub_poll(monkeypatch, provider, {"id": "cgt-1"}, _succeeded())

    response = await provider.video("p", _MODEL, poll_interval=0)

    assert len(poller.calls) == 2  # the id-only create payload and the status-less read both keep polling
    assert response.media is not None


@pytest.mark.asyncio
async def test_poll_interval_is_passed_to_sleep(monkeypatch) -> None:
    transport = _CreateTransport({"id": "cgt-1", "status": "queued"})
    provider = _provider(transport)
    _stub_poll(monkeypatch, provider, _succeeded())
    intervals: list[float] = []

    async def _record(seconds: float) -> None:
        intervals.append(seconds)

    monkeypatch.setattr(seedance_module, "_sleep", _record)
    await provider.video("p", _MODEL, poll_interval=2.5)

    assert intervals == [2.5]


@pytest.mark.asyncio
async def test_task_url_derives_from_the_resolved_create_url(monkeypatch) -> None:
    """A call-site path= retargets the create POST — the poll must follow it,
    never fall back to the brand host."""
    transport = _CreateTransport()
    provider = _provider(transport)
    poller = _stub_poll(monkeypatch, provider, _succeeded())

    await provider.video("p", _MODEL, path="https://proxy.internal/ark/tasks", poll_interval=0)

    assert transport.posts[-1]["path"] == "https://proxy.internal/ark/tasks"
    assert poller.calls == [{"task_id": "cgt-1", "url": "https://proxy.internal/ark/tasks"}]


# ── Failures — all land in the friendly response ───────────────


@pytest.mark.asyncio
async def test_create_payload_without_a_task_id_fails_friendly(monkeypatch) -> None:
    transport = _CreateTransport({"code": "InvalidParameter", "message": "bad model"})
    provider = _provider(transport)
    poller = _stub_poll(monkeypatch, provider, _succeeded())

    response = await provider.video("p", _MODEL, poll_interval=0)

    assert response.media is None
    assert "no task id" in response.message.content
    assert poller.calls == []  # a phantom task is never polled


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status,error,expected",
    [
        ("failed", {"code": "PolicyViolation", "message": "content violates policy"}, "content violates policy"),
        ("failed", "quota exceeded", "quota exceeded"),
        ("cancelled", None, "cancelled"),
        ("expired", {"code": "TaskExpired"}, "TaskExpired"),
        ("processing", None, "processing"),  # unknown states are terminal, not polled forever
    ],
)
async def test_terminal_task_states_fail_friendly(monkeypatch, status, error, expected) -> None:
    task: dict = {"id": "cgt-1", "status": status}
    if error is not None:
        task["error"] = error
    transport = _CreateTransport()
    provider = _provider(transport)
    _stub_poll(monkeypatch, provider, task)

    response = await provider.video("p", _MODEL, poll_interval=0)

    assert response.media is None
    assert "seedance video generation failed" in response.message.content
    assert status in response.message.content
    assert expected in response.message.content


@pytest.mark.asyncio
async def test_poll_timeout_fails_before_the_first_read(monkeypatch) -> None:
    transport = _CreateTransport()
    provider = _provider(transport)
    poller = _stub_poll(monkeypatch, provider, _succeeded())

    response = await provider.video("p", _MODEL, poll_interval=0, poll_timeout=0)

    assert poller.calls == []  # the deadline is checked before every read
    assert "timed out after 0s" in response.message.content
    assert response.media is None


@pytest.mark.asyncio
async def test_succeeded_without_a_video_url_fails_friendly(monkeypatch) -> None:
    transport = _CreateTransport()
    provider = _provider(transport)
    _stub_poll(monkeypatch, provider, {"id": "cgt-1", "status": "succeeded", "content": {}})

    response = await provider.video("p", _MODEL, poll_interval=0)

    assert response.media is None
    assert "succeeded without a content.video_url" in response.message.content


@pytest.mark.asyncio
async def test_binary_true_is_rejected_and_fails_friendly(monkeypatch) -> None:
    transport = _CreateTransport(b"JUNK")  # a bytes-out answer cannot be a task
    provider = _provider(transport)

    response = await provider.video("p", _MODEL, binary=True)

    assert response.media is None
    assert "binary=True is not supported" in response.message.content


# ── _fetch_task — the poll read itself ─────────────────────────


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict | None = None, text: str = "") -> None:
        self.status_code = status_code
        self.is_success = 200 <= status_code < 300
        self._payload = payload
        self.text = text

    def json(self) -> dict:
        assert self._payload is not None
        return self._payload


class _FakeHttpx:
    """Stands in for the module-level httpx import inside seedance.py."""

    def __init__(self, response: _FakeResponse) -> None:
        self.response = response
        self.requests: list[dict] = []

    def AsyncClient(self, *, timeout=None):  # noqa: N802 — mirrors httpx's own name
        fake = self

        class _Client:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return None

            async def get(self, url, headers=None):
                fake.requests.append({"url": url, "headers": headers})
                return fake.response

        return _Client()


@pytest.mark.asyncio
async def test_fetch_task_gets_the_task_url_with_bearer_headers(monkeypatch) -> None:
    fake = _FakeHttpx(_FakeResponse(200, payload={"id": "cgt-1", "status": "running"}))
    monkeypatch.setattr(seedance_module, "httpx", fake)
    provider = _provider(_CreateTransport())

    task = await provider._fetch_task("cgt-1", url=_TASK_URL)

    assert task == {"id": "cgt-1", "status": "running"}
    assert fake.requests == [
        {"url": _TASK_URL + "/cgt-1", "headers": {"Authorization": "Bearer k", "Content-Type": "application/json"}},
    ]


@pytest.mark.asyncio
async def test_fetch_task_non_2xx_raises_with_the_body_excerpt(monkeypatch) -> None:
    fake = _FakeHttpx(_FakeResponse(404, text='{"error": "not found"}'))
    monkeypatch.setattr(seedance_module, "httpx", fake)
    provider = _provider(_CreateTransport())

    with pytest.raises(RuntimeError, match=r"HTTP 404 .*not found"):
        await provider._fetch_task("cgt-9", url=_TASK_URL)


# ── Output handover + the capability E2E ───────────────────────


@pytest.mark.asyncio
async def test_output_handler_sees_the_finished_task(monkeypatch) -> None:
    transport = _CreateTransport()
    provider = _provider(transport)
    _stub_poll(monkeypatch, provider, {"id": "cgt-1", "status": "running"}, _succeeded())
    seen: dict = {}

    def handler(payload: dict) -> CLIverResponse:
        seen.update(payload)
        return CLIverResponse(message=CLIverMessage(role="assistant", content="handled"))

    response = await provider.video("p", _MODEL, poll_interval=0, output_handler=handler)

    assert seen["status"] == "succeeded"
    assert seen["content"]["video_url"].startswith("https://tos.example/")
    assert response.message.content == "handled"


@pytest.mark.asyncio
async def test_generate_video_works_out_of_the_box_on_the_seedance_preset(monkeypatch, tmp_path) -> None:
    """Preset → binding → provider → Ark task wire → saved mp4, through the
    real SeedanceProvider class (the SDK builder is the only stub)."""
    from cliver_core import presets
    from cliver_core.provider.providers import openai as openai_module

    transport = _CreateTransport()
    monkeypatch.setattr(openai_module, "resolve_brand_sdk", lambda *args, **kwargs: transport)
    poller = _PollStub(_succeeded())
    monkeypatch.setattr(SeedanceProvider, "_fetch_task", poller)
    video = b"\x00\x00\x00 ftypmp42" + b"\x00" * 40
    monkeypatch.setattr("cliver_core.media._download_url", lambda url: video)

    binding = presets.seedance(api_key="k", poll_interval=0)
    response = await generate_video(prompt="a drone shot", binding=binding, output_dir=str(tmp_path))

    call = transport.posts[-1]
    assert call["path"] == _TASK_URL  # the preset's pinned path resolved to the Ark URL
    assert call["body"]["model"] == _MODEL
    assert poller.calls[0]["url"] == _TASK_URL
    (mc,) = response.media
    assert mc.mime_type == "video/mp4"
    assert (tmp_path / "generated_0.mp4").read_bytes() == video
    await binding.aclose()
