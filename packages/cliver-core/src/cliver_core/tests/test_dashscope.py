"""DashScope (Alibaba Model Studio / Bailian) — the async media-task wire.

The create POST goes through an injected transport stub; the poll read
(``_fetch_task``), the poll cadence (``_sleep``), the upload calls, and the
media download (``cliver_core.media._download_url``) are stubbed — no network.
API keys here are literal dummies; the suite never reads key environment
variables for real values.
"""

from __future__ import annotations

import base64
from pathlib import Path

import pytest

from cliver_core.capabilities import generate_image
from cliver_core.errors import ProviderError
from cliver_core.media import MediaContent, MediaType
from cliver_core.provider.providers import dashscope as dashscope_module
from cliver_core.provider.providers.dashscope import DashScopeProvider

_BASE = "https://dashscope.aliyuncs.com/api/v1"
_IMAGE_PATH = "/services/aigc/text2image/image-synthesis"
_BACKGROUND_PATH = "/services/aigc/background-generation/generation"
_VIDEO_PATH = "/services/aigc/video-generation/video-synthesis"
_IMAGE_URL = _BASE + _IMAGE_PATH
_BACKGROUND_URL = _BASE + _BACKGROUND_PATH
_VIDEO_URL = _BASE + _VIDEO_PATH
_T2I_MODEL = "wanx2.1-t2i-turbo"
_BG_MODEL = "wanx-background-generation-v2"
_VIDEO_MODEL = "wanx2.1-t2v-turbo"
_TASK_ID = "d577cd66-9f3e-4bc0-9f1e-7a1b2c3d4e5f"


class _CreateTransport:
    """Records the create POST; answers one canned task-submission payload."""

    def __init__(self, payload: object = None) -> None:
        self.posts: list[dict] = []
        self.payload = (
            {"request_id": "req-1", "output": {"task_id": _TASK_ID, "task_status": "PENDING"}}
            if payload is None
            else payload
        )

    async def post(self, path: str, body: dict, *, files=None, binary: bool = False, headers=None):
        self.posts.append({"path": path, "body": body, "binary": binary, "headers": headers})
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


def _provider(transport: _CreateTransport, base_url: str = _BASE) -> DashScopeProvider:
    return DashScopeProvider(api_key="k", base_url=base_url, transport=transport)


def _stub_poll(monkeypatch, provider: DashScopeProvider, *payloads: dict) -> _PollStub:
    poller = _PollStub(*payloads)
    monkeypatch.setattr(provider, "_fetch_task", poller)
    return poller


def _task(status: str, **output) -> dict:
    return {"request_id": "req-1", "output": {"task_id": _TASK_ID, "task_status": status, **output}}


def _image_succeeded(url: str = "https://oss.example/out.png") -> dict:
    return _task("SUCCEEDED", results=[{"url": url}])


def _video_succeeded(url: str = "https://oss.example/out.mp4") -> dict:
    return _task("SUCCEEDED", video_url=url)


def _http_image(url: str = "https://cdn.example/product.jpg") -> MediaContent:
    return MediaContent(type=MediaType.IMAGE, data=url, mime_type="image/jpeg")


# ── The DashScope surface — descriptors ────────────────────────


def test_dashscope_locks_the_brand_surface() -> None:
    provider = _provider(_CreateTransport())
    assert provider.provider_name() == "dashscope"
    assert provider.base_url == _BASE
    assert DashScopeProvider.default_base_url == _BASE
    assert provider._resolve_media_gen_url("image") == _IMAGE_URL
    assert provider._resolve_media_gen_url("video") == _VIDEO_URL
    # No OpenAI-fashion media endpoints on this host; tts/asr fail fast...
    assert provider.media_path("tts") is None
    assert provider.media_path("asr") is None
    # ...while the media-scoped opt-out leaves embed/rerank alone.
    assert provider.media_path("embed") is not None


def test_dashscope_registry_and_preset_resolution() -> None:
    from cliver_core import presets
    from cliver_core.provider.providers import create_provider, resolve_provider_class

    assert resolve_provider_class(provider_name="dashscope", protocol="openai") is DashScopeProvider
    provider = create_provider(api_key="k", provider_name="dashscope")
    assert isinstance(provider, DashScopeProvider)
    assert provider.base_url == _BASE

    image_binding = presets.dashscope_image(api_key="k")
    assert image_binding.provider_name == "dashscope"
    assert image_binding.protocol == "openai"
    assert image_binding.call_defaults()["path"] == _IMAGE_PATH

    background = presets.dashscope_background(api_key="k")
    assert background.call_defaults()["path"] == _BACKGROUND_PATH
    assert background.params == {"model_version": "v3", "n": 1}

    video = presets.dashscope_video(api_key="k")
    assert video.call_defaults()["path"] == _VIDEO_PATH
    assert video.params == {"duration": 5, "size": "1280*720"}

    # path=None defers to the brand map
    assert "path" not in presets.dashscope_image(api_key="k", path=None).call_defaults()


@pytest.mark.asyncio
async def test_dashscope_preset_resolves_to_the_same_url_as_the_brand_map() -> None:
    from cliver_core import presets

    cases = [
        (presets.dashscope_image, "image", _IMAGE_PATH),
        (presets.dashscope_background, "image", _BACKGROUND_PATH),
        (presets.dashscope_video, "video", _VIDEO_PATH),
    ]
    for factory, purpose, path in cases:
        binding = factory(api_key="k")
        provider = binding.get_provider()
        assert isinstance(provider, DashScopeProvider)
        assert provider.base_url == _BASE
        assert provider._resolve_media_gen_url(purpose, binding.call_defaults()["path"]) == _BASE + path
        await binding.aclose()


# ── The create body — DashScope's input/parameters shape ───────


@pytest.mark.asyncio
async def test_t2i_body_wraps_prompt_in_input_and_wraps_params(monkeypatch) -> None:
    transport = _CreateTransport()
    provider = _provider(transport)
    _stub_poll(monkeypatch, provider, _image_succeeded())

    await provider.image("a cat on a mat", _T2I_MODEL, size="1024x1024", n=1, poll_interval=0)

    call = transport.posts[-1]
    assert call["path"] == _IMAGE_URL
    assert call["binary"] is False
    assert call["body"] == {
        "model": _T2I_MODEL,
        "input": {"prompt": "a cat on a mat"},
        "parameters": {"size": "1024*1024", "n": 1},
    }


@pytest.mark.asyncio
async def test_negative_prompt_rides_the_input_not_the_parameters(monkeypatch) -> None:
    transport = _CreateTransport()
    provider = _provider(transport)
    _stub_poll(monkeypatch, provider, _image_succeeded())

    await provider.image("p", _T2I_MODEL, negative_prompt="blurry", poll_interval=0)

    body = transport.posts[-1]["body"]
    assert body["input"] == {"prompt": "p", "negative_prompt": "blurry"}
    assert body["parameters"] == {}


@pytest.mark.asyncio
async def test_background_body_uses_the_first_image_and_the_prompt_as_ref_prompt(monkeypatch) -> None:
    transport = _CreateTransport()
    provider = _provider(transport)
    _stub_poll(monkeypatch, provider, _image_succeeded())

    await provider.image(
        "clean seamless studio backdrop",
        _BG_MODEL,
        media=[_http_image()],
        path=_BACKGROUND_PATH,
        model_version="v3",
        n=1,
        poll_interval=0,
    )

    call = transport.posts[-1]
    assert call["path"] == _BACKGROUND_URL
    assert call["body"] == {
        "model": _BG_MODEL,
        "input": {
            "ref_prompt": "clean seamless studio backdrop",
            "base_image_url": "https://cdn.example/product.jpg",
        },
        "parameters": {"model_version": "v3", "n": 1},
    }


@pytest.mark.asyncio
async def test_background_body_maps_a_second_image_to_ref_image_url(monkeypatch) -> None:
    transport = _CreateTransport()
    provider = _provider(transport)
    _stub_poll(monkeypatch, provider, _image_succeeded())

    await provider.image(
        "poster style",
        _BG_MODEL,
        media=[_http_image(), _http_image("https://cdn.example/style.png")],
        path=_BACKGROUND_PATH,
        poll_interval=0,
    )

    body = transport.posts[-1]["body"]
    assert body["input"]["base_image_url"] == "https://cdn.example/product.jpg"
    assert body["input"]["ref_image_url"] == "https://cdn.example/style.png"


@pytest.mark.asyncio
async def test_video_body_carries_prompt_img_url_and_duration(monkeypatch) -> None:
    transport = _CreateTransport()
    provider = _provider(transport)
    _stub_poll(monkeypatch, provider, _video_succeeded())

    await provider.video(
        "a slow orbit",
        _VIDEO_MODEL,
        media=[_http_image()],
        duration=5,
        size="1280x720",
        poll_interval=0,
    )

    call = transport.posts[-1]
    assert call["path"] == _VIDEO_URL
    assert call["body"] == {
        "model": _VIDEO_MODEL,
        "input": {"prompt": "a slow orbit", "img_url": "https://cdn.example/product.jpg"},
        "parameters": {"duration": 5, "size": "1280*720"},
    }


@pytest.mark.asyncio
async def test_poll_knobs_never_reach_the_create_body(monkeypatch) -> None:
    transport = _CreateTransport()
    provider = _provider(transport)
    _stub_poll(monkeypatch, provider, _image_succeeded())

    await provider.image("p", _T2I_MODEL, poll_interval=0, poll_timeout=30)

    body = transport.posts[-1]["body"]
    assert "poll_interval" not in body["parameters"]
    assert "poll_timeout" not in body["parameters"]


# ── The poll loop — create → task → result ─────────────────────


@pytest.mark.asyncio
async def test_poll_happy_path_pending_running_succeeded_returns_the_result_url(monkeypatch, tmp_path) -> None:
    transport = _CreateTransport()  # the create answer is PENDING
    provider = _provider(transport)
    poller = _stub_poll(monkeypatch, provider, _task("RUNNING"), _image_succeeded())
    image = b"\x89PNG\r\n\x1a\n" + b"\x00" * 40
    monkeypatch.setattr("cliver_core.media._download_url", lambda url: image)

    response = await provider.image("a cat", _T2I_MODEL, poll_interval=0, output_dir=str(tmp_path))

    assert [call["task_id"] for call in poller.calls] == [_TASK_ID, _TASK_ID]
    assert all(call["url"] == _IMAGE_URL for call in poller.calls)
    (mc,) = response.media
    assert mc.type == MediaType.IMAGE
    assert mc.data == "https://oss.example/out.png"
    saved = Path(mc.saved_path)
    assert saved.name == "generated_0.png"
    assert saved.read_bytes() == image
    assert response.message.content.startswith("Generated 1 image(s) via dashscope.")


@pytest.mark.asyncio
async def test_poll_interval_is_passed_to_sleep(monkeypatch) -> None:
    transport = _CreateTransport()
    provider = _provider(transport)
    _stub_poll(monkeypatch, provider, _image_succeeded())
    intervals: list[float] = []

    async def _record(seconds: float) -> None:
        intervals.append(seconds)

    monkeypatch.setattr(dashscope_module, "_sleep", _record)
    await provider.image("p", _T2I_MODEL, poll_interval=2.5)

    assert intervals == [2.5]


@pytest.mark.asyncio
async def test_video_task_returns_output_video_url(monkeypatch) -> None:
    transport = _CreateTransport()
    provider = _provider(transport)
    _stub_poll(monkeypatch, provider, _video_succeeded())

    response = await provider.video("p", _VIDEO_MODEL, poll_interval=0)

    (mc,) = response.media
    assert mc.type == MediaType.VIDEO
    assert mc.data == "https://oss.example/out.mp4"


@pytest.mark.asyncio
async def test_multiple_image_results_become_multiple_media_items(monkeypatch) -> None:
    transport = _CreateTransport()
    provider = _provider(transport)
    _stub_poll(
        monkeypatch,
        provider,
        _task("SUCCEEDED", results=[{"url": "https://oss.example/1.png"}, {"url": "https://oss.example/2.png"}]),
    )

    response = await provider.image("p", _T2I_MODEL, n=2, poll_interval=0)

    assert [mc.data for mc in response.media] == ["https://oss.example/1.png", "https://oss.example/2.png"]
    assert response.message.content.startswith("Generated 2 image(s) via dashscope.")


# ── Failures — all land in the friendly response ───────────────


@pytest.mark.asyncio
async def test_failed_task_fails_with_the_vendor_message(monkeypatch) -> None:
    transport = _CreateTransport()
    provider = _provider(transport)
    _stub_poll(
        monkeypatch,
        provider,
        _task("FAILED", code="DataInspectionFailed", message="content violates policy"),
    )

    response = await provider.image("p", _T2I_MODEL, poll_interval=0)

    assert response.media is None
    assert "dashscope image generation failed" in response.message.content
    assert "FAILED" in response.message.content
    assert "content violates policy" in response.message.content


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["CANCELED", "UNKNOWN"])
async def test_terminal_task_states_fail_friendly(monkeypatch, status) -> None:
    transport = _CreateTransport()
    provider = _provider(transport)
    _stub_poll(monkeypatch, provider, _task(status))

    response = await provider.image("p", _T2I_MODEL, poll_interval=0)

    assert response.media is None
    assert status in response.message.content


@pytest.mark.asyncio
async def test_succeeded_with_empty_results_fails_friendly(monkeypatch) -> None:
    transport = _CreateTransport()
    provider = _provider(transport)
    _stub_poll(monkeypatch, provider, _task("SUCCEEDED", results=[]))

    response = await provider.image("p", _T2I_MODEL, poll_interval=0)

    assert response.media is None
    assert "succeeded without a result url" in response.message.content


@pytest.mark.asyncio
async def test_succeeded_with_result_errors_fails_with_the_vendor_message(monkeypatch) -> None:
    transport = _CreateTransport()
    provider = _provider(transport)
    _stub_poll(
        monkeypatch,
        provider,
        _task("SUCCEEDED", results=[{"code": "OutputImageSensitiveContentDetected", "message": "blocked"}]),
    )

    response = await provider.image("p", _T2I_MODEL, poll_interval=0)

    assert response.media is None
    assert "OutputImageSensitiveContentDetected" in response.message.content
    assert "blocked" in response.message.content


@pytest.mark.asyncio
async def test_succeeded_video_without_a_url_fails_friendly(monkeypatch) -> None:
    transport = _CreateTransport()
    provider = _provider(transport)
    _stub_poll(monkeypatch, provider, _task("SUCCEEDED"))

    response = await provider.video("p", _VIDEO_MODEL, poll_interval=0)

    assert response.media is None
    assert "succeeded without an output.video_url" in response.message.content


@pytest.mark.asyncio
async def test_poll_timeout_zero_fails_before_the_first_read(monkeypatch) -> None:
    transport = _CreateTransport()
    provider = _provider(transport)
    poller = _stub_poll(monkeypatch, provider, _image_succeeded())

    response = await provider.image("p", _T2I_MODEL, poll_interval=0, poll_timeout=0)

    assert poller.calls == []  # the deadline is checked before every read
    assert "timed out after 0s" in response.message.content
    assert response.media is None


@pytest.mark.asyncio
async def test_http_200_error_envelope_is_a_failure_response(monkeypatch) -> None:
    transport = _CreateTransport({"code": "InvalidParameter", "message": "bad model", "request_id": "req-1"})
    provider = _provider(transport)
    poller = _stub_poll(monkeypatch, provider, _image_succeeded())

    response = await provider.image("p", "no-such-model", poll_interval=0)

    assert response.media is None
    assert "bad model" in response.message.content
    assert poller.calls == []  # a phantom task is never polled


@pytest.mark.asyncio
async def test_binary_true_is_rejected_and_fails_friendly(monkeypatch) -> None:
    transport = _CreateTransport(b"JUNK")  # a bytes-out answer cannot be a task
    provider = _provider(transport)

    response = await provider.image("p", _T2I_MODEL, binary=True)

    assert response.media is None
    assert "binary=True is not supported" in response.message.content


# ── Headers — the async flag and the OSS resolver ──────────────


@pytest.mark.asyncio
async def test_create_sends_the_async_header(monkeypatch) -> None:
    transport = _CreateTransport()
    provider = _provider(transport)
    _stub_poll(monkeypatch, provider, _image_succeeded())

    await provider.image("p", _T2I_MODEL, poll_interval=0)

    assert transport.posts[-1]["headers"] == {"X-DashScope-Async": "enable"}


@pytest.mark.asyncio
async def test_oss_resolve_header_is_sent_iff_the_body_has_an_oss_url(monkeypatch) -> None:
    transport = _CreateTransport()
    provider = _provider(transport)
    _stub_poll(monkeypatch, provider, _image_succeeded())

    await provider.image(
        "p",
        _BG_MODEL,
        media=[MediaContent(type=MediaType.IMAGE, data="oss://bucket/dir/p.jpg", mime_type="image/jpeg")],
        path=_BACKGROUND_PATH,
        poll_interval=0,
    )

    assert transport.posts[-1]["headers"] == {
        "X-DashScope-Async": "enable",
        "X-DashScope-OssResourceResolve": "enable",
    }


# ── Local reference images — upload to oss:// ─────────────────


@pytest.mark.asyncio
async def test_data_uri_media_is_uploaded_once_and_the_body_carries_the_oss_url(monkeypatch) -> None:
    transport = _CreateTransport()
    provider = _provider(transport)
    _stub_poll(monkeypatch, provider, _image_succeeded())
    uploads: list[dict] = []

    async def fake_upload(data: bytes, mime: str, *, model: str, filename: str) -> str:
        uploads.append({"data": data, "mime": mime, "model": model, "filename": filename})
        return f"oss://bucket/dir/{filename}"

    monkeypatch.setattr(provider, "upload_media", fake_upload)
    raw = b"\x89PNG\r\n\x1a\nfake-ref-image"

    source = MediaContent(
        type=MediaType.IMAGE,
        data="data:image/png;base64," + base64.b64encode(raw).decode(),
        mime_type="image/png",
    )
    await provider.image(
        "make the backdrop clean",
        _BG_MODEL,
        media=[source],
        path=_BACKGROUND_PATH,
        poll_interval=0,
    )

    assert uploads == [
        {"data": raw, "mime": "image/png", "model": _BG_MODEL, "filename": "reference_0.png"},
    ]
    body = transport.posts[-1]["body"]
    assert body["input"]["base_image_url"] == "oss://bucket/dir/reference_0.png"
    assert "data:" not in str(body)
    assert transport.posts[-1]["headers"]["X-DashScope-OssResourceResolve"] == "enable"


@pytest.mark.asyncio
async def test_raw_base64_media_is_wrapped_then_uploaded(monkeypatch) -> None:
    transport = _CreateTransport()
    provider = _provider(transport)
    _stub_poll(monkeypatch, provider, _image_succeeded())
    uploads: list[dict] = []

    async def fake_upload(data: bytes, mime: str, *, model: str, filename: str) -> str:
        uploads.append({"data": data, "mime": mime, "filename": filename})
        return f"oss://bucket/dir/{filename}"

    monkeypatch.setattr(provider, "upload_media", fake_upload)
    raw = b"jpeg-bytes"
    source = MediaContent(
        type=MediaType.IMAGE,
        data=base64.b64encode(raw).decode(),
        mime_type="image/jpeg",
    )

    await provider.image("p", _BG_MODEL, media=[source], path=_BACKGROUND_PATH, poll_interval=0)

    assert uploads == [{"data": raw, "mime": "image/jpeg", "filename": "reference_0.jpg"}]
    assert transport.posts[-1]["body"]["input"]["base_image_url"] == "oss://bucket/dir/reference_0.jpg"


@pytest.mark.asyncio
async def test_public_http_media_passes_through_without_an_upload(monkeypatch) -> None:
    transport = _CreateTransport()
    provider = _provider(transport)
    _stub_poll(monkeypatch, provider, _image_succeeded())
    uploads: list[dict] = []

    async def fake_upload(data: bytes, mime: str, *, model: str, filename: str) -> str:  # pragma: no cover
        uploads.append({"data": data})
        return "oss://unused"

    monkeypatch.setattr(provider, "upload_media", fake_upload)

    await provider.image("p", _BG_MODEL, media=[_http_image()], path=_BACKGROUND_PATH, poll_interval=0)

    assert uploads == []
    assert transport.posts[-1]["body"]["input"]["base_image_url"] == "https://cdn.example/product.jpg"
    assert "X-DashScope-OssResourceResolve" not in transport.posts[-1]["headers"]


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict | None = None, text: str = "") -> None:
        self.status_code = status_code
        self.is_success = 200 <= status_code < 300
        self._payload = payload
        self.text = text

    def json(self) -> dict:
        assert self._payload is not None
        return self._payload


class _UploadHttpx:
    """Stands in for the module-level httpx import — records the policy GET + OSS POST."""

    def __init__(self, policy: dict, upload_status: int = 200) -> None:
        self.policy = policy
        self.upload_status = upload_status
        self.gets: list[dict] = []
        self.posts: list[dict] = []

    def AsyncClient(self, *, timeout=None):  # noqa: N802 — mirrors httpx's own name
        fake = self

        class _Client:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return None

            async def get(self, url, params=None, headers=None):
                fake.gets.append({"url": url, "params": params, "headers": headers})
                return _FakeResponse(200, payload=fake.policy)

            async def post(self, url, data=None, files=None):
                fake.posts.append({"url": url, "data": data, "files": files})
                return _FakeResponse(fake.upload_status)

        return _Client()


def _policy() -> dict:
    return {
        "request_id": "req-1",
        "data": {
            "policy": "pol",
            "signature": "sig",
            "upload_dir": "dashscope-instant/xxx/2026-10-29/yyy",
            "upload_host": "https://dashscope-file-xxx.oss-cn-beijing.aliyuncs.com",
            "expire_in_seconds": 300,
            "max_file_size_mb": 100,
            "oss_access_key_id": "LTA-ak",
            "x_oss_object_acl": "private",
            "x_oss_forbid_overwrite": "true",
        },
    }


@pytest.mark.asyncio
async def test_upload_media_gets_a_policy_then_posts_the_multipart(monkeypatch) -> None:
    fake = _UploadHttpx(_policy())
    monkeypatch.setattr(dashscope_module, "httpx", fake)
    provider = _provider(_CreateTransport())

    url = await provider.upload_media(b"PNGDATA", "image/png", model=_BG_MODEL, filename="cat.png")

    assert url == "oss://dashscope-instant/xxx/2026-10-29/yyy/cat.png"
    assert fake.gets == [
        {
            "url": _BASE + "/uploads",
            "params": {"action": "getPolicy", "model": _BG_MODEL},
            "headers": {"Authorization": "Bearer k", "Content-Type": "application/json"},
        }
    ]
    assert fake.posts == [
        {
            "url": "https://dashscope-file-xxx.oss-cn-beijing.aliyuncs.com",
            "data": {
                "OSSAccessKeyId": "LTA-ak",
                "policy": "pol",
                "Signature": "sig",
                "key": "dashscope-instant/xxx/2026-10-29/yyy/cat.png",
                "x-oss-object-acl": "private",
                "x-oss-forbid-overwrite": "true",
                "success_action_status": "200",
            },
            "files": {"file": ("cat.png", b"PNGDATA", "image/png")},
        }
    ]


@pytest.mark.asyncio
async def test_upload_media_failure_raises_a_provider_error(monkeypatch) -> None:
    fake = _UploadHttpx(_policy(), upload_status=403)
    monkeypatch.setattr(dashscope_module, "httpx", fake)
    provider = _provider(_CreateTransport())

    with pytest.raises(ProviderError, match=r"upload failed: HTTP 403"):
        await provider.upload_media(b"PNGDATA", "image/png", model=_BG_MODEL, filename="cat.png")


@pytest.mark.asyncio
async def test_upload_policy_without_data_raises_a_provider_error(monkeypatch) -> None:
    fake = _UploadHttpx({"request_id": "req-1"})
    monkeypatch.setattr(dashscope_module, "httpx", fake)
    provider = _provider(_CreateTransport())

    with pytest.raises(ProviderError, match="no data"):
        await provider.upload_media(b"PNGDATA", "image/png", model=_BG_MODEL, filename="cat.png")


# ── _fetch_task — the task read itself ─────────────────────────


class _TaskHttpx:
    """Stands in for the module-level httpx import inside dashscope.py."""

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
async def test_fetch_task_reads_the_tasks_route_not_the_create_route(monkeypatch) -> None:
    fake = _TaskHttpx(_FakeResponse(200, payload=_task("RUNNING")))
    monkeypatch.setattr(dashscope_module, "httpx", fake)
    provider = _provider(_CreateTransport())

    task = await provider._fetch_task(_TASK_ID, url=_IMAGE_URL)

    assert task == _task("RUNNING")
    assert fake.requests == [
        {
            "url": _BASE + "/tasks/" + _TASK_ID,
            "headers": {"Authorization": "Bearer k", "Content-Type": "application/json"},
        }
    ]


@pytest.mark.asyncio
async def test_fetch_task_derives_the_tasks_root_from_a_proxied_create_url(monkeypatch) -> None:
    fake = _TaskHttpx(_FakeResponse(200, payload=_task("RUNNING")))
    monkeypatch.setattr(dashscope_module, "httpx", fake)
    provider = _provider(_CreateTransport())

    await provider._fetch_task(_TASK_ID, url="https://proxy.internal/api/v1" + _IMAGE_PATH)

    assert fake.requests[0]["url"] == "https://proxy.internal/api/v1/tasks/" + _TASK_ID


@pytest.mark.asyncio
async def test_fetch_task_non_2xx_raises_with_the_body_excerpt(monkeypatch) -> None:
    fake = _TaskHttpx(_FakeResponse(404, text='{"code": "NotFound"}'))
    monkeypatch.setattr(dashscope_module, "httpx", fake)
    provider = _provider(_CreateTransport())

    with pytest.raises(RuntimeError, match=r"HTTP 404 .*NotFound"):
        await provider._fetch_task("missing", url=_IMAGE_URL)


# ── The capability E2E — preset → binding → provider → task wire ──


@pytest.mark.asyncio
async def test_generate_image_works_out_of_the_box_on_the_background_preset(monkeypatch, tmp_path) -> None:
    """Preset → binding → provider → DashScope task wire → saved png, through
    the real DashScopeProvider class (the SDK builder is the only stub)."""
    from cliver_core import presets
    from cliver_core.provider.providers import openai as openai_module

    transport = _CreateTransport()
    monkeypatch.setattr(openai_module, "resolve_brand_sdk", lambda *args, **kwargs: transport)
    poller = _PollStub(_image_succeeded())
    monkeypatch.setattr(DashScopeProvider, "_fetch_task", poller)
    image = b"\x89PNG\r\n\x1a\n" + b"\x00" * 40
    monkeypatch.setattr("cliver_core.media._download_url", lambda url: image)

    binding = presets.dashscope_background(api_key="k", poll_interval=0)
    response = await generate_image(
        prompt="clean seamless studio backdrop, soft daylight",
        binding=binding,
        media=[_http_image()],
        output_dir=str(tmp_path),
    )

    call = transport.posts[-1]
    assert call["path"] == _BACKGROUND_URL  # the preset's pinned path resolved to the DashScope URL
    assert call["body"] == {
        "model": _BG_MODEL,
        "input": {
            "ref_prompt": "clean seamless studio backdrop, soft daylight",
            "base_image_url": "https://cdn.example/product.jpg",
        },
        "parameters": {"model_version": "v3", "n": 1},
    }
    assert call["headers"] == {"X-DashScope-Async": "enable"}
    assert poller.calls[0]["url"] == _BACKGROUND_URL
    (mc,) = response.media
    assert mc.mime_type == "image/png"
    assert (tmp_path / "generated_0.png").read_bytes() == image
    await binding.aclose()


# ── D4 — the qwen chat brand ───────────────────────────────────


def test_qwen_is_an_openai_compatible_chat_brand() -> None:
    from cliver_core.provider.providers import resolve_provider_class
    from cliver_core.provider.providers.qwen import QwenProvider

    assert resolve_provider_class(provider_name="qwen", protocol="openai") is QwenProvider
    assert QwenProvider.default_base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert QwenProvider.provider_name() == "qwen"


@pytest.mark.asyncio
async def test_qwen_preset_builds_the_brand_provider() -> None:
    from cliver_core import presets
    from cliver_core.provider.providers.qwen import QwenProvider

    binding = presets.qwen(api_key="k")
    assert binding.model == "qwen-plus"
    assert binding.purpose == "llm"
    assert binding.provider_name == "qwen"
    provider = binding.get_provider()
    assert isinstance(provider, QwenProvider)
    assert provider.base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    await binding.aclose()


# ── Qwen-Image 3.0 — the synchronous multimodal wire ──────────


_QWEN_IMAGE_MODEL = "qwen-image-3.0"
_MULTIMODAL_PATH = "/services/aigc/multimodal-generation/generation"
_MULTIMODAL_URL = _BASE + _MULTIMODAL_PATH


def _qwen_image_succeeded(url: str = "https://oss.example/qwen.png") -> dict:
    return {
        "output": {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": [{"image": url, "type": "image"}],
                    },
                }
            ]
        },
        "usage": {"input_image_count": 1, "output_image_count": 1},
        "request_id": "req-2",
    }


def test_qwen_image_locks_the_synchronous_surface() -> None:
    from cliver_core.provider.providers.qwen_image import QwenImageProvider

    provider = QwenImageProvider(api_key="k", base_url=_BASE, transport=_CreateTransport())
    assert provider.provider_name() == "qwen-image"
    assert provider._resolve_media_gen_url("image") == _MULTIMODAL_URL
    # The sync multimodal endpoint serves images only.
    assert provider.media_path("video") is None
    assert provider.media_path("tts") is None


@pytest.mark.asyncio
async def test_qwen_image_body_is_one_multimodal_user_message() -> None:
    from cliver_core.provider.providers.qwen_image import QwenImageProvider

    transport = _CreateTransport(_qwen_image_succeeded())
    provider = QwenImageProvider(api_key="k", base_url=_BASE, transport=transport)
    raw = b"\x89PNG\r\n\x1a\nreference"
    b64 = base64.b64encode(raw).decode()

    response = await provider.image(
        "place the product on a light-gray backdrop and add the headline ACME",
        _QWEN_IMAGE_MODEL,
        media=[
            MediaContent(type=MediaType.IMAGE, data=b64, mime_type="image/png"),
            _http_image("https://cdn.example/style.jpg"),
        ],
        size="1024x1024",
        n=1,
    )

    call = transport.posts[-1]
    assert call["path"] == _MULTIMODAL_URL
    # Data URIs are accepted directly by this wire — no upload, no async flag.
    assert call["headers"] is None
    assert call["body"] == {
        "model": _QWEN_IMAGE_MODEL,
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"image": f"data:image/png;base64,{b64}"},
                        {"image": "https://cdn.example/style.jpg"},
                        {"text": "place the product on a light-gray backdrop and add the headline ACME"},
                    ],
                }
            ]
        },
        "parameters": {"size": "1024*1024", "n": 1},
    }
    (mc,) = response.media
    assert mc.data == "https://oss.example/qwen.png"


@pytest.mark.asyncio
async def test_qwen_image_sends_the_oss_resolver_header_only_for_oss_urls() -> None:
    from cliver_core.provider.providers.qwen_image import QwenImageProvider

    transport = _CreateTransport(_qwen_image_succeeded())
    provider = QwenImageProvider(api_key="k", base_url=_BASE, transport=transport)

    await provider.image("p", _QWEN_IMAGE_MODEL, media=[_http_image("oss://bucket/dir/cat.png")])
    assert transport.posts[-1]["headers"] == {"X-DashScope-OssResourceResolve": "enable"}

    await provider.image("p", _QWEN_IMAGE_MODEL, media=[_http_image("https://cdn.example/cat.png")])
    assert transport.posts[-1]["headers"] is None


@pytest.mark.asyncio
async def test_qwen_image_sync_response_downloads_without_polling(monkeypatch, tmp_path) -> None:
    from cliver_core.provider.providers.qwen_image import QwenImageProvider

    transport = _CreateTransport(_qwen_image_succeeded())
    provider = QwenImageProvider(api_key="k", base_url=_BASE, transport=transport)
    poller = _stub_poll(monkeypatch, provider, _image_succeeded())
    image = b"\x89PNG\r\n\x1a\n" + b"\x00" * 40
    monkeypatch.setattr("cliver_core.media._download_url", lambda url: image)

    response = await provider.image("p", _QWEN_IMAGE_MODEL, output_dir=str(tmp_path))

    assert poller.calls == []  # synchronous — no task to poll
    (mc,) = response.media
    assert Path(mc.saved_path).read_bytes() == image
    assert Path(mc.saved_path).name == "generated_0.png"


@pytest.mark.asyncio
async def test_qwen_image_preset_wiring() -> None:
    from cliver_core import presets
    from cliver_core.provider.providers.qwen_image import QwenImageProvider

    binding = presets.qwen_image(api_key="k")
    assert binding.model == _QWEN_IMAGE_MODEL
    assert binding.purpose == "image"
    assert binding.provider_name == "qwen-image"
    assert binding.protocol == "openai"
    assert binding.call_defaults()["path"] == _MULTIMODAL_PATH
    provider = binding.get_provider()
    assert isinstance(provider, QwenImageProvider)
    assert provider.base_url == _BASE
    assert provider._resolve_media_gen_url("image", binding.call_defaults()["path"]) == _MULTIMODAL_URL
    await binding.aclose()


@pytest.mark.asyncio
async def test_generate_image_with_references_works_on_the_qwen_image_preset(monkeypatch, tmp_path) -> None:
    from cliver_core import presets
    from cliver_core.provider.providers import openai as openai_module
    from cliver_core.provider.providers.qwen_image import QwenImageProvider

    transport = _CreateTransport(_qwen_image_succeeded())
    monkeypatch.setattr(openai_module, "resolve_brand_sdk", lambda *args, **kwargs: transport)
    image = b"\x89PNG\r\n\x1a\n" + b"\x00" * 40
    monkeypatch.setattr("cliver_core.media._download_url", lambda url: image)

    binding = presets.qwen_image(api_key="k")
    response = await generate_image(
        prompt="keep the product, white studio background, headline SALE",
        binding=binding,
        media=[_http_image()],
        size="1024x1024",
        output_dir=str(tmp_path),
    )

    call = transport.posts[-1]
    assert call["path"] == _MULTIMODAL_URL
    assert call["body"]["input"]["messages"][0]["content"] == [
        {"image": "https://cdn.example/product.jpg"},
        {"text": "keep the product, white studio background, headline SALE"},
    ]
    assert call["body"]["parameters"] == {"size": "1024*1024"}
    (mc,) = response.media
    assert Path(mc.saved_path) == tmp_path / "generated_0.png"
    assert Path(mc.saved_path).read_bytes() == image
    assert isinstance(binding.get_provider(), QwenImageProvider)
    await binding.aclose()
