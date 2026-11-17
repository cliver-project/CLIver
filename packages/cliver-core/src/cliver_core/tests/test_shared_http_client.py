"""One shared ``httpx.AsyncClient`` per provider for the raw-HTTP seams.

The SDK transports already pool connections inside their client; the raw
seams (embed / rerank / System One via ``_post_json_endpoint``, the
DashScope uploads and task reads, and the Seedance task reads) must reuse
one provider-owned client too, so repeated calls and every poll iteration
keep the TCP/TLS connection instead of reconnecting.  No network here: the
client is a recording fake.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from cliver_core import ClientOptions
from cliver_core import provider as provider_module
from cliver_core.messages import CLIverMessage
from cliver_core.provider import CLIverRequest, CLIverResponse, Provider
from cliver_core.provider.providers import dashscope as dashscope_module
from cliver_core.provider.providers import seedance as seedance_module
from cliver_core.provider.providers.dashscope import DashScopeProvider
from cliver_core.provider.providers.openai import OpenAIProvider
from cliver_core.provider.providers.seedance import SeedanceProvider

_BASE = "https://dashscope.aliyuncs.com/api/v1"
_IMAGE_PATH = "/services/aigc/text2image/image-synthesis"
_ARK_BASE = "https://ark.cn-beijing.volces.com/api/v3"
_ARK_TASK_URL = _ARK_BASE + "/contents/generations/tasks"


class _FakeResponse:
    def __init__(self, payload: dict | None = None, status_code: int = 200) -> None:
        self._payload = {} if payload is None else payload
        self.status_code = status_code
        self.is_success = 200 <= status_code < 300
        self.text = "" if self.is_success else f"HTTP {status_code}"

    def json(self) -> dict:
        return self._payload


class _RecordingClient:
    """Stands in for ``httpx.AsyncClient``; records every construction and call."""

    constructed: list["_RecordingClient"] = []

    def __init__(self, *args, **kwargs) -> None:
        type(self).constructed.append(self)
        self.init_kwargs = kwargs
        self.posts: list[dict] = []
        self.gets: list[dict] = []
        self.closed = 0
        self.responses: list[_FakeResponse] = []

    def queue(self, payload: dict | None = None, status_code: int = 200) -> None:
        self.responses.append(_FakeResponse(payload, status_code))

    async def post(self, url: str, **kwargs) -> _FakeResponse:
        self.posts.append({"url": url, **kwargs})
        return self.responses.pop(0) if self.responses else _FakeResponse()

    async def get(self, url: str, **kwargs) -> _FakeResponse:
        self.gets.append({"url": url, **kwargs})
        return self.responses.pop(0) if self.responses else _FakeResponse()

    async def aclose(self) -> None:
        self.closed += 1


class _FakeTransport:
    """The minimal ``BrandSDK`` stub — the create POST is never exercised here."""

    def __init__(self) -> None:
        self.closed = 0

    async def post(self, path, body, *, files=None, binary=False, headers=None):
        return {}

    async def close(self) -> None:
        self.closed += 1


class _RawHTTPProvider(Provider):
    """Concrete provider that exercises only the raw-HTTP seams."""

    protocol = "openai"
    brand = "raw-http-test"

    async def chat(self, request: CLIverRequest) -> CLIverResponse:
        return CLIverResponse(message=CLIverMessage(role="assistant", content="ok"))

    async def stream(self, request):  # pragma: no cover — chat test double only
        if False:
            yield None


def _forbidden_client(**kwargs):
    raise AssertionError("the provider must reuse its shared client, not build a new one")


class _FakeLimits:
    """Records the pool kwargs handed to ``httpx.Limits``."""

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


@pytest.fixture(autouse=True)
def _reset_constructions() -> None:
    _RecordingClient.constructed.clear()


@pytest.mark.asyncio
async def test_http_client_is_lazy_shared_and_recreated_after_close(monkeypatch) -> None:
    monkeypatch.setattr(provider_module, "httpx", SimpleNamespace(AsyncClient=_RecordingClient))
    provider = _RawHTTPProvider(api_key="k", base_url="https://example.test/v1")

    first = provider._http_client()
    assert first is provider._http_client()
    assert _RecordingClient.constructed == [first]

    await provider.close()
    assert first.closed == 1

    second = provider._http_client()
    assert second is not first
    assert len(_RecordingClient.constructed) == 2


@pytest.mark.asyncio
async def test_http_client_applies_user_agent_and_pool_limits(monkeypatch) -> None:
    monkeypatch.setattr(
        provider_module,
        "httpx",
        SimpleNamespace(AsyncClient=_RecordingClient, Limits=_FakeLimits),
    )
    provider = _RawHTTPProvider(
        api_key="k",
        base_url="https://example.test/v1",
        client=ClientOptions(
            user_agent="cliver-test/1.0",
            max_connections=8,
            max_keepalive_connections=4,
            keepalive_expiry=2.5,
        ),
    )

    client = provider._http_client()

    assert client.init_kwargs["headers"] == {"User-Agent": "cliver-test/1.0"}
    assert client.init_kwargs["limits"].kwargs == {
        "max_connections": 8,
        "max_keepalive_connections": 4,
        "keepalive_expiry": 2.5,
    }


@pytest.mark.asyncio
async def test_http_client_keeps_httpx_defaults_without_client_options(monkeypatch) -> None:
    monkeypatch.setattr(
        provider_module,
        "httpx",
        SimpleNamespace(AsyncClient=_RecordingClient, Limits=_FakeLimits),
    )
    provider = _RawHTTPProvider(api_key="k", base_url="https://example.test/v1")

    client = provider._http_client()

    assert client.init_kwargs == {}


@pytest.mark.asyncio
async def test_post_json_endpoint_reuses_the_shared_client(monkeypatch) -> None:
    provider = _RawHTTPProvider(api_key="k", base_url="https://example.test/v1")
    client = _RecordingClient()
    monkeypatch.setattr(provider, "_http_client", lambda: client)

    await provider._post_json_endpoint("/rerank", {"query": "q"}, label="rerank", timeout=7.5)
    await provider._post_json_endpoint("/embeddings", {"input": ["a"]}, label="embeddings")

    assert [call["url"] for call in client.posts] == [
        "https://example.test/v1/rerank",
        "https://example.test/v1/embeddings",
    ]
    assert client.posts[0]["timeout"] == 7.5
    assert client.posts[1]["timeout"] == 120.0


@pytest.mark.asyncio
async def test_protocol_provider_close_closes_the_shared_client(monkeypatch) -> None:
    monkeypatch.setattr(provider_module, "httpx", SimpleNamespace(AsyncClient=_RecordingClient))
    transport = _FakeTransport()
    provider = OpenAIProvider(api_key="k", base_url="https://example.test/v1", transport=transport)

    client = provider._http_client()
    await provider.close()

    assert client.closed == 1
    assert transport.closed == 1


@pytest.mark.asyncio
async def test_dashscope_task_poll_reuses_the_shared_client(monkeypatch) -> None:
    provider = DashScopeProvider(api_key="k", base_url=_BASE, transport=_FakeTransport())
    client = _RecordingClient()
    client.queue({"output": {"task_status": "SUCCEEDED"}})
    monkeypatch.setattr(provider, "_http_client", lambda: client)
    monkeypatch.setattr(dashscope_module, "httpx", SimpleNamespace(AsyncClient=_forbidden_client), raising=False)

    payload = await provider._fetch_task("t-1", url=_BASE + _IMAGE_PATH)

    assert payload["output"]["task_status"] == "SUCCEEDED"
    assert [call["url"] for call in client.gets] == [f"{_BASE}/tasks/t-1"]
    assert client.gets[0]["timeout"] == provider._POLL_REQUEST_TIMEOUT


@pytest.mark.asyncio
async def test_dashscope_upload_reuses_the_shared_client(monkeypatch) -> None:
    provider = DashScopeProvider(api_key="k", base_url=_BASE, transport=_FakeTransport())
    client = _RecordingClient()
    client.queue(
        {
            "data": {
                "upload_host": "https://oss.example.test",
                "upload_dir": "dashscope",
                "policy": "policy",
                "signature": "sig",
                "oss_access_key_id": "key-id",
            }
        }
    )
    client.queue({"ok": True})
    monkeypatch.setattr(provider, "_http_client", lambda: client)
    monkeypatch.setattr(dashscope_module, "httpx", SimpleNamespace(AsyncClient=_forbidden_client), raising=False)

    url = await provider.upload_media(b"image", "image/png", model="wanx", filename="a.png")

    assert url == "oss://dashscope/a.png"
    assert client.gets[0]["timeout"] == provider._POLL_REQUEST_TIMEOUT
    assert client.posts[0]["url"] == "https://oss.example.test"
    assert client.posts[0]["timeout"] == provider._UPLOAD_TIMEOUT


@pytest.mark.asyncio
async def test_seedance_task_poll_reuses_the_shared_client(monkeypatch) -> None:
    provider = SeedanceProvider(api_key="k", base_url=_ARK_BASE, transport=_FakeTransport())
    client = _RecordingClient()
    client.queue({"status": "succeeded"})
    monkeypatch.setattr(provider, "_http_client", lambda: client)
    monkeypatch.setattr(seedance_module, "httpx", SimpleNamespace(AsyncClient=_forbidden_client), raising=False)

    payload = await provider._fetch_task("cgt-1", url=_ARK_TASK_URL)

    assert payload["status"] == "succeeded"
    assert [call["url"] for call in client.gets] == [f"{_ARK_TASK_URL}/cgt-1"]
    assert client.gets[0]["timeout"] == provider._POLL_REQUEST_TIMEOUT
