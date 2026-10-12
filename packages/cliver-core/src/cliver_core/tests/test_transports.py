"""Transport resolution, I/O pipes, and provider protocol shape (no network).

Covers:
- ``cliver.provider_sdks`` entry-point discovery (third-party brand SDKs)
- resolve_brand_sdk preference order (third-party adapter → builtin
  protocol SDK → ValueError on unknown protocol, empty base URL, empty key)
- transports as pure pipes: prepared body in → parsed wire payload out
  (both SDK transports via httpx.MockTransport), including the
  typed-kwargs/extra_body split landing verbatim on the wire, the reserved
  terminal `__cliver_final__` event (SDK-aggregated message + usage), and
  JSON-schema options (`response_format` / `output_config`) arriving untouched
- protocol providers owning the shape over a fake pipe: request assembly,
  response conversion, tool-name restore, stream-event filtering, terminal
  chunk finalize, and fail-fast on a missing terminal event
"""

from __future__ import annotations

import asyncio
import base64
import json

import pytest

from cliver_core.messages import CLIverMessage
from cliver_core.provider import CLIverRequest
from cliver_core.provider.providers.anthropic import AnthropicProvider
from cliver_core.provider.providers.openai import OpenAIProvider
from cliver_core.provider.transports import (
    registered_adapters,
    resolve_brand_sdk,
)
from cliver_core.tool import CLIverTool

# ── Resolution ─────────────────────────────────────────────────


def test_resolve_default_brand_uses_sdk() -> None:
    transport = resolve_brand_sdk("openai", api_key="k", base_url="https://example.test/v1")
    assert transport.label == "openai-sdk"
    asyncio.run(transport.close())


def test_resolve_unknown_brand_uses_protocol_sdk() -> None:
    # No third-party adapter for the brand → the protocol's builtin SDK
    # serves it through its base_url.
    transport = resolve_brand_sdk("openai", api_key="k", base_url="https://example.test/v1", brand="deepseek")
    assert transport.label == "openai-sdk"
    asyncio.run(transport.close())


def test_resolve_third_party_adapter_wins(monkeypatch) -> None:
    class _ZhipuAdapter:
        supported_protocols = ["openai"]
        label = "zhipu-sdk"

        @classmethod
        def create(cls, **kwargs):
            return cls()

    import cliver_core.provider.transports as transports_pkg

    monkeypatch.setattr(transports_pkg, "registered_adapters", lambda: {"zhipu": _ZhipuAdapter})
    transport = resolve_brand_sdk("openai", api_key="k", base_url="https://example.test/v1", brand="zhipu")
    assert transport.label == "zhipu-sdk"


def test_resolve_broken_adapter_falls_back_to_builtin(monkeypatch) -> None:
    class _BrokenAdapter:
        supported_protocols = ["openai"]

        @classmethod
        def create(cls, **kwargs):
            raise ImportError("brand SDK not installed")

    import cliver_core.provider.transports as transports_pkg

    monkeypatch.setattr(transports_pkg, "registered_adapters", lambda: {"openai": _BrokenAdapter})
    transport = resolve_brand_sdk("openai", api_key="k", base_url="https://example.test/v1")
    assert transport.label == "openai-sdk"
    asyncio.run(transport.close())


def test_resolve_adapter_protocol_mismatch_falls_back_to_builtin(monkeypatch) -> None:
    class _AnthropicOnlyAdapter:
        supported_protocols = ["anthropic"]

        @classmethod
        def create(cls, **kwargs):
            raise AssertionError("must not be used")

    import cliver_core.provider.transports as transports_pkg

    monkeypatch.setattr(transports_pkg, "registered_adapters", lambda: {"glm": _AnthropicOnlyAdapter})
    transport = resolve_brand_sdk("openai", api_key="k", base_url="https://example.test/v1", brand="glm")
    assert transport.label == "openai-sdk"
    asyncio.run(transport.close())


def test_resolve_unknown_protocol_raises() -> None:
    with pytest.raises(ValueError, match="Unknown protocol 'mqtt'"):
        resolve_brand_sdk("mqtt", api_key="k", base_url="https://example.test")


def test_resolve_empty_base_url_raises() -> None:
    # create_provider always resolves a URL, so this fires for a protocol
    # provider constructed by hand — fail loudly instead of letting the SDK
    # build a URL-less client.  Ordered before the key guard: with both
    # missing, the endpoint is the more fundamental config error.
    with pytest.raises(ValueError, match="needs a base_url"):
        resolve_brand_sdk("openai", api_key="k", base_url="")
    with pytest.raises(ValueError, match="'anthropic' provider needs a base_url"):
        resolve_brand_sdk("anthropic", api_key="k", base_url="")
    with pytest.raises(ValueError, match="needs a base_url"):
        resolve_brand_sdk("openai", api_key="", base_url="")


def test_resolve_empty_base_url_still_tries_a_registered_brand_adapter_first(monkeypatch) -> None:
    # A third-party adapter may not need a URL at all — the guard is on the
    # builtin path only.
    class _URLlessAdapter:
        supported_protocols = ["openai"]
        label = "urlless-sdk"

        @classmethod
        def create(cls, **kwargs):
            return cls()

    import cliver_core.provider.transports as transports_pkg

    monkeypatch.setattr(transports_pkg, "registered_adapters", lambda: {"gizmo": _URLlessAdapter})
    transport = resolve_brand_sdk("openai", api_key="k", base_url="", brand="gizmo")
    assert transport.label == "urlless-sdk"


def test_resolve_empty_key_names_the_brand_env_var() -> None:
    # The SDK's own error would suggest OPENAI_API_KEY — misleading for a
    # third-party brand; the guard names the brand's actual variable.
    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        resolve_brand_sdk("openai", api_key="", base_url="https://api.deepseek.com/v1", brand="deepseek")
    with pytest.raises(ValueError, match="'glm' provider needs an API key"):
        resolve_brand_sdk("anthropic", api_key="", base_url="https://open.bigmodel.cn/api/anthropic", brand="glm")
    with pytest.raises(ValueError, match="SEEDANCE_API_KEY"):
        resolve_brand_sdk("openai", api_key="", base_url="https://ark.cn-beijing.volces.com/api/v3", brand="seedance")


def test_resolve_empty_key_without_env_entry_offers_the_placeholder() -> None:
    # keyless local endpoints (Ollama) keep failing at the same point as
    # before — but with a legible message instead of the SDK's credentials error
    with pytest.raises(ValueError, match="placeholder"):
        resolve_brand_sdk("openai", api_key="", base_url="http://localhost:11434/v1", brand="ollama")


def test_resolve_empty_key_still_tries_a_registered_brand_adapter_first(monkeypatch) -> None:
    class _KeylessAdapter:
        supported_protocols = ["openai"]
        label = "keyless-sdk"

        @classmethod
        def create(cls, **kwargs):
            return cls()

    import cliver_core.provider.transports as transports_pkg

    monkeypatch.setattr(transports_pkg, "registered_adapters", lambda: {"localbrand": _KeylessAdapter})
    transport = resolve_brand_sdk("openai", api_key="", base_url="https://example.test/v1", brand="localbrand")
    assert transport.label == "keyless-sdk"


def test_registered_adapters_are_third_party_only() -> None:
    # The builtin SDK transports are NOT entry points — only third parties
    # register under the group (none in this environment).
    assert isinstance(registered_adapters(), dict)


# ── create_provider wiring (brand → transport) ─────────────────


@pytest.mark.asyncio
async def test_create_provider_picks_transport_by_brand() -> None:
    from cliver_core.provider.providers import create_provider

    # Third-party brand: no adapter registered → the protocol's builtin SDK
    deepseek = create_provider(api_key="k", protocol="openai", provider_name="deepseek")
    assert deepseek.brand == "deepseek"
    assert deepseek.transport.label == "openai-sdk"
    await deepseek.close()

    deepseek_anthropic = create_provider(api_key="k", protocol="anthropic", provider_name="deepseek")
    assert deepseek_anthropic.transport.label == "anthropic-sdk"
    await deepseek_anthropic.close()

    # Protocol-default brand: the official SDK adapter
    openai = create_provider(api_key="k", protocol="openai", provider_name="openai")
    assert openai.transport.label == "openai-sdk"
    await openai.close()

    seedance = create_provider(api_key="k", protocol="openai", provider_name="seedance")
    assert seedance.transport.label == "openai-sdk"
    await seedance.close()


# ── SDK transports (official SDKs + httpx.MockTransport) ───────


def _sse(*payloads: object) -> bytes:
    lines = []
    for p in payloads:
        data = p if isinstance(p, str) else json.dumps(p)
        lines.append(f"data: {data}\n\n")
    return "".join(lines).encode()


@pytest.mark.asyncio
async def test_openai_sdk_pipe_returns_wire_dicts() -> None:
    import httpx
    import openai

    from cliver_core.provider.transports.openai_sdk import OpenAISDK

    posts: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/audio/transcriptions"):
            raw = request.content
            posts.append(
                {
                    "multipart": request.headers["content-type"].startswith("multipart/form-data"),
                    "has_model": b"whisper-1" in raw,
                    "has_file": b"RIFF-DATA" in raw,
                    "language": request.headers.get("language"),
                }
            )
            return httpx.Response(200, json={"text": "hello world"})
        if request.url.path.endswith("/audio/speech"):
            posts.append({"path": request.url.path})
            return httpx.Response(200, content=b"AUDIOBYTES", headers={"content-type": "audio/mpeg"})

        body = json.loads(request.content)
        posts.append(body)
        if body.get("stream"):
            return httpx.Response(
                200,
                content=_sse(
                    {
                        "id": "c1",
                        "object": "chat.completion.chunk",
                        "created": 1,
                        "model": "m",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"role": "assistant", "content": "Hel", "reasoning_content": "tho"},
                                "finish_reason": None,
                            }
                        ],
                    },
                    {
                        "id": "c1",
                        "object": "chat.completion.chunk",
                        "created": 1,
                        "model": "m",
                        "choices": [{"index": 0, "delta": {"content": "lo"}, "finish_reason": None}],
                    },
                    # tool call fragmented mid-JSON — the SDK aggregates it
                    {
                        "id": "c1",
                        "object": "chat.completion.chunk",
                        "created": 1,
                        "model": "m",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "id": "call_1",
                                            "type": "function",
                                            "function": {"name": "echo", "arguments": '{"va'},
                                        }
                                    ]
                                },
                                "finish_reason": None,
                            }
                        ],
                    },
                    {
                        "id": "c1",
                        "object": "chat.completion.chunk",
                        "created": 1,
                        "model": "m",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"tool_calls": [{"index": 0, "function": {"arguments": 'lue": "x"}'}}]},
                                "finish_reason": "tool_calls",
                            }
                        ],
                    },
                    {
                        "id": "c1",
                        "object": "chat.completion.chunk",
                        "created": 1,
                        "model": "m",
                        "choices": [],
                        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
                    },
                    "[DONE]",
                ),
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(
            200,
            json={
                "id": "c1",
                "object": "chat.completion",
                "created": 1,
                "model": "m",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Hi",
                            # vendor extra — must survive the model_dump() round-trip
                            "reasoning_content": "thought",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            },
        )

    client = openai.AsyncOpenAI(
        api_key="k",
        base_url="https://example.test/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    transport = OpenAISDK(client)

    # the SDK parses; as_dict() hands the wire dict back — vendor extras included
    result = await transport.chat({"model": "m", "messages": [{"role": "user", "content": "hi"}]})
    assert result["choices"][0]["message"]["content"] == "Hi"
    assert result["choices"][0]["message"]["reasoning_content"] == "thought"
    assert result["usage"]["prompt_tokens"] == 3

    # typed-kwargs / extra_body split: vendor extras land in the posted JSON
    # exactly as prepared — wire-identical to posting the body verbatim
    vendor_body = {
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.5,
        "enable_thinking": True,
        # structured outputs: a JSON-schema response_format passes through untouched
        "response_format": {"type": "json_schema", "json_schema": {"name": "out", "schema": {"type": "object"}}},
    }
    await transport.chat(vendor_body)
    assert posts[-1] == vendor_body

    payloads = [
        p
        async for p in transport.stream(
            {
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                # structured outputs + streaming: the JSON-schema knob rides the
                # wire untouched and the SDK does not parse the content
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "out", "schema": {"type": "object"}},
                },
                # a caller's stream_options must survive the include_usage injection
                "stream_options": {"continuous_usage_stats": True},
            }
        )
    ]
    delta_frames = [p for p in payloads if p.get("choices")]
    assert "".join(p["choices"][0]["delta"].get("content") or "" for p in delta_frames) == "Hello"
    # the terminal event carries the SDK-aggregated final message — vendor
    # extras survive the aggregation, fragmented tool-call arguments are
    # joined into one JSON string, usage comes through the raw usage dict,
    # and a JSON-schema response_format stays inert (``parsed`` is None)
    final = payloads[-1]["__cliver_final__"]
    message = final["message"]
    assert message["role"] == "assistant"
    assert message["content"] == "Hello"
    assert message["reasoning_content"] == "tho"
    assert message["parsed"] is None
    (tool_call,) = message["tool_calls"]
    assert (tool_call["id"], tool_call["function"]["name"]) == ("call_1", "echo")
    assert tool_call["function"]["arguments"] == '{"value": "x"}'
    assert final["usage"]["prompt_tokens"] == 3
    # the transport owns making it a streaming call — flags and the schema
    # option land on the wire verbatim; include_usage is merged into (never
    # replaces) the caller's stream_options
    assert posts[-1]["stream"] is True
    assert posts[-1]["stream_options"] == {"continuous_usage_stats": True, "include_usage": True}
    assert posts[-1]["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "out", "schema": {"type": "object"}},
    }

    # post(): binary and multipart-with-files (ASR)
    spoken = await transport.post("/audio/speech", {"model": "m", "input": "hi"}, binary=True)
    assert spoken == b"AUDIOBYTES"

    asr = await transport.post(
        "/audio/transcriptions",
        {"model": "whisper-1", "language": "en"},
        files={"file": ("clip.wav", b"RIFF-DATA", "audio/wav")},
        headers={"language": "en"},
    )
    assert asr["text"] == "hello world"
    assert posts[-1] == {"multipart": True, "has_model": True, "has_file": True, "language": "en"}

    await transport.close()


@pytest.mark.asyncio
async def test_openai_sdk_pipe_restores_a_vendor_repeated_role() -> None:
    """MiniMax/GLM repeat ``role`` in every delta frame (live-observed).

    The SDK's delta merge concatenates strings, so raw aggregation of the
    final message yields ``"assistantassistant..."`` (pinned by a raw-SDK
    probe) — the transport must hand out the protocol-fixed assistant role.
    """
    import httpx
    import openai

    from cliver_core.messages import CLIverMessage
    from cliver_core.provider.transports.openai_sdk import OpenAISDK

    def frame(content: str) -> dict:
        return {
            "id": "c1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "m",
            "choices": [{"index": 0, "delta": {"content": content, "role": "assistant"}, "finish_reason": None}],
        }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=_sse(
                frame("Hel"),
                frame("lo"),
                {
                    "id": "c1",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": "m",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                },
                {
                    "id": "c1",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": "m",
                    "choices": [],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
                },
                "[DONE]",
            ),
            headers={"content-type": "text/event-stream"},
        )

    client = openai.AsyncOpenAI(
        api_key="k",
        base_url="https://example.test/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    transport = OpenAISDK(client)

    payloads = [p async for p in transport.stream({"model": "m", "messages": [{"role": "user", "content": "hi"}]})]

    message = payloads[-1]["__cliver_final__"]["message"]
    # the consumer (AgentCore) validates the role — "assistant" not "assistantassistant"
    assert CLIverMessage.from_native(message, "openai").role == "assistant"
    assert message["content"] == "Hello"
    # delta frames still pass through raw — the vendor's repeated role is untouched
    content_frames = [p for p in payloads[:-1] if p.get("choices") and p["choices"][0]["delta"].get("content")]
    assert [p["choices"][0]["delta"]["role"] for p in content_frames] == ["assistant", "assistant"]

    await transport.close()
    import anthropic
    import httpx2  # this SDK version ships on httpx2

    from cliver_core.provider.transports.anthropic_sdk import AnthropicSDK

    def anthropic_sse(*events: dict) -> bytes:
        return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()

    posts: list[dict] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path.endswith("/audio/transcriptions"):
            raw = request.content
            posts.append(
                {
                    "multipart": request.headers["content-type"].startswith("multipart/form-data"),
                    "has_model": b"whisper-1" in raw,
                    "has_file": b"RIFF-DATA" in raw,
                    "language": request.headers.get("language"),
                    # the client's own auth header must survive the merge
                    "api_key": request.headers.get("x-api-key"),
                }
            )
            return httpx2.Response(200, json={"text": "hello world"})
        if request.url.path.endswith("/audio/speech"):
            return httpx2.Response(200, content=b"AUDIOBYTES", headers={"content-type": "audio/mpeg"})

        body = json.loads(request.content)
        posts.append(body)
        if body.get("stream"):
            return httpx2.Response(
                200,
                content=anthropic_sse(
                    {
                        "type": "message_start",
                        "message": {
                            "id": "msg_1",
                            "type": "message",
                            "role": "assistant",
                            "model": "m",
                            "content": [],
                            "stop_reason": None,
                            "stop_sequence": None,
                            "usage": {"input_tokens": 3, "output_tokens": 0},
                        },
                    },
                    {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}},
                    {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "hmm"}},
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "signature_delta", "signature": "sig"},
                    },
                    {"type": "content_block_stop", "index": 0},
                    {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
                    {
                        "type": "content_block_delta",
                        "index": 1,
                        "delta": {"type": "text_delta", "text": "Hi"},
                    },
                    {"type": "content_block_stop", "index": 1},
                    {
                        "type": "message_delta",
                        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                        "usage": {"output_tokens": 2},
                    },
                    {"type": "message_stop"},
                ),
                headers={"content-type": "text/event-stream"},
            )
        return httpx2.Response(
            200,
            json={
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "m",
                "content": [
                    {"type": "thinking", "thinking": "hmm", "signature": "sig"},
                    {"type": "text", "text": "Hi"},
                    {"type": "tool_use", "id": "tu_1", "name": "my_tool", "input": {"v": 1}},
                ],
                "stop_reason": "tool_use",
                "stop_sequence": None,
                "usage": {"input_tokens": 3, "output_tokens": 2},
            },
        )

    client = anthropic.AsyncAnthropic(
        api_key="k",
        base_url="https://example.test",
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )
    transport = AnthropicSDK(client)

    # the SDK parses; as_dict() hands the wire dict back — vendor extras
    # (signature) included, names untouched (restore is provider work now)
    result = await transport.chat({"model": "m", "max_tokens": 64, "messages": [{"role": "user", "content": "hi"}]})
    assert result["id"] == "msg_1"
    assert result["content"][0]["thinking"] == "hmm"
    assert result["content"][0]["signature"] == "sig"
    assert result["content"][2]["name"] == "my_tool"
    assert result["usage"]["input_tokens"] == 3

    # typed-kwargs / extra_body split: this SDK version has no typed
    # temperature param, yet it lands in the posted JSON exactly as
    # prepared — wire-identical to posting the body verbatim
    temp_body = {
        "model": "m",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.5,
        # structured outputs: a JSON-schema output_config passes through untouched
        "output_config": {"format": {"type": "json_schema", "schema": {"type": "object"}}},
    }
    await transport.chat(temp_body)
    assert posts[-1] == temp_body

    payloads = [
        p
        async for p in transport.stream(
            {"model": "m", "max_tokens": 64, "messages": [{"role": "user", "content": "hi"}]}
        )
    ]
    deltas = [p for p in payloads if p.get("type") == "content_block_delta"]
    assert "".join(p["delta"].get("text") or "" for p in deltas) == "Hi"
    # the terminal event carries the SDK-aggregated final message — the
    # thinking block's signature is aggregated by the SDK, not by us
    final = payloads[-1]["__cliver_final__"]
    blocks = final["message"]["content"]
    assert final["message"]["role"] == "assistant"
    assert blocks[0]["type"] == "thinking"
    assert (blocks[0]["thinking"], blocks[0]["signature"]) == ("hmm", "sig")
    assert blocks[1]["type"] == "text" and blocks[1]["text"] == "Hi"
    assert final["usage"]["input_tokens"] == 3
    assert final["usage"]["output_tokens"] == 2
    # the SDK's stream() helper is itself the streaming call
    assert posts[-1]["stream"] is True

    # post(): binary and multipart-with-files (ASR)
    spoken = await transport.post("/audio/speech", {"model": "m", "input": "hi"}, binary=True)
    assert spoken == b"AUDIOBYTES"

    asr = await transport.post(
        "/audio/transcriptions",
        {"model": "whisper-1", "language": "en"},
        files={"file": ("clip.wav", b"RIFF-DATA", "audio/wav")},
        headers={"language": "en"},
    )
    assert asr["text"] == "hello world"
    assert posts[-1] == {
        "multipart": True,
        "has_model": True,
        "has_file": True,
        "language": "en",
        "api_key": "k",
    }

    await transport.close()


@pytest.mark.asyncio
async def test_anthropic_sdk_posts_an_absolute_url_verbatim_with_bearer() -> None:
    """MiniMax media posts absolute URLs on its own ``/v1`` domain through a
    client whose base_url is the anthropic-shaped chat root — the SDK must
    neither re-base the URL nor drop the Bearer auth (the media pipeline
    relies on exactly this)."""
    import anthropic
    import httpx2

    from cliver_core.provider.transports.anthropic_sdk import AnthropicSDK

    seen: list[dict] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append({"url": str(request.url), "auth": request.headers.get("authorization")})
        return httpx2.Response(200, json={"base_resp": {"status_code": 0}, "data": {"audio": "00ff"}})

    client = anthropic.AsyncAnthropic(
        auth_token="bearer-key",  # MiniMax rides Bearer auth, not x-api-key
        base_url="https://api.minimaxi.com/anthropic",
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )
    transport = AnthropicSDK(client)

    result = await transport.post("https://api.minimaxi.com/v1/t2a_v2", {"model": "m", "text": "hi"})

    assert seen[-1]["url"] == "https://api.minimaxi.com/v1/t2a_v2"  # absolute — base_url ignored
    assert seen[-1]["auth"] == "Bearer bearer-key"
    assert result["data"]["audio"] == "00ff"

    await transport.close()


@pytest.mark.asyncio
async def test_openai_sdk_posts_an_absolute_url_verbatim() -> None:
    """The openai transport passes absolute media URLs (brand maps may point
    at another domain) through verbatim, base_url notwithstanding."""
    import httpx
    import openai

    from cliver_core.provider.transports.openai_sdk import OpenAISDK

    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, json={"data": [{"url": "https://cdn.example/x.png"}]})

    client = openai.AsyncOpenAI(
        api_key="k",
        base_url="https://api.example/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    transport = OpenAISDK(client)

    result = await transport.post("https://edge.example/media/v9/generate", {"model": "m"})

    assert seen[-1] == "https://edge.example/media/v9/generate"
    assert result["data"][0]["url"] == "https://cdn.example/x.png"

    await transport.close()


# ── Protocol providers over a fake pipe ────────────────────────


class _FakeTransport:
    """Pure pipe: records bodies, replays canned wire payloads."""

    label = "fake"

    def __init__(self, chat_payload: dict | None = None, stream_payloads: list[dict] | None = None) -> None:
        self.chat_calls: list[dict] = []
        self.stream_calls: list[dict] = []
        self.post_calls: list[dict] = []
        self.closed = False
        self.chat_payload = chat_payload or {}
        self.stream_payloads = stream_payloads or []

    async def chat(self, body: dict) -> dict:
        self.chat_calls.append(body)
        return self.chat_payload

    async def stream(self, body: dict):
        self.stream_calls.append(body)
        for payload in self.stream_payloads:
            yield payload

    async def post(self, path, body, *, files=None, binary=False):
        self.post_calls.append({"path": path, "body": body, "files": files, "binary": binary})
        if binary:
            return b"AUDIO"
        if path.endswith("/audio/transcriptions"):
            return {"text": "transcribed"}
        return {"data": [{"url": "https://example.test/x.png"}]}

    async def close(self) -> None:
        self.closed = True


def _tool(name: str) -> CLIverTool:
    """Tool definition for protocol conversion (``execute`` is never called here)."""
    return CLIverTool(name=name, description="d", parameters={"type": "object"}, execute=lambda **kwargs: [])


_OPENAI_CHAT_PAYLOAD = {
    "id": "c1",
    "object": "chat.completion",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "reasoning_content": "deep thought",
                "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "echo", "arguments": '{"v": 1}'}}
                ],
            },
            "finish_reason": "tool_calls",
        }
    ],
    "usage": {"prompt_tokens": 3, "completion_tokens": 2},
}


@pytest.mark.asyncio
async def test_openai_provider_builds_body_and_converts_payload() -> None:
    from cliver_core.events import InferenceEvent, InferenceEventType

    events: list[InferenceEvent] = []

    async def on_event(event: InferenceEvent) -> None:
        events.append(event)

    transport = _FakeTransport(chat_payload=_OPENAI_CHAT_PAYLOAD)
    provider = OpenAIProvider("k", "https://example.test/v1", on_event=on_event, transport=transport)

    response = await provider.chat(
        CLIverRequest(
            messages=[CLIverMessage(role="user", content="hi")],
            tools=[],
            model="m",
            options={"temperature": 0.5},
        )
    )

    # request side: one assembled wire body — model, messages, options; no
    # tools key for an empty toolset, no stream flags (the transport owns those)
    assert transport.chat_calls == [{"model": "m", "messages": [{"role": "user", "content": "hi"}], "temperature": 0.5}]

    # response side: one conversion — CLIver types + the native message readable
    message = response.message
    assert message.role == "assistant"
    assert message.content is None
    assert message.tool_calls[0].name == "echo"
    assert message.tool_calls[0].args == {"v": 1}
    # vendor extras live in the native message, not on a side bag
    assert message.native_protocol == "openai"
    assert message.native["reasoning_content"] == "deep thought"
    assert message.native["tool_calls"] == _OPENAI_CHAT_PAYLOAD["choices"][0]["message"]["tool_calls"]
    # the native is a detached copy — never aliases the response payload
    assert message.native is not _OPENAI_CHAT_PAYLOAD["choices"][0]["message"]
    assert response.usage.input_tokens == 3
    # the request echo rides the response (AgentCore collects it into the trace)
    assert response.request_info["provider"] == "openai"
    assert response.request_info["messages"][-1]["content"] == "hi"
    assert [e.event for e in events] == [InferenceEventType.STARTED, InferenceEventType.COMPLETED]

    await provider.close()
    assert transport.closed


@pytest.mark.asyncio
async def test_openai_provider_stream_converts_deltas_and_skips_usage_frames() -> None:
    from cliver_core.events import InferenceEvent, InferenceEventType
    from cliver_core.messages import UsageInfo

    events: list[InferenceEvent] = []

    async def on_event(event: InferenceEvent) -> None:
        events.append(event)

    transport = _FakeTransport(
        stream_payloads=[
            {"choices": [{"index": 0, "delta": {"role": "assistant"}}]},
            {"choices": [{"index": 0, "delta": {"content": "Hel"}}]},
            {"choices": [{"index": 0, "delta": {"content": "lo", "reasoning_content": "..."}}]},
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [{"index": 0, "id": "call_1", "function": {"name": "echo", "arguments": ""}}]
                        },
                    }
                ]
            },
            {"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 2}},  # usage-only frame
            {
                "__cliver_final__": {
                    "message": {
                        "role": "assistant",
                        "content": "Hello",
                        "reasoning_content": "...",
                        "tool_calls": [
                            {"id": "call_1", "type": "function", "function": {"name": "echo", "arguments": '{"v": 1}'}}
                        ],
                    },
                    "usage": {"prompt_tokens": 3, "completion_tokens": 2},
                }
            },
        ]
    )
    provider = OpenAIProvider("k", "https://example.test/v1", on_event=on_event, transport=transport)

    chunks = [
        c
        async for c in provider.stream(
            CLIverRequest(
                messages=[CLIverMessage(role="user", content="hi")],
                model="m",
                options={"temperature": 0.5},
            )
        )
    ]

    # the recorded body carries the options; stream flags stay out (transport owns them)
    assert transport.stream_calls == [
        {"model": "m", "messages": [{"role": "user", "content": "hi"}], "temperature": 0.5}
    ]

    assert "".join(c.content or "" for c in chunks) == "Hello"
    assert chunks[2].vendor_ext["reasoning_content"] == "..."
    tool_chunks = [tc for c in chunks for tc in (c.tool_call_chunks or [])]
    assert tool_chunks[0].name == "echo"
    assert tool_chunks[0].id == "call_1"
    # the usage-only frame carries no delta — it produced no chunk
    assert len(chunks) == 5

    # terminal chunk — the SDK-aggregated native message + converted usage
    terminal = chunks[-1]
    assert terminal.content is None and terminal.tool_call_chunks is None
    assert terminal.native["content"] == "Hello"
    assert terminal.native["reasoning_content"] == "..."
    assert terminal.native["tool_calls"][0]["function"]["name"] == "echo"
    assert terminal.usage == UsageInfo(input_tokens=3, output_tokens=2)

    emitted = [e.event for e in events]
    assert InferenceEventType.FIRST_TOKEN in emitted
    assert InferenceEventType.TOOL_CALL in emitted
    assert emitted[-1] == InferenceEventType.COMPLETED


_ANTHROPIC_CHAT_PAYLOAD = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "content": [
        {"type": "thinking", "thinking": "hmm"},
        {"type": "text", "text": "ok"},
        {"type": "tool_use", "id": "tu_1", "name": "my_tool", "input": {"v": 1}},
    ],
    "usage": {"input_tokens": 5, "output_tokens": 2, "cache_read_input_tokens": 7},
}


@pytest.mark.asyncio
async def test_anthropic_provider_prepares_request_and_restores_names() -> None:
    transport = _FakeTransport(chat_payload=_ANTHROPIC_CHAT_PAYLOAD)
    provider = AnthropicProvider("k", "https://example.test", transport=transport)

    response = await provider.chat(
        CLIverRequest(
            messages=[
                CLIverMessage(role="system", content="sys"),
                # block-form content — the splitter must read text blocks too
                CLIverMessage(role="system", content=[{"type": "text", "text": "blocks"}]),
                CLIverMessage(role="user", content="hi"),
            ],
            tools=[_tool("my.tool")],
            model="m",
            options={"temperature": 0.5},
        )
    )

    # request side: system split out, dotted name sanitized, max_tokens
    # defaulted, options passed through (messages converted via to_native)
    body = transport.chat_calls[0]
    assert body["system"] == "sys\n\nblocks"
    assert body["messages"] == [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    assert body["tools"][0]["name"] == "my_tool"
    assert body["max_tokens"] == 4096
    assert body["temperature"] == 0.5

    # response side: content blocks joined, thinking kept in native, name restored
    message = response.message
    assert message.content == "ok"
    assert message.native_protocol == "anthropic"
    assert message.native["role"] == "assistant"
    assert message.native["content"][0] == {"type": "thinking", "thinking": "hmm"}
    assert message.tool_calls[0].name == "my.tool"  # restored from "my_tool"
    assert message.tool_calls[0].args == {"v": 1}
    assert response.usage.cache_read_input_tokens == 7


@pytest.mark.asyncio
async def test_anthropic_provider_sanitizes_echoed_tool_use_names() -> None:
    """Echoed assistant tool_use blocks satisfy the name constraint.

    Passthrough natives and stream-rebuilt messages may carry original
    names (e.g. ``my.tool``) — the API rejects those in ``tool_use``
    blocks, so the provider re-sanitizes with the definitions' forward map.
    """
    transport = _FakeTransport(chat_payload=_ANTHROPIC_CHAT_PAYLOAD)
    provider = AnthropicProvider("k", "https://example.test", transport=transport)

    echoed = CLIverMessage.from_native(
        {"role": "assistant", "content": [{"type": "tool_use", "id": "tu_9", "name": "my.tool", "input": {"v": 2}}]},
        "anthropic",
    )
    await provider.chat(
        CLIverRequest(
            messages=[CLIverMessage(role="user", content="hi"), echoed],
            tools=[_tool("my.tool")],
            model="m",
        )
    )

    body = transport.chat_calls[0]
    assert body["messages"][1]["content"][0]["name"] == "my_tool"
    # the message's native dict is never mutated — a sanitized copy was built
    assert echoed.native["content"][0]["name"] == "my.tool"


@pytest.mark.asyncio
async def test_anthropic_provider_names_consistent_across_paths() -> None:
    """One name convention: responses expose original names, ``native`` included.

    Both chat and stream restore the caller's names on the way in; the echo
    re-sanitizes to the wire-valid form — so both paths produce the
    identical request message.
    """
    tool_use = {"type": "tool_use", "id": "tu_1", "name": "my_tool", "input": {"v": 1}}
    transport = _FakeTransport(
        chat_payload={
            "id": "msg_1",
            "role": "assistant",
            "content": [dict(tool_use)],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
    )
    provider = AnthropicProvider("k", "https://example.test", transport=transport)

    response = await provider.chat(
        CLIverRequest(messages=[CLIverMessage(role="user", content="hi")], tools=[_tool("my.tool")], model="m")
    )
    chatted = response.message
    assert chatted.tool_calls[0].name == "my.tool"
    assert chatted.native["content"][0]["name"] == "my.tool"  # native agrees with the fields

    stream_transport = _FakeTransport(
        stream_payloads=[{"__cliver_final__": {"message": {"role": "assistant", "content": [dict(tool_use)]}}}]
    )
    stream_provider = AnthropicProvider("k", "https://example.test", transport=stream_transport)
    chunks = [
        c
        async for c in stream_provider.stream(
            CLIverRequest(messages=[CLIverMessage(role="user", content="hi")], tools=[_tool("my.tool")], model="m")
        )
    ]
    streamed = CLIverMessage.from_native(chunks[-1].native, "anthropic")  # what AgentCore does

    assert streamed.tool_calls[0].name == "my.tool"
    assert streamed.to_native("anthropic") == chatted.to_native("anthropic")

    # the echo re-sanitizes to the wire-valid name
    await provider.chat(
        CLIverRequest(
            messages=[CLIverMessage(role="user", content="hi"), streamed],
            tools=[_tool("my.tool")],
            model="m",
        )
    )
    assert transport.chat_calls[-1]["messages"][-1]["content"][0]["name"] == "my_tool"


@pytest.mark.asyncio
async def test_anthropic_provider_colliding_names_restore_uniquely() -> None:
    """``my.tool`` + ``my_tool`` both declared: each echo is restored exactly once.

    ``my_tool`` is the sanitized form of ``my.tool`` and also a real name —
    the second tool is deduped to ``my_tool_1``.  Restoration must map each
    block back once (no re-mapping of an already-restored name), and the
    echo must re-sanitize to the same deduped wire names.
    """
    transport = _FakeTransport(
        chat_payload={
            "id": "msg_1",
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "tu_1", "name": "my_tool", "input": {}},
                {"type": "tool_use", "id": "tu_2", "name": "my_tool_1", "input": {}},
            ],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
    )
    provider = AnthropicProvider("k", "https://example.test", transport=transport)
    tools = [_tool("my.tool"), _tool("my_tool")]

    response = await provider.chat(
        CLIverRequest(messages=[CLIverMessage(role="user", content="hi")], tools=tools, model="m")
    )
    assert [tc.name for tc in response.message.tool_calls] == ["my.tool", "my_tool"]

    echo = response.message
    await provider.chat(
        CLIverRequest(messages=[CLIverMessage(role="user", content="hi"), echo], tools=tools, model="m")
    )
    wire_names = [b["name"] for b in transport.chat_calls[-1]["messages"][-1]["content"]]
    assert wire_names == ["my_tool", "my_tool_1"]


@pytest.mark.asyncio
async def test_anthropic_provider_stream_restores_names_and_emits_first_token() -> None:
    from cliver_core.events import InferenceEvent, InferenceEventType
    from cliver_core.messages import UsageInfo

    events: list[InferenceEvent] = []

    async def on_event(event: InferenceEvent) -> None:
        events.append(event)

    transport = _FakeTransport(
        stream_payloads=[
            {"type": "message_start", "message": {"usage": {"input_tokens": 1}}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "a"}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "r"}},
            # signature deltas are not surfaced as chunks — the SDK folds the
            # signature into the terminal message
            {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "sig"}},
            {
                "type": "content_block_start",
                "index": 1,
                "content_block": {"type": "tool_use", "id": "tu_1", "name": "my_tool"},
            },
            {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "input_json_delta", "partial_json": '{"v": 1}'},
            },
            {"type": "message_stop"},
            {
                "__cliver_final__": {
                    "message": {
                        "role": "assistant",
                        "content": [
                            {"type": "thinking", "thinking": "r", "signature": "sig"},
                            {"type": "text", "text": "a"},
                            {"type": "tool_use", "id": "tu_1", "name": "my_tool", "input": {"v": 1}},
                        ],
                    },
                    "usage": {"input_tokens": 1, "output_tokens": 2},
                }
            },
        ]
    )
    provider = AnthropicProvider("k", "https://example.test", on_event=on_event, transport=transport)

    chunks = [
        c
        async for c in provider.stream(
            CLIverRequest(messages=[CLIverMessage(role="user", content="hi")], tools=[_tool("my.tool")], model="m")
        )
    ]

    # bookkeeping events (message_start / message_stop) and the signature
    # delta produce no chunks; text, thinking, tool_use start and
    # input_json_delta each produce one
    assert [c.content for c in chunks] == ["a", None, None, None, None]
    assert chunks[1].vendor_ext["thinking"] == "r"
    tool_chunks = [tc for c in chunks for tc in (c.tool_call_chunks or [])]
    assert tool_chunks[0].name == "my.tool"  # restored from sanitized "my_tool"
    assert tool_chunks[0].id == "tu_1"
    assert tool_chunks[1].args_delta == '{"v": 1}'

    # terminal chunk — the aggregated native message with sanitized tool
    # names restored for dispatch
    terminal = chunks[-1]
    assert terminal.native["content"][0]["signature"] == "sig"
    assert terminal.native["content"][2]["name"] == "my.tool"
    assert terminal.usage == UsageInfo(input_tokens=1, output_tokens=2)

    emitted = [e.event for e in events]
    assert InferenceEventType.FIRST_TOKEN in emitted
    assert InferenceEventType.TOOL_CALL in emitted
    tool_event = next(e for e in events if e.event is InferenceEventType.TOOL_CALL)
    assert tool_event.data["tool_name"] == "my.tool"
    assert emitted[-1] == InferenceEventType.COMPLETED


@pytest.mark.asyncio
async def test_provider_stream_without_terminal_event_fails_fast() -> None:
    """A transport that never yields the reserved terminal event is an adapter bug."""
    openai_transport = _FakeTransport(stream_payloads=[{"choices": [{"index": 0, "delta": {"content": "Hi"}}]}])
    openai_provider = OpenAIProvider("k", "https://example.test/v1", transport=openai_transport)
    openai_request = CLIverRequest(messages=[CLIverMessage(role="user", content="hi")], model="m")
    with pytest.raises(RuntimeError, match="__cliver_final__"):
        [c async for c in openai_provider.stream(openai_request)]

    anthropic_transport = _FakeTransport(
        stream_payloads=[{"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hi"}}]
    )
    anthropic_provider = AnthropicProvider("k", "https://example.test", transport=anthropic_transport)
    anthropic_request = CLIverRequest(messages=[CLIverMessage(role="user", content="hi")], model="m")
    with pytest.raises(RuntimeError, match="__cliver_final__"):
        [c async for c in anthropic_provider.stream(anthropic_request)]


@pytest.mark.asyncio
async def test_structured_output_options_pass_through_untouched() -> None:
    """Vendor JSON-schema knobs ride ``options`` verbatim — chat and stream alike.

    OpenAI's ``response_format`` and Anthropic's ``output_config`` are
    never inspected by cliver-core; they land in the request body as given.
    """
    schema = {"type": "json_schema", "json_schema": {"name": "out", "schema": {"type": "object"}}}
    user = [CLIverMessage(role="user", content="hi")]
    openai_transport = _FakeTransport(
        chat_payload=_OPENAI_CHAT_PAYLOAD,
        stream_payloads=[{"__cliver_final__": {"message": {"role": "assistant", "content": "{}"}}}],
    )
    openai_provider = OpenAIProvider("k", "https://example.test/v1", transport=openai_transport)

    await openai_provider.chat(CLIverRequest(messages=user, model="m", options={"response_format": schema}))
    assert openai_transport.chat_calls[0]["response_format"] == schema

    _ = [
        c
        async for c in openai_provider.stream(
            CLIverRequest(messages=user, model="m", options={"response_format": schema})
        )
    ]
    assert openai_transport.stream_calls[0]["response_format"] == schema

    output_config = {"format": {"type": "json_schema", "schema": {"type": "object"}}}
    anthropic_transport = _FakeTransport(
        chat_payload=_ANTHROPIC_CHAT_PAYLOAD,
        stream_payloads=[{"__cliver_final__": {"message": {"role": "assistant", "content": []}}}],
    )
    anthropic_provider = AnthropicProvider("k", "https://example.test", transport=anthropic_transport)

    await anthropic_provider.chat(CLIverRequest(messages=user, model="m", options={"output_config": output_config}))
    assert anthropic_transport.chat_calls[0]["output_config"] == output_config

    _ = [
        c
        async for c in anthropic_provider.stream(
            CLIverRequest(messages=user, model="m", options={"output_config": output_config})
        )
    ]
    assert anthropic_transport.stream_calls[0]["output_config"] == output_config


@pytest.mark.asyncio
async def test_openai_provider_media_goes_through_transport() -> None:
    from cliver_core.media import MediaContent, MediaType

    transport = _FakeTransport()
    provider = OpenAIProvider("k", "https://example.test/v1", transport=transport)

    # image — reference media becomes a data URL; result URL is wrapped
    ref = MediaContent(type=MediaType.IMAGE, data="AAAA", mime_type="image/png")
    response = await provider.image("a cat", "img-model", media=[ref])
    call = transport.post_calls[-1]
    assert call["path"] == "https://example.test/v1/images/generations"  # absolute — transport-verbatim
    assert call["body"]["image"] == "data:image/png;base64,AAAA"
    assert response.media[0].data == "https://example.test/x.png"

    # tts → binary post to the speech endpoint; the bytes come back b64-wrapped
    tts_response = await provider.tts("say hi", "tts-model", voice="nova")
    call = transport.post_calls[-1]
    assert call["path"] == "https://example.test/v1/audio/speech"
    assert call["binary"] is True
    assert call["body"]["input"] == "say hi" and call["body"]["voice"] == "nova"
    assert call["body"]["response_format"] == "mp3"
    (mc,) = tts_response.media
    assert mc.mime_type == "audio/mpeg"
    assert base64.b64decode(mc.data) == b"AUDIO"

    # video
    await provider.video("a sunset", "v-model")
    assert transport.post_calls[-1]["path"] == "https://example.test/v1/videos/generations"

    # ASR — multipart upload
    audio = MediaContent(type=MediaType.AUDIO, data="QUJD", mime_type="audio/wav")
    at = await provider.asr(audio, "whisper-1", language="en")
    call = transport.post_calls[-1]
    assert call["path"] == "https://example.test/v1/audio/transcriptions"
    assert call["body"] == {"model": "whisper-1", "language": "en"}
    assert call["files"] == {"file": ("audio.wav", b"ABC", "audio/wav")}
    assert at.message.content == "transcribed"


@pytest.mark.asyncio
async def test_openai_video_accepts_the_video_url_field() -> None:
    """Some OpenAI-fashion video endpoints name the result URL ``video_url``."""
    from cliver_core.media import MediaType

    class _VideoTransport:
        async def post(self, path, body, *, files=None, binary=False):
            return {"video_url": "https://cdn.example/clip.mp4"}

    provider = OpenAIProvider("k", "https://example.test/v1", transport=_VideoTransport())
    (mc,) = (await provider.video("a sunset", "v-model")).media
    assert mc.type == MediaType.VIDEO
    assert mc.data == "https://cdn.example/clip.mp4"
    assert mc.mime_type == "video/mp4"
