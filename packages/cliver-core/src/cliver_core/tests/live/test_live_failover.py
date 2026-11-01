"""Live: a retryable provider failure moves to the next same-purpose LLM.

Every check mounts two ``llm`` bindings.  The fallback is a real provider
(DeepSeek); the primary is built to fail the way a real outage does, and
the last check uses a real vendor endpoint that is known to answer 429 on
this account.

A local stub endpoint stands in for the failing vendor because no provider
can be asked to return a 429 on demand.  The request still travels through
the real vendor SDK over real HTTP, so the classification under test
(``status_code`` / exception-class based ``retryable``) is the production
one.  Covered failures:

* HTTP 429 — the ``retryable`` -> next binding path (chat and stream);
* connection refused — an ``APIConnectionError`` on both protocols (loopback
  bypasses any system HTTP proxy so the refused path stays deterministic);
* HTTP 400 — a non-retryable failure that must **not** fail over;
* a real GLM 429 (insufficient balance) -> a real DeepSeek answer, when
  this account's pay-as-you-go endpoint still refuses premium models;
* failover stickiness — after a mid-run switch, later iterations of the
  same run stay on the fallback (the primary is not retried).
"""

from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from cliver_core import AgentCore, ModelBinding, ModelBindings, ProviderError, presets
from cliver_core.events import InferenceEventType
from cliver_core.tests.live.support import (
    LIVE_OFF_REASON,
    EventCollector,
    drain,
    live_enabled,
    skip_unless_keys,
    skip_with_warning,
    stream_text,
    stream_trace,
)
from cliver_core.tool import tool

pytestmark = pytest.mark.skipif(not live_enabled(), reason=LIVE_OFF_REASON)


class StubEndpoint:
    """A local HTTP endpoint that answers every POST with one fixed status."""

    def __init__(self, status: int) -> None:
        self.status = status
        self.requests: list[dict[str, Any]] = []
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        stub = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self) -> None:  # noqa: N802 — stdlib hook name
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b""
                try:
                    body = json.loads(raw or b"{}")
                except json.JSONDecodeError:
                    body = {"raw": raw.decode("utf-8", "replace")}
                stub.requests.append({"path": self.path, "body": body})

                payload = json.dumps(
                    {"error": {"message": f"stub status {stub.status}", "type": "stub_error"}}
                ).encode()
                self.send_response(stub.status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args: Any) -> None:
                pass  # keep the pytest output clean

        return Handler

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}/v1"

    def __enter__(self) -> "StubEndpoint":
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


def _stub_binding(stub: StubEndpoint, events: EventCollector) -> ModelBinding:
    """A primary binding aimed at the stub — no vendor key is used or needed."""
    return presets.deepseek(
        api_key="stub-key",
        base_url=stub.base_url,
        model="stub-primary",
        client_max_retries=0,
        client_timeout=10,
        temperature=0,
        on_event=events,
    )


def _real_fallback(events: EventCollector) -> ModelBinding:
    return presets.deepseek(temperature=0, on_event=events)


def _errors(events: EventCollector) -> list[Any]:
    return [event for event in events.events if event.event == InferenceEventType.ERROR]


# A tool-using run so a mid-run failover spans more than one model call.
@tool(name="lookup_order", description="Return the packed status, ETA, and carrier for one order id.")
def lookup_order(order_id: str) -> str:
    return f"Order {order_id}: packed, ETA 2 days, carrier Foo Logistics."


ORDER_SYSTEM = (
    "You are an order assistant. For any order question you must call lookup_order "
    "exactly once with the order id, then answer in one short sentence."
)


async def test_chat_fails_over_from_a_429_to_a_real_llm() -> None:
    skip_unless_keys("DEEPSEEK_API_KEY")
    primary_events = EventCollector()
    fallback_events = EventCollector()
    run_id = "live-failover-429"

    with StubEndpoint(429) as stub:
        agent = AgentCore(
            ModelBindings(llm=[_stub_binding(stub, primary_events), _real_fallback(fallback_events)]),
            run_timeout=120,
        )
        try:
            response = await agent.chat("Reply with exactly the word OK.", run_id=run_id)
        finally:
            await agent.aclose()

    assert response.message.content
    assert response.run_id == run_id
    assert response.usage_total is not None

    # The primary really was called once and classified retryable.
    assert len(stub.requests) == 1
    assert stub.requests[0]["body"].get("model") == "stub-primary"
    errors = _errors(primary_events)
    assert len(errors) == 1, f"expected one primary error event, got {errors}"
    assert errors[0].data.get("retryable") is True
    assert errors[0].run_id == run_id

    # The real fallback answered, and its events carry the same run id.
    trace = response.message.trace or []
    assert trace and trace[0]["model"] == presets.deepseek().model
    assert fallback_events.for_run(run_id)


async def test_stream_fails_over_before_the_first_chunk() -> None:
    skip_unless_keys("DEEPSEEK_API_KEY")
    primary_events = EventCollector()
    fallback_events = EventCollector()
    run_id = "live-failover-stream"

    with StubEndpoint(429) as stub:
        agent = AgentCore(
            ModelBindings(llm=[_stub_binding(stub, primary_events), _real_fallback(fallback_events)]),
            run_timeout=120,
        )
        try:
            chunks = await drain(agent.stream("Reply with exactly the word OK.", run_id=run_id))
        finally:
            await agent.aclose()

    assert stream_text(chunks).strip()
    entries = stream_trace(chunks)
    assert entries and entries[0]["model"] == presets.deepseek().model
    assert len(stub.requests) == 1
    errors = _errors(primary_events)
    assert len(errors) == 1
    assert errors[0].data.get("retryable") is True
    assert fallback_events.for_run(run_id)


@pytest.mark.parametrize(
    ("protocol", "factory"),
    [
        pytest.param("openai", presets.deepseek, id="openai-protocol"),
        pytest.param("anthropic", presets.deepseek_anthropic, id="anthropic-protocol"),
    ],
)
async def test_chat_fails_over_from_a_network_error(protocol: str, factory, monkeypatch) -> None:
    skip_unless_keys("DEEPSEEK_API_KEY")
    primary_events = EventCollector()
    fallback_events = EventCollector()
    run_id = f"live-failover-network-{protocol}"

    # Bypass a system HTTP proxy for loopback.  Without this, a machine-wide
    # proxy (macOS system settings included) answers 502 for the unreachable
    # port instead of letting the SDK raise its connection error.  The check
    # is about the refused-connection path, so pin it.
    no_proxy = ",".join(filter(None, [os.environ.get("NO_PROXY", ""), "127.0.0.1,localhost"]))
    monkeypatch.setenv("NO_PROXY", no_proxy)
    monkeypatch.setenv("no_proxy", no_proxy)

    # Nothing listens on port 1 — the real SDK raises its connection error.
    primary = factory(
        api_key="stub-key",
        base_url="http://127.0.0.1:1/v1",
        model="stub-primary",
        client_max_retries=0,
        client_timeout=5,
        temperature=0,
        on_event=primary_events,
    )
    fallback = factory(temperature=0, on_event=fallback_events)
    agent = AgentCore(ModelBindings(llm=[primary, fallback]), run_timeout=120)
    try:
        response = await agent.chat("Reply with exactly the word OK.", run_id=run_id)
    finally:
        await agent.aclose()

    assert response.message.content
    assert response.run_id == run_id
    errors = _errors(primary_events)
    assert len(errors) == 1, f"expected one primary error event, got {errors}"
    assert errors[0].data.get("retryable") is True
    assert "connect" in (errors[0].data.get("error") or "").lower()
    trace = response.message.trace or []
    assert trace and trace[0]["model"] == fallback.model
    assert fallback_events.for_run(run_id)


async def test_chat_does_not_fail_over_on_a_non_retryable_failure() -> None:
    skip_unless_keys("DEEPSEEK_API_KEY")
    primary_events = EventCollector()
    fallback_events = EventCollector()

    with StubEndpoint(400) as stub:
        agent = AgentCore(
            ModelBindings(llm=[_stub_binding(stub, primary_events), _real_fallback(fallback_events)]),
            run_timeout=60,
        )
        try:
            with pytest.raises(ProviderError) as raised:
                await agent.chat("Reply with exactly the word OK.", run_id="live-no-failover-400")
        finally:
            await agent.aclose()

    assert raised.value.retryable is False
    assert "400" in str(raised.value)
    assert len(stub.requests) == 1
    assert not fallback_events.events, "the fallback must not run after a non-retryable failure"


async def test_chat_stays_on_the_fallback_after_a_mid_run_failover() -> None:
    skip_unless_keys("DEEPSEEK_API_KEY")
    primary_events = EventCollector()
    fallback_events = EventCollector()
    run_id = "live-failover-sticky"

    with StubEndpoint(429) as stub:
        agent = AgentCore(
            ModelBindings(llm=[_stub_binding(stub, primary_events), _real_fallback(fallback_events)]),
            builtin_tools=[lookup_order],
            run_timeout=180,
        )
        try:
            response = await agent.chat(
                "Where is order G-700? Call lookup_order exactly once with G-700.",
                system_prompt=ORDER_SYSTEM,
                run_id=run_id,
            )
        finally:
            await agent.aclose()

    assert response.message.content
    # Two model calls happened (tool round + final answer), both on the fallback:
    # the primary was tried once, at the start of the run, and never again.
    assert len(stub.requests) == 1
    trace = response.message.trace or []
    assert len(trace) >= 2
    assert {entry["model"] for entry in trace} == {presets.deepseek().model}


async def test_failover_from_a_real_429_on_the_glm_preset() -> None:
    """This account's GLM pay-as-you-go endpoint answers premium models with 429.

    When the account has balance the primary simply answers; the leg is then
    skipped with a warning rather than asserting a failure the vendor no
    longer produces.
    """
    skip_unless_keys("GLM_API_KEY", "DEEPSEEK_API_KEY")
    primary_events = EventCollector()
    fallback_events = EventCollector()
    run_id = "live-failover-real-429"

    # presets.glm() resolves the pay-as-you-go default endpoint, not the
    # coding-plan override used by the brand smoke test.
    primary = presets.glm(temperature=0, on_event=primary_events)
    fallback = _real_fallback(fallback_events)
    agent = AgentCore(ModelBindings(llm=[primary, fallback]), run_timeout=120)
    try:
        response = await agent.chat("Reply with exactly the word OK.", run_id=run_id)
    finally:
        await agent.aclose()

    errors = _errors(primary_events)
    if not errors:
        skip_with_warning(
            "live verification skipped: the GLM pay-as-you-go endpoint answered, so this "
            "account now has balance and the real-429 failover leg was not exercised"
        )
    assert errors[0].data.get("retryable") is True, f"GLM failed non-retryably: {errors[0].data}"
    assert "429" in (errors[0].data.get("error") or "")

    assert response.message.content
    assert response.run_id == run_id
    trace = response.message.trace or []
    assert trace and trace[0]["model"] == fallback.model
    assert fallback_events.for_run(run_id)
