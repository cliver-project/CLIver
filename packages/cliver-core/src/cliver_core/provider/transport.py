"""Transport layer for protocol providers.

A :class:`BrandSDK` is a thin I/O pipe over ONE client — the brand's official
SDK when one is registered, else the protocol's builtin adapter (the official
OpenAI / Anthropic SDK pointed at the brand's ``base_url``).  It does no
message work at all: the provider hands it a fully prepared wire-format
request body, and it returns the endpoint's parsed payload as a plain dict —
exactly what the SDK already parsed (``as_dict``).  All protocol shape
(request assembly, response extraction, conversion to CLIver types) lives in
the protocol providers; no intermediate message types exist between the SDK
and the provider's single conversion.

Resolution lives in :mod:`cliver_core.provider.transports`:

- third-party brand SDK adapter registered under the ``cliver.provider_sdks``
  entry-point group (a wrapper around e.g. the MiniMax / GLM SDK)
- builtin → the official OpenAI / Anthropic SDK (a hard cliver-core
  dependency) speaking the standard wire protocol
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Protocol, runtime_checkable


def as_dict(obj: Any) -> dict[str, Any]:
    """Best-effort dump of an SDK object to a plain dict.

    Pydantic-based SDK objects expose ``model_dump()`` (which keeps extra
    vendor fields such as ``reasoning_content``); plain dicts pass through.
    """
    dump = getattr(obj, "model_dump", None)
    if callable(dump):
        try:
            return dump()
        except Exception:  # pragma: no cover - defensive
            pass
    return obj if isinstance(obj, dict) else {}


@runtime_checkable
class BrandSDK(Protocol):
    """Thin I/O pipe over one brand's SDK — or the protocol's builtin one.

    Convention: every transport class provides ``label``,
    ``supported_protocols`` and a ``create(*, api_key, base_url, protocol,
    user_agent, use_bearer_auth)`` classmethod.  ``create`` may accept the
    optional ``timeout`` and ``max_retries`` keywords for SDK-client tuning
    (forwarded only when the binding sets them) and extra keyword arguments
    of its own (tests use them).

    The interface is deliberately primitive — body dicts in, payload dicts
    out — so adapters stay trivial and no intermediate message types creep in
    between the SDK and the provider's conversion.  ``stream`` owns making the
    call a streaming call (wire flags / the SDK's streaming helper); the body
    itself is passed through verbatim.

    Wire shape is the contract: ``chat`` returns, and ``stream`` yields,
    dicts in the protocol's native form (``role`` / ``content`` /
    ``tool_calls`` for openai, content blocks for anthropic) — exactly the
    keys :meth:`CLIverMessage.from_native` reads.  A brand SDK adapter
    normalizes its own structures into that shape; vendor extras need no
    mapping — they ride in the dict, stay verbatim on ``CLIverMessage.native``,
    and round-trip through ``to_native``.

    Plain JSON dicts are the only currency in both directions — no SDK
    objects cross this seam, and nothing outside the JSON structure is
    inspected (vendor quirks such as thinking fields are read by their
    documented keys, per SDK).
    """

    label: str
    # Short identifier for logs/diagnostics, e.g. "openai-sdk", "anthropic-sdk".

    async def chat(self, body: dict) -> dict:
        """One non-streaming completion: POST the prepared *body*, return the parsed payload."""
        ...

    def stream(self, body: dict) -> AsyncIterator[dict]:
        """Streaming completion — the protocol's stream payloads, delta by delta.

        Payloads are the SDK's parsed wire dicts, yielded verbatim, and the
        stream MUST end with exactly one reserved terminal event::

            {"__cliver_final__": {"message": <native message payload>,
                                  "usage": <raw usage dict | None>}}

        ``message`` is the aggregated assistant message in the protocol's
        per-message wire shape — the keys :meth:`CLIverMessage.from_native`
        reads (vendor extras and thinking signatures included); ``usage`` is
        the vendor's raw usage dict when the stream carries one.  The SDK's
        own aggregator produces it — both builtin SDKs provide one; an
        adapter whose SDK cannot MUST accumulate it on its side of the seam
        (the provider has no fallback path).
        """
        ...

    async def post(
        self,
        path: str,
        body: dict,
        *,
        files: dict | None = None,
        binary: bool = False,
        headers: dict | None = None,
    ) -> Any:
        """Raw vendor call for media endpoints (images, speech, video, ...).

        Returns parsed JSON, or the raw response bytes when ``binary=True``.
        *headers* is per-request extras for brands that take an option as an
        HTTP header (e.g. MiniMax's transcription ``language``); providers
        pass it only when non-empty, so implementing it is optional for
        custom transports.
        """
        ...

    async def close(self) -> None:
        """Release the underlying client and its connections."""
        ...
