"""Official ``openai`` package transport — a hard cliver-core dependency.

Pure I/O pipe: the SDK handles all transport, communication and stream
aggregation — the provider-prepared body is passed to
``chat.completions.create`` / ``chat.completions.stream`` and the SDK's
parsed response is returned as a wire dict (``as_dict``).
"""

from __future__ import annotations

import inspect
import json
from typing import Any, AsyncIterator

from openai import AsyncOpenAI

from cliver_core.provider.transport import as_dict

# Typed create() parameters, derived from the installed SDK's signature so
# the split never goes stale: known keys become kwargs, everything else
# merges into ``extra_body`` (which the SDK folds into the top-level JSON —
# wire-identical to posting the body verbatim).
_CALL_PASSTHROUGH = frozenset({"extra_headers", "extra_query", "extra_body", "timeout"})
# A throwaway client (no request is ever sent) reaches the typed signature
# in a version-robust way.
_REQUEST_PARAMS = (
    frozenset(inspect.signature(AsyncOpenAI(api_key="x").chat.completions.create).parameters) - _CALL_PASSTHROUGH
)


def _split_body(body: dict) -> tuple[dict, dict | None]:
    """Split a wire body into (typed kwargs, extra_body|None)."""
    kwargs = {k: v for k, v in body.items() if k in _REQUEST_PARAMS}
    extra = {k: v for k, v in body.items() if k not in _REQUEST_PARAMS}
    return kwargs, (extra or None)


def _normalize_streamed_role(message: dict) -> dict:
    """Restore the protocol-fixed role on an SDK-aggregated streamed message.

    A streamed chat message is assistant-role by construction, but the SDK's
    delta merge (``accumulate_delta``) concatenates *every* string field — so
    a vendor that repeats the literal role in each delta frame (MiniMax, GLM:
    ``"delta": {"content": "...", "role": "assistant"}`` on every chunk)
    aggregates to ``"assistantassistant..."``, which then fails role
    validation downstream.  Vendors that omit the role entirely land here
    too.  Empty payloads (no choices arrived at all) pass through untouched.
    """
    if message and message.get("role") != "assistant":
        return {**message, "role": "assistant"}
    return message


class OpenAISDK:
    """BrandSDK over the official ``openai`` SDK (AsyncOpenAI client).

    The builtin transport for the ``openai`` protocol — every
    OpenAI-compatible endpoint (DeepSeek, GLM, Ollama, ...) is consumed
    through it with a custom ``base_url``.
    """

    supported_protocols = ["openai"]
    label = "openai-sdk"

    def __init__(self, client: AsyncOpenAI):
        self._client = client

    @classmethod
    def create(
        cls,
        *,
        api_key: str,
        base_url: str,
        protocol: str = "openai",
        user_agent: str | None = None,
        use_bearer_auth: bool = False,
    ) -> "OpenAISDK":
        headers = {"User-Agent": user_agent} if user_agent else {}
        return cls(AsyncOpenAI(api_key=api_key, base_url=base_url, default_headers=headers))

    # ── BrandSDK ───────────────────────────────────────────────

    async def chat(self, body: dict) -> dict:
        kwargs, extra = _split_body(body)
        response = await self._client.chat.completions.create(**kwargs, extra_body=extra)
        return as_dict(response)

    async def stream(self, body: dict) -> AsyncIterator[dict]:
        # The SDK's stream() helper IS the streaming call and it aggregates
        # the final message for us — no client-side assembly.  Ordinary
        # chunks carry the deltas (vendor extras included); the reserved
        # terminal event carries the aggregated message + raw usage.
        body = dict(body)
        body.pop("stream", None)  # the helper takes no `stream` kwarg
        kwargs, extra = _split_body(body)
        # usage arrives on a trailing choices-less chunk; ``include_usage``
        # is the terminal's contract, so it rides along with any caller
        # stream_options instead of overriding them
        kwargs["stream_options"] = {**(kwargs.get("stream_options") or {}), "include_usage": True}
        async with self._client.chat.completions.stream(**kwargs, extra_body=extra) as sdk_stream:
            async for event in sdk_stream:
                if event.type == "chunk":  # raw chunk — vendor extras ride along
                    yield as_dict(event.chunk)
            # The aggregation keeps vendor extras (reasoning_content, ...)
            # because the SDK's delta merge is a generic dict merge and its
            # API models allow extras — pinned end-to-end by the SDK pipe
            # test.  If a future SDK drops them, re-attach the observed
            # delta keys on this side of the seam.
            final = await sdk_stream.get_final_completion()

        payload = as_dict(final)
        choices = payload.get("choices") or []
        message = (choices[0].get("message") if choices else None) or {}
        yield {"__cliver_final__": {"message": _normalize_streamed_role(message), "usage": payload.get("usage")}}

    async def post(
        self,
        path: str,
        body: dict,
        *,
        files: dict | None = None,
        binary: bool = False,
        headers: dict | None = None,
    ) -> Any:
        options: dict[str, Any] = {}
        if files is not None:
            # The SDK only serializes `body` into multipart form fields (and
            # lets httpx generate the boundary) when this header is set;
            # without it, `body` would be dropped from the request.
            options["headers"] = {"Content-Type": "multipart/form-data"}
        if headers:
            # Per-request extras (e.g. MiniMax's transcription `language`)
            # merge over the multipart Content-Type; the client's default
            # headers (auth) are untouched.
            options["headers"] = {**options.get("headers", {}), **headers}
        raw = await self._client.post(path, cast_to=bytes, body=body or None, files=files, options=options)
        if binary:
            return raw
        return json.loads(raw) if raw else {}

    async def close(self) -> None:
        await self._client.close()
