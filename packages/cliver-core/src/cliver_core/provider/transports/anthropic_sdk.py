"""Official ``anthropic`` package transport — a hard cliver-core dependency.

Pure I/O pipe: the SDK handles all transport and communication — the
provider-prepared body is passed to ``messages.create`` / ``messages.stream``
and the SDK's parsed response is returned as a wire dict (``as_dict``).
"""

from __future__ import annotations

import inspect
import json
from typing import Any, AsyncIterator

from anthropic import AsyncAnthropic

from cliver_core.provider.transport import as_dict

# Typed create() parameters, derived from the installed SDK's signature so
# the split never goes stale: known keys become kwargs, everything else
# merges into ``extra_body`` (which the SDK folds into the top-level JSON —
# wire-identical to posting the body verbatim).  Needed because e.g.
# anthropic 1.12.1's create() has no typed temperature/top_p/top_k params,
# yet they are perfectly valid on the wire.
_CALL_PASSTHROUGH = frozenset({"extra_headers", "extra_query", "extra_body", "timeout"})
# A throwaway client (no request is ever sent) reaches the typed signature
# in a version-robust way.
_REQUEST_PARAMS = (
    frozenset(inspect.signature(AsyncAnthropic(api_key="x").messages.create).parameters) - _CALL_PASSTHROUGH
)


def _split_body(body: dict) -> tuple[dict, dict | None]:
    """Split a wire body into (typed kwargs, extra_body|None)."""
    kwargs = {k: v for k, v in body.items() if k in _REQUEST_PARAMS}
    extra = {k: v for k, v in body.items() if k not in _REQUEST_PARAMS}
    return kwargs, (extra or None)


class AnthropicSDK:
    """BrandSDK over the official ``anthropic`` SDK (AsyncAnthropic client).

    The builtin transport for the ``anthropic`` protocol.
    """

    supported_protocols = ["anthropic"]
    label = "anthropic-sdk"

    def __init__(self, client: AsyncAnthropic):
        self._client = client

    @classmethod
    def create(
        cls,
        *,
        api_key: str,
        base_url: str,
        protocol: str = "anthropic",
        user_agent: str | None = None,
        use_bearer_auth: bool = False,
    ) -> "AnthropicSDK":
        headers = {"User-Agent": user_agent} if user_agent else {}
        # Anthropic SDK: api_key → x-api-key header, auth_token → Authorization: Bearer
        # Third-party providers (DeepSeek, MiniMax) use Bearer auth.
        if use_bearer_auth:
            client = AsyncAnthropic(auth_token=api_key, base_url=base_url, default_headers=headers)
        else:
            client = AsyncAnthropic(api_key=api_key, base_url=base_url, default_headers=headers)
        return cls(client)

    # ── BrandSDK ───────────────────────────────────────────────

    async def chat(self, body: dict) -> dict:
        kwargs, extra = _split_body(body)
        response = await self._client.messages.create(**kwargs, extra_body=extra)
        return as_dict(response)

    async def stream(self, body: dict) -> AsyncIterator[dict]:
        # The SDK's stream() helper is itself the streaming call — the body is
        # passed through verbatim (stream() takes no `stream` kwarg) — and it
        # aggregates the final message (thinking signatures included) for us.
        body = dict(body)
        body.pop("stream", None)
        kwargs, extra = _split_body(body)
        async with self._client.messages.stream(**kwargs, extra_body=extra) as sdk_stream:
            async for event in sdk_stream:
                yield as_dict(event)
            final = await sdk_stream.get_final_message()

        payload = as_dict(final)
        yield {
            "__cliver_final__": {
                "message": {"role": "assistant", "content": payload.get("content") or []},
                "usage": payload.get("usage"),
            }
        }

    async def post(
        self,
        path: str,
        body: dict,
        *,
        files: dict | None = None,
        binary: bool = False,
    ) -> Any:
        options: dict[str, Any] = {}
        if files is not None:
            # The SDK only serializes `body` into multipart form fields (and
            # lets httpx generate the boundary) when this header is set;
            # without it, `body` would be dropped from the request.
            options["headers"] = {"Content-Type": "multipart/form-data"}
        raw = await self._client.post(path, cast_to=bytes, body=body or None, files=files, options=options)
        if binary:
            return raw
        return json.loads(raw) if raw else {}

    async def close(self) -> None:
        await self._client.close()
