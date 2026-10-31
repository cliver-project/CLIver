"""System One over the shared ``/v1/systemone`` HTTP wire.

Jev (TypeSafe), Aliyun Bailian's decision model, and a local ``laya-serve``
all serve one shape::

    POST {base_url}/v1/systemone
    {"model": ..., "state": ..., "questions": {...}}
    -> {"model": ..., "answers": {...}, "usage": {...}, "latency_ms": ...}

The body builder and answer parser are the shared ones from
:mod:`cliver_core.system_one`; this class only owns the transport call.
Bailian reaches this wire with ``base_url`` ending in ``/compatible-mode``
(the workspace id is part of the hostname), TypeSafe with
``https://api.typesafe.ai``.

A local ``laya`` backend does not speak HTTP — see
:class:`~cliver_core.local.providers.LocalSystemOneProvider`.
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Mapping

from cliver_core.errors import CapabilityNotSupported, ProviderError
from cliver_core.messages import CLIverMessageChunk
from cliver_core.provider import CLIverRequest, Provider
from cliver_core.system_one import Question, SystemOneResult, build_request, parse_result

__all__ = ["SystemOneProvider"]


class SystemOneProvider(Provider):
    """HTTP System One provider for Jev / Bailian / ``laya-serve``.

    Construction goes through ``create_provider(provider_class=...)`` from
    :meth:`cliver_core.bindings.ModelBinding.system_one`, which supplies the
    base URL and key.  ``timeout`` is the per-request default; a call-site
    ``timeout=`` option wins.

    The transport is plain ``httpx`` — the same ``_post_json_endpoint`` seam
    rerank and embeddings use — so the provider needs no brand SDK and no
    registry entry.  Non-2xx responses raise :class:`ProviderError` with the
    status-derived ``retryable`` flag; connection and timeout failures are
    classified retryable too, which lets ``AgentCore`` fail over to another
    ``system_one`` binding.
    """

    protocol = "system-one"
    brand = "system-one"

    DEFAULT_PATH = "/v1/systemone"
    DEFAULT_TIMEOUT = 10.0

    def __init__(
        self,
        api_key: str,
        base_url: str,
        user_agent: str | None = None,
        *,
        logger: logging.Logger | None = None,
        on_event: Any = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        super().__init__(api_key, base_url, logger=logger)
        self.user_agent = user_agent
        self.on_event = on_event
        self.timeout = timeout

    async def chat(self, request: CLIverRequest):  # pragma: no cover - not a chat model
        raise CapabilityNotSupported(f"{self.provider_name()} is a decision model; it cannot chat")

    async def stream(self, request: CLIverRequest) -> AsyncIterator[CLIverMessageChunk]:
        raise CapabilityNotSupported(f"{self.provider_name()} is a decision model; it cannot stream")
        if False:  # pragma: no cover - makes this an async generator, matching the ABC
            yield None

    async def system_one(
        self,
        *,
        state: Any,
        questions: Mapping[str, Question],
        model: str,
        **options: Any,
    ) -> SystemOneResult:
        """POST the typed questions and parse the structured answers.

        ``path`` and ``timeout`` are consumed here (a call-site ``path=``
        overrides :attr:`DEFAULT_PATH`; a call-site ``timeout=`` overrides
        the instance default); every other option rides the request body
        verbatim, so vendor-specific knobs pass through unchanged.
        """
        if not questions:
            raise ValueError("system one requires at least one question")
        path = options.pop("path", None) or self.DEFAULT_PATH
        timeout = options.pop("timeout", None) or self.timeout
        body = build_request(model=model, state=state, questions=questions)
        body.update(options)
        try:
            payload = await self._post_json_endpoint(
                path,
                body,
                label="system one",
                timeout=timeout,
            )
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(
                f"{self.provider_name()} system one request failed: {exc}",
                retryable=True,
            ) from exc
        return parse_result(payload)
