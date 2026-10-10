"""Generic OpenAI-compatible provider.

Owns the conversation-level OpenAI-protocol shape — request assembly and
response extraction — over a :class:`~cliver_core.provider.transport.BrandSDK`
transport: native message dicts in, inference events around, CLIver objects
out.  The transport is a pure I/O pipe (the brand's official SDK when
registered, else the builtin OpenAI SDK adapter — see
:func:`cliver_core.provider.transports.resolve_brand_sdk`); provider and
transport exchange wire-format dicts, so exactly ONE conversion happens on
each side and the native message stays readable on the result
(``CLIverMessage.native`` — per-message shape lives on the message itself).

Media generation rides the base :class:`~cliver_core.provider.Provider`
pipeline (shared path resolution, transport I/O, item wrapping, file saving);
this class contributes only the OpenAI wire shape — request field names and
which purposes return raw bytes.
"""

from __future__ import annotations

import logging
import time
from typing import Any, AsyncIterator
from uuid import uuid4

from cliver_core.events import EventHandler, InferenceEvent, InferenceEventType
from cliver_core.media import MediaContent, MediaType
from cliver_core.messages import (
    CLIverMessage,
    CLIverMessageChunk,
    ToolCallChunk,
    UsageInfo,
)
from cliver_core.provider import CLIverRequest, CLIverResponse, Provider
from cliver_core.provider.transport import BrandSDK
from cliver_core.provider.transports import resolve_brand_sdk
from cliver_core.tool import CLIverTool

logger = logging.getLogger(__name__)


class OpenAIProvider(Provider):
    """OpenAI-compatible chat/completions provider.

    The generic provider for the ``openai`` protocol — every
    OpenAI-compatible endpoint (DeepSeek, GLM, Ollama, ...) is consumed
    through a subclass, or through this class with a custom ``base_url``.
    """

    protocol = "openai"
    brand = "openai"
    default_base_url = "https://api.openai.com/v1"

    def __init__(
        self,
        api_key: str,
        base_url: str,
        user_agent: str | None = None,
        *,
        logger: "logging.Logger | None" = None,
        on_event: EventHandler | None = None,
        transport: BrandSDK | None = None,
    ):
        super().__init__(api_key, base_url, logger=logger)
        self.user_agent = user_agent
        self.on_event = on_event
        # The brand selects the transport: a ``cliver.provider_sdks`` adapter
        # for this brand if registered, else the protocol's builtin SDK
        # adapter pointed at this base_url.  Tests inject a transport.
        self.transport: BrandSDK = transport or resolve_brand_sdk(
            self.protocol,
            api_key=api_key,
            base_url=base_url,
            brand=self.provider_name(),
            user_agent=user_agent,
        )

    async def close(self) -> None:
        """Close the underlying transport."""
        await self.transport.close()

    # ── Conversion ──────────────────────────────────────────

    def tool_to_native(self, tool: CLIverTool) -> dict:
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }

    def _build_body(
        self,
        messages: list[dict],
        native_tools: list[dict] | None,
        model: str,
        options: dict[str, Any],
    ) -> dict[str, Any]:
        """Assemble the chat/completions request body (wire format)."""
        body: dict[str, Any] = {"model": model, "messages": messages, **options}
        if native_tools:
            body["tools"] = native_tools
        return body

    def _response_from_native(self, data: dict[str, Any]) -> CLIverResponse:
        """Convert one ``chat.completion`` wire payload — the single response-side hop."""
        choices = data.get("choices") or []
        message = (choices[0].get("message") if choices else None) or {}

        # Shallow copy: the message dict detaches from the response payload;
        # vendor extras (reasoning_content, ...) stay in native.
        return CLIverResponse(
            message=CLIverMessage.from_native(dict(message), "openai"),
            usage=self._usage_from_raw(data.get("usage")),
        )

    @staticmethod
    def _usage_from_raw(raw_usage: dict[str, Any] | None) -> UsageInfo | None:
        if not raw_usage:
            return None
        return UsageInfo(
            input_tokens=raw_usage.get("prompt_tokens") or 0,
            output_tokens=raw_usage.get("completion_tokens") or 0,
        )

    @staticmethod
    def _chunk_from_native(delta: dict[str, Any]) -> CLIverMessageChunk:
        """Convert one streamed ``choices[0].delta`` — the single response-side hop."""
        vendor_ext: dict[str, str] = {}
        reasoning = delta.get("reasoning_content")
        if reasoning:
            vendor_ext["reasoning_content"] = reasoning

        tool_call_chunks: list[ToolCallChunk] | None = None
        raw_calls = delta.get("tool_calls")
        if raw_calls:
            tool_call_chunks = []
            for raw in raw_calls:
                tc = ToolCallChunk(index=raw.get("index") or 0)
                if raw.get("id"):
                    tc.id = raw["id"]
                function = raw.get("function") or {}
                if function.get("name"):
                    tc.name = function["name"]
                if function.get("arguments"):
                    tc.args_delta = function["arguments"]
                tool_call_chunks.append(tc)

        return CLIverMessageChunk(
            content=delta.get("content"),
            tool_call_chunks=tool_call_chunks,
            vendor_ext=vendor_ext,
        )

    # ── API Calls ───────────────────────────────────────────

    async def chat(self, request: CLIverRequest) -> CLIverResponse:
        messages = [m.to_native(self.protocol) for m in request.messages]
        options = self.filter_options(request.options)
        tools = request.tools or []

        request_id = str(uuid4())
        start = time.monotonic()
        await self._emit(
            InferenceEvent(
                event=InferenceEventType.STARTED,
                model=request.model,
                provider=self.protocol,
                request_id=request_id,
                data={"message_count": len(messages), "tool_count": len(tools)},
            )
        )

        native_tools = [self.tool_to_native(t) for t in tools] if tools else None
        body = self._build_body(messages, native_tools, request.model, options)
        try:
            data = await self.transport.chat(body)
        except Exception as e:
            await self._emit(
                InferenceEvent(
                    event=InferenceEventType.ERROR,
                    model=request.model,
                    provider=self.protocol,
                    request_id=request_id,
                    data={"error": str(e)},
                )
            )
            raise

        await self._emit(
            InferenceEvent(
                event=InferenceEventType.COMPLETED,
                model=request.model,
                provider=self.protocol,
                request_id=request_id,
                data={"total_latency_ms": (time.monotonic() - start) * 1000},
            )
        )

        # Attach request diagnostic data for session turn inspection
        # (AgentCore collects it into the Re-Act trace).
        response = self._response_from_native(data)
        response.request_info = {
            "model": request.model,
            "provider": self.provider_name(),
            "options": options,
            "system_prompt": next(
                (m.content for m in request.messages if m.role == "system"),
                None,
            ),
            "messages": [
                m.model_dump(exclude={"native", "native_protocol", "trace"}, exclude_none=True)
                for m in request.messages
            ],
            "tools": [t.model_dump(exclude={"execute"}) for t in tools],
        }

        return self.on_response(response)

    async def stream(self, request: CLIverRequest) -> AsyncIterator[CLIverMessageChunk]:
        messages = [m.to_native(self.protocol) for m in request.messages]
        options = self.filter_options(request.options)
        tools = request.tools or []

        request_id = str(uuid4())
        start = time.monotonic()
        first_token_emitted = False

        await self._emit(
            InferenceEvent(
                event=InferenceEventType.STARTED,
                model=request.model,
                provider=self.protocol,
                request_id=request_id,
                data={"message_count": len(messages), "tool_count": len(tools)},
            )
        )

        native_tools = [self.tool_to_native(t) for t in tools] if tools else None
        body = self._build_body(messages, native_tools, request.model, options)
        saw_final = False
        try:
            async for data in self.transport.stream(body):
                if "__cliver_final__" in data:
                    # Terminal event — the SDK-aggregated native assistant
                    # message.  Consumed before protocol mapping; it never
                    # reaches _chunk_from_native.
                    final = data["__cliver_final__"]
                    saw_final = True
                    yield self.on_chunk(
                        CLIverMessageChunk(
                            native=final["message"],
                            usage=self._usage_from_raw(final.get("usage")),
                        )
                    )
                    continue
                choices = data.get("choices") or []
                if not choices:
                    continue  # usage-only final payload carries no delta
                chunk = self._chunk_from_native(choices[0].get("delta") or {})

                if not first_token_emitted and (
                    chunk.content or chunk.vendor_ext.get("reasoning_content") or chunk.tool_call_chunks
                ):
                    first_token_emitted = True
                    await self._emit(
                        InferenceEvent(
                            event=InferenceEventType.FIRST_TOKEN,
                            model=request.model,
                            provider=self.protocol,
                            request_id=request_id,
                            data={"latency_ms": (time.monotonic() - start) * 1000},
                        )
                    )

                for tc in chunk.tool_call_chunks or []:
                    if tc.id:
                        await self._emit(
                            InferenceEvent(
                                event=InferenceEventType.TOOL_CALL,
                                model=request.model,
                                provider=self.protocol,
                                request_id=request_id,
                                data={"tool_name": tc.name or "?", "tool_call_id": tc.id},
                            )
                        )

                yield self.on_chunk(chunk)

            if not saw_final:
                # Adapter bug — the stream contract requires the terminal
                # event; fail fast instead of returning a partial message.
                raise RuntimeError(
                    f"{getattr(self.transport, 'label', 'transport')} stream ended "
                    "without the terminal __cliver_final__ event"
                )
        except Exception as e:
            await self._emit(
                InferenceEvent(
                    event=InferenceEventType.ERROR,
                    model=request.model,
                    provider=self.protocol,
                    request_id=request_id,
                    data={"error": str(e)},
                )
            )
            raise

        await self._emit(
            InferenceEvent(
                event=InferenceEventType.COMPLETED,
                model=request.model,
                provider=self.protocol,
                request_id=request_id,
                data={"total_latency_ms": (time.monotonic() - start) * 1000},
            )
        )

    async def _emit(self, event: InferenceEvent) -> None:
        if self.on_event:
            await self.on_event(event)

    # ── Media generation — the OpenAI wire shape on the shared pipeline ──
    #
    # Path resolution, transport I/O, and file saving live on the base
    # ``Provider`` media pipeline; this class declares only the OpenAI
    # fashion: per-purpose request field names and which purposes return
    # raw bytes.

    #: ``/audio/speech`` returns raw audio bytes, not a JSON envelope.
    _BINARY_MEDIA_PURPOSES = frozenset({"tts"})

    def _build_media_gen_body(
        self,
        prompt: str,
        model: str,
        purpose: str,
        *,
        media: list[MediaContent] | None = None,
        **options,
    ) -> dict[str, Any]:
        """Assemble the OpenAI-fashion request body for one media purpose.

        ``image`` carries a reference image (editing / variation) as a data
        URL; ``tts`` speaks ``input`` / ``voice`` / ``response_format``;
        everything else is the generic ``{"model", "prompt", ...options}``.
        """
        if purpose == "image":
            body: dict[str, Any] = {"model": model, "prompt": prompt, **options}
            reference = next((m for m in media or [] if getattr(m, "type", None) == MediaType.IMAGE), None)
            if reference is not None:
                image = reference.data
                if not image.startswith("data:"):
                    image = f"data:{reference.mime_type};base64,{image}"
                body["image"] = image
            return body
        if purpose == "tts":
            return {
                "model": model,
                "input": prompt,
                "voice": options.pop("voice", "alloy"),
                "response_format": options.pop("response_format", "mp3"),
                **options,
            }
        return {"model": model, "prompt": prompt, **options}

    def _extract_media_items(self, data: dict, purpose: str) -> list[dict]:
        """OpenAI-fashion items; video responses may name the URL ``video_url``."""
        items = super()._extract_media_items(data, purpose)
        if purpose != "video":
            return items
        return [
            item if item.get("url") or not item.get("video_url") else {**item, "url": item["video_url"]}
            for item in items
        ]
