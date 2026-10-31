"""Generic Anthropic-compatible provider.

Owns the conversation-level Anthropic-protocol shape — system split,
tool_result merging, tool-name sanitization, param building — over a
:class:`~cliver_core.provider.transport.BrandSDK` transport: native message
dicts in, inference events around, CLIver objects out.  The transport is a
pure I/O pipe (the brand's official SDK when registered, else the builtin
Anthropic SDK adapter — see
:func:`cliver_core.provider.transports.resolve_brand_sdk`); provider and
transport exchange wire-format dicts, so exactly ONE conversion happens on
each side and the native message stays readable on the result
(``CLIverMessage.native`` — per-message shape lives on the message itself).
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, AsyncIterator
from uuid import uuid4

from cliver_core.errors import ProviderError
from cliver_core.events import EventHandler, InferenceEvent, InferenceEventType, dispatch_event
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

# Anthropic tool name constraint: ^[a-zA-Z0-9_-]{1,128}$
_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{1,128}$")


# ── Request shaping ────────────────────────────────────────────


def _split_system(messages: list[dict]) -> tuple[str | None, list[dict]]:
    """Pull system-role messages out — Anthropic takes ``system`` separately.

    Content may be a plain string or a list of text blocks — messages are
    built protocol-agnostically and converted per protocol upstream, so
    both shapes reach the provider.
    """
    system_parts = []
    conversation = []
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "system":
            text = m.get("content", "")
            if isinstance(text, list):
                text = "\n".join(b.get("text") or "" for b in text if isinstance(b, dict) and b.get("type") == "text")
            if isinstance(text, str) and text:
                system_parts.append(text)
        else:
            conversation.append(m)
    system = "\n\n".join(system_parts) if system_parts else None
    return system, conversation


def _merge_tool_results(conv_messages: list[dict]) -> list[dict]:
    """Merge consecutive tool_result-only user messages into one.

    Anthropic requires all ``tool_result`` blocks for a given assistant
    message to live in a single subsequent user message.  CLIver's
    Re-Act loop emits one message per tool call, so we merge them here.
    """
    merged: list[dict] = []
    for m in conv_messages:
        if not isinstance(m, dict) or m.get("role") != "user":
            merged.append(m)
            continue
        content = m.get("content")
        if not isinstance(content, list) or not content:
            merged.append(m)
            continue
        is_all_tool_results = all(isinstance(b, dict) and b.get("type") == "tool_result" for b in content)
        if is_all_tool_results and merged and merged[-1].get("role") == "user":
            prev_content = merged[-1].get("content")
            if isinstance(prev_content, list) and any(
                isinstance(b, dict) and b.get("type") == "tool_result" for b in prev_content
            ):
                prev_content.extend(content)
                continue
        merged.append(m)
    return merged


def _sanitize_tool_names(tools: list[dict]) -> tuple[dict[str, str], dict[str, str]]:
    """Ensure tool names match Anthropic's ^[a-zA-Z0-9_-]{1,128}$ constraint.

    Operates on native tool dicts (each has a ``name`` key).

    Returns (forward_map, reverse_map):
        forward_map: original_name → sanitized_name  (for request building)
        reverse_map: sanitized_name → original_name  (for response restoration)
    """
    forward: dict[str, str] = {}
    reverse: dict[str, str] = {}
    seen: set[str] = set()

    for tool in tools:
        name = tool.get("name") or ""
        if _NAME_RE.match(name) and name not in seen:
            forward[name] = name
            reverse[name] = name
            seen.add(name)
            continue

        sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", name)[:128]
        if not sanitized:
            sanitized = "tool"

        base = sanitized
        counter = 1
        while sanitized in seen:
            suffix = f"_{counter}"
            sanitized = base[: 128 - len(suffix)] + suffix
            counter += 1

        forward[name] = sanitized
        reverse[sanitized] = name
        seen.add(sanitized)

    return forward, reverse


def _restore_tool_use_names(message: dict[str, Any], reverse: dict[str, str]) -> dict[str, Any]:
    """Restore original tool names in a response message's ``tool_use`` blocks.

    The API echoes the sanitized names it received; the message built from
    its payload must expose the original names — tool dispatch and
    diagnostics use them, and the name is what the caller declared.  This is
    the single restore point, so ``native`` and the derived fields agree in
    every path (chat and stream); wire-validity is re-established when the
    next request is built (``_sanitize_conversation_tool_uses``).  Builds
    new dicts; never mutates the payload in place.
    """
    if not reverse:
        return message
    restored: list[Any] = []
    for block in message.get("content") or []:
        if isinstance(block, dict) and block.get("type") == "tool_use":
            original = reverse.get(block.get("name") or "")
            if original and original != block.get("name"):
                block = {**block, "name": original}
        restored.append(block)
    return {**message, "content": restored}


def _sanitize_conversation_tool_uses(messages: list[dict], forward: dict[str, str]) -> list[dict]:
    """Sanitize ``tool_use`` block names inside echoed assistant messages.

    Tool *definitions* are sanitized by ``_sanitize_tool_names``; assistant
    messages echoed back carry the caller's original names (``my.tool``) —
    response messages are restored to originals on the way in, in every
    path — and names outside Anthropic's ``^[a-zA-Z0-9_-]{1,128}$``
    constraint are rejected by the API.  Same forward map, so names stay
    consistent with the definitions.  Builds new message/block dicts;
    never mutates ``CLIverMessage.native`` in place.
    """
    sanitized: list[dict] = []
    for m in messages:
        content = m.get("content") if isinstance(m, dict) else None
        if not isinstance(content, list) or m.get("role") != "assistant":
            sanitized.append(m)
            continue
        changed = False
        blocks = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                name = block.get("name") or ""
                new_name = forward.get(name)
                if new_name is None and not _NAME_RE.match(name):
                    new_name = re.sub(r"[^a-zA-Z0-9_-]", "_", name)[:128] or "tool"
                if new_name and new_name != name:
                    block = {**block, "name": new_name}
                    changed = True
            blocks.append(block)
        sanitized.append({**m, "content": blocks} if changed else m)
    return sanitized


def _build_params(options: dict[str, Any]) -> dict[str, Any]:
    """Extract known Anthropic params. Does NOT mutate the input dict."""
    known_keys = {
        "max_tokens",
        "max_completion_tokens",
        "temperature",
        "top_p",
        "top_k",
        "thinking",
    }
    max_tokens = options.get("max_tokens", options.get("max_completion_tokens", 4096))
    params: dict[str, Any] = {"max_tokens": max_tokens}
    for key in ("temperature", "top_p", "top_k", "thinking"):
        if key in options:
            params[key] = options[key]
    # Pass through anything else
    params.update({k: v for k, v in options.items() if k not in known_keys})
    return params


class AnthropicProvider(Provider):
    """Anthropic-native messages API provider."""

    protocol = "anthropic"
    brand = "anthropic"
    default_base_url = "https://api.anthropic.com"
    supports_client_options = True

    def __init__(
        self,
        api_key: str,
        base_url: str,
        user_agent: str | None = None,
        *,
        logger: "logging.Logger | None" = None,
        on_event: EventHandler | None = None,
        transport: BrandSDK | None = None,
        client_timeout: float | None = None,
        client_max_retries: int | None = None,
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
            use_bearer_auth=getattr(self.__class__, "_anthropic_use_bearer_auth", False),
            timeout=client_timeout,
            max_retries=client_max_retries,
        )

    async def close(self) -> None:
        """Close the underlying transport."""
        await self.transport.close()

    # ── Conversion ──────────────────────────────────────────

    def tool_to_native(self, tool: CLIverTool) -> dict:
        return {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.parameters,
        }

    def _prepare_request(
        self,
        model: str,
        messages: list[dict],
        tools: list[dict] | None,
        options: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, str]]:
        """Build the messages-API request body + the tool-name reverse map."""
        system, conversation = _split_system(messages)
        conversation = _merge_tool_results(conversation)

        reverse: dict[str, str] = {}
        native_tools: list[dict] | None = None
        if tools:
            forward, reverse = _sanitize_tool_names(tools)
            native_tools = []
            for tool in tools:
                native = dict(tool)
                native["name"] = forward.get(native.get("name"), native.get("name"))
                native_tools.append(native)
            # echoed assistant tool_use blocks must satisfy the same name
            # constraint as the definitions (passthrough natives and
            # stream-rebuilt messages may carry original names)
            conversation = _sanitize_conversation_tool_uses(conversation, forward)

        body = _build_params(options)
        body.update(model=model, messages=conversation)
        if native_tools:
            body["tools"] = native_tools
        if system:
            body["system"] = system
        return body, reverse

    def _response_from_native(self, data: dict[str, Any], reverse: dict[str, str]) -> CLIverResponse:
        """Convert one ``message`` wire payload — the single response-side hop."""
        # The native message is role + content blocks; usage/id live on the
        # response, not the message.  Copy the block list so the message's
        # native detaches from the response payload, and restore original
        # tool names so native and the derived fields agree (the request
        # path re-sanitizes on echo).
        message = CLIverMessage.from_native(
            _restore_tool_use_names(
                {"role": "assistant", "content": list(data.get("content") or [])},
                reverse,
            ),
            "anthropic",
        )
        return CLIverResponse(message=message, usage=self._usage_from_raw(data.get("usage")))

    @staticmethod
    def _usage_from_raw(raw_usage: dict[str, Any] | None) -> UsageInfo | None:
        if not raw_usage:
            return None
        return UsageInfo(
            input_tokens=raw_usage.get("input_tokens") or 0,
            output_tokens=raw_usage.get("output_tokens") or 0,
            cache_read_input_tokens=raw_usage.get("cache_read_input_tokens"),
        )

    @staticmethod
    def _chunk_from_event(data: dict[str, Any], reverse: dict[str, str]) -> CLIverMessageChunk | None:
        """Convert one stream event — or ``None`` when there is nothing to emit."""
        event_type = data.get("type")

        if event_type == "content_block_delta":
            delta = data.get("delta") or {}
            delta_type = delta.get("type")
            if delta_type == "text_delta":
                return CLIverMessageChunk(content=delta.get("text"))
            if delta_type == "thinking_delta":
                vendor_ext: dict[str, str] = {}
                thinking = delta.get("thinking")
                if thinking:
                    vendor_ext["thinking"] = thinking
                return CLIverMessageChunk(vendor_ext=vendor_ext)
            if delta_type == "signature_delta":
                # The thinking-block signature is not a delta — the SDK's
                # aggregator folds it into the terminal message.
                return None
            if delta_type == "input_json_delta":
                return CLIverMessageChunk(
                    tool_call_chunks=[
                        ToolCallChunk(
                            index=data.get("index") or 0,
                            args_delta=delta.get("partial_json"),
                        )
                    ]
                )

        elif event_type == "content_block_start":
            block = data.get("content_block") or {}
            if block.get("type") == "tool_use":
                name = block.get("name")
                if name and name in reverse:
                    name = reverse[name]
                return CLIverMessageChunk(
                    tool_call_chunks=[
                        ToolCallChunk(
                            index=data.get("index") or 0,
                            id=block.get("id"),
                            name=name,
                        )
                    ]
                )
            if block.get("type") == "thinking":
                vendor_ext = {}
                thinking = block.get("thinking") or ""
                if thinking:
                    vendor_ext["thinking"] = thinking
                return CLIverMessageChunk(vendor_ext=vendor_ext)

        return None

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
                run_id=request.run_id,
                data={"message_count": len(messages), "tool_count": len(tools)},
            )
        )

        native_tools = [self.tool_to_native(t) for t in tools] if tools else None
        body, reverse = self._prepare_request(request.model, messages, native_tools, options)
        try:
            data = await self._transport_chat(body)
        except ProviderError as exc:
            await self._emit(
                InferenceEvent(
                    event=InferenceEventType.ERROR,
                    model=request.model,
                    provider=self.protocol,
                    request_id=request_id,
                    run_id=request.run_id,
                    data={"error": str(exc), "retryable": exc.retryable},
                )
            )
            raise

        usage = self._usage_from_raw(data.get("usage"))
        await self._emit(
            InferenceEvent(
                event=InferenceEventType.COMPLETED,
                model=request.model,
                provider=self.protocol,
                request_id=request_id,
                run_id=request.run_id,
                data={
                    "total_latency_ms": (time.monotonic() - start) * 1000,
                    "usage": usage.model_dump() if usage else None,
                },
            )
        )

        # Attach request diagnostic data for session turn inspection
        # (AgentCore collects it into the Re-Act trace).
        response = self._response_from_native(data, reverse)
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
            "tools": [t.model_dump(exclude={"execute", "execute_async"}) for t in tools],
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
                run_id=request.run_id,
                data={"message_count": len(messages), "tool_count": len(tools)},
            )
        )

        native_tools = [self.tool_to_native(t) for t in tools] if tools else None
        body, reverse = self._prepare_request(request.model, messages, native_tools, options)
        saw_final = False
        completed_usage: UsageInfo | None = None
        try:
            async for data in self._transport_stream(body):
                if "__cliver_final__" in data:
                    # Terminal event — the SDK-aggregated native assistant
                    # message.  Consumed before protocol mapping; it never
                    # reaches _chunk_from_event.
                    final = data["__cliver_final__"]
                    saw_final = True
                    completed_usage = self._usage_from_raw(final.get("usage"))
                    yield self.on_chunk(
                        CLIverMessageChunk(
                            native=_restore_tool_use_names(final["message"], reverse),
                            usage=completed_usage,
                        )
                    )
                    continue
                chunk = self._chunk_from_event(data, reverse)
                if chunk is None:
                    continue  # bookkeeping event (message_start, ping, ...)

                if not first_token_emitted and (
                    chunk.content or chunk.vendor_ext.get("thinking") or chunk.tool_call_chunks
                ):
                    first_token_emitted = True
                    await self._emit(
                        InferenceEvent(
                            event=InferenceEventType.FIRST_TOKEN,
                            model=request.model,
                            provider=self.protocol,
                            request_id=request_id,
                            run_id=request.run_id,
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
                                run_id=request.run_id,
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
            data: dict[str, Any] = {"error": str(e)}
            if isinstance(e, ProviderError):
                data["retryable"] = e.retryable
            await self._emit(
                InferenceEvent(
                    event=InferenceEventType.ERROR,
                    model=request.model,
                    provider=self.protocol,
                    request_id=request_id,
                    run_id=request.run_id,
                    data=data,
                )
            )
            raise

        await self._emit(
            InferenceEvent(
                event=InferenceEventType.COMPLETED,
                model=request.model,
                provider=self.protocol,
                request_id=request_id,
                run_id=request.run_id,
                data={
                    "total_latency_ms": (time.monotonic() - start) * 1000,
                    "usage": completed_usage.model_dump() if completed_usage else None,
                },
            )
        )

    async def _emit(self, event: InferenceEvent) -> None:
        await dispatch_event(self.on_event, event)
