"""OpenAI-compatible chat/completions protocol engine."""

from __future__ import annotations

import json
import logging
import time
from typing import Any
from uuid import uuid4

from openai import NOT_GIVEN, AsyncOpenAI

from cliver.events import EventHandler, InferenceEvent, InferenceEventType
from cliver.messages import (
    CLIverMessage,
    CLIverMessageChunk,
    ToolCall,
    ToolCallChunk,
    UsageInfo,
)
from cliver.provider import CLIverResponse
from cliver.provider.engine import ProtocolEngine
from cliver.tool import CLIverTool

logger = logging.getLogger(__name__)


class OpenAIEngine(ProtocolEngine):
    """OpenAI-compatible chat/completions protocol."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        on_event: EventHandler | None = None,
    ):
        super().__init__(api_key, base_url, on_event)
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    # ── Conversion ──────────────────────────────────────────

    def msg_to_native(self, msg: CLIverMessage) -> dict:
        native: dict[str, Any] = {"role": msg.role, "content": msg.content}

        if msg.tool_calls:
            native["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.args, ensure_ascii=False),
                    },
                }
                for tc in msg.tool_calls
            ]
        if msg.tool_call_id:
            native["tool_call_id"] = msg.tool_call_id

        return native

    def tool_to_native(self, tool: CLIverTool) -> dict:
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }

    def extract_cliver_message(self, choice) -> CLIverMessage:
        m = choice.message
        vendor_ext: dict[str, Any] = {}

        rc = getattr(m, "reasoning_content", None)
        if rc:
            vendor_ext["reasoning_content"] = rc

        content = m.content
        tool_calls = None
        if m.tool_calls:
            tool_calls = [
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    args=json.loads(tc.function.arguments),
                )
                for tc in m.tool_calls
            ]
            if not content:
                content = None

        return CLIverMessage(
            role="assistant",
            content=content,
            tool_calls=tool_calls,
            vendor_ext=vendor_ext,
        )

    def extract_chunk(self, delta) -> CLIverMessageChunk:
        chunk = CLIverMessageChunk(
            content=delta.content,
            vendor_ext={},
        )

        rc = getattr(delta, "reasoning_content", None)
        if rc:
            chunk.vendor_ext["reasoning_content"] = rc

        if delta.tool_calls:
            tc_chunks = []
            for tc in delta.tool_calls:
                tc_chunk = ToolCallChunk(index=tc.index or 0)
                if tc.id:
                    tc_chunk.id = tc.id
                if tc.function and tc.function.name:
                    tc_chunk.name = tc.function.name
                if tc.function and tc.function.arguments:
                    tc_chunk.args_delta = tc.function.arguments
                tc_chunks.append(tc_chunk)
            chunk.tool_call_chunks = tc_chunks

        return chunk

    # ── API Calls ───────────────────────────────────────────

    async def chat(self, messages, tools, model, options):
        request_id = str(uuid4())
        start = time.monotonic()
        await self._emit(
            InferenceEvent(
                event=InferenceEventType.STARTED,
                model=model,
                provider="openai",
                request_id=request_id,
                data={"message_count": len(messages), "tool_count": len(tools)},
            )
        )

        try:
            native_tools = [self.tool_to_native(t) for t in tools] if tools else NOT_GIVEN
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                tools=native_tools,
                **options,
            )
        except Exception as e:
            await self._emit(
                InferenceEvent(
                    event=InferenceEventType.ERROR,
                    model=model,
                    provider="openai",
                    request_id=request_id,
                    data={"error": str(e)},
                )
            )
            raise

        choice = response.choices[0]
        usage = None
        if response.usage:
            usage = UsageInfo(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            )

        await self._emit(
            InferenceEvent(
                event=InferenceEventType.COMPLETED,
                model=model,
                provider="openai",
                request_id=request_id,
                data={"total_latency_ms": (time.monotonic() - start) * 1000},
            )
        )

        return CLIverResponse(
            message=self.extract_cliver_message(choice),
            usage=usage,
        )

    async def stream(self, messages, tools, model, options):
        request_id = str(uuid4())
        start = time.monotonic()
        first_token_emitted = False

        await self._emit(
            InferenceEvent(
                event=InferenceEventType.STARTED,
                model=model,
                provider="openai",
                request_id=request_id,
                data={"message_count": len(messages), "tool_count": len(tools)},
            )
        )

        try:
            native_tools = [self.tool_to_native(t) for t in tools] if tools else NOT_GIVEN
            stream = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                tools=native_tools,
                stream=True,
                stream_options={"include_usage": True},
                **options,
            )
        except Exception as e:
            await self._emit(
                InferenceEvent(
                    event=InferenceEventType.ERROR,
                    model=model,
                    provider="openai",
                    request_id=request_id,
                    data={"error": str(e)},
                )
            )
            raise

        async for sdk_chunk in stream:
            if not sdk_chunk.choices:
                continue

            delta = sdk_chunk.choices[0].delta

            if not first_token_emitted and (delta.content or (delta.tool_calls and delta.tool_calls[0].function)):
                first_token_emitted = True
                await self._emit(
                    InferenceEvent(
                        event=InferenceEventType.FIRST_TOKEN,
                        model=model,
                        provider="openai",
                        request_id=request_id,
                        data={"latency_ms": (time.monotonic() - start) * 1000},
                    )
                )

            chunk = self.extract_chunk(delta)

            if delta.tool_calls:
                for tc in delta.tool_calls:
                    if tc.id:
                        await self._emit(
                            InferenceEvent(
                                event=InferenceEventType.TOOL_CALL,
                                model=model,
                                provider="openai",
                                request_id=request_id,
                                data={
                                    "tool_name": tc.function.name if tc.function else "?",
                                    "tool_call_id": tc.id,
                                },
                            )
                        )

            yield chunk

        await self._emit(
            InferenceEvent(
                event=InferenceEventType.COMPLETED,
                model=model,
                provider="openai",
                request_id=request_id,
                data={"total_latency_ms": (time.monotonic() - start) * 1000},
            )
        )

    async def _emit(self, event: InferenceEvent) -> None:
        if self.on_event:
            await self.on_event(event)
