"""Chat SSE route for the admin portal."""

from __future__ import annotations

import json
import logging
from typing import Callable

from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

logger = logging.getLogger(__name__)


def get_chat_routes(context: dict, require_auth: Callable) -> list:
    """Return chat API route (SSE streaming)."""

    @require_auth
    async def handle_chat(request: Request):
        gateway = context.get("gateway")
        if not gateway or not getattr(gateway, "_agent_core", None):
            return JSONResponse({"error": "Agent not available"}, status_code=503)

        session_manager = context.get("cli_session_manager")

        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

        prompt = (body.get("message") or body.get("prompt", "")).strip()
        if not prompt:
            return JSONResponse({"error": "'message' (or 'prompt') is required"}, status_code=400)

        executor = gateway._agent_core
        model = body.get("model") or executor.default_model
        system_message = body.get("system_message")
        raw_history = body.get("conversation_history") or []
        tool_names = body.get("filter_tools")
        save_media_dir = body.get("save_media_dir")
        conversation_id = body.get("session_id") or body.get("conversation_id")

        # -- Conversation persistence --
        session_id = None
        if conversation_id and session_manager:
            existing = session_manager.get_session_info(conversation_id)
            if not existing:
                return JSONResponse({"error": "Conversation not found"}, status_code=404)
            session_id = conversation_id
            # Load persisted turns as history if client didn't pass it explicitly
            if not raw_history:
                loaded = session_manager.load_turns(session_id)
                raw_history = [{"role": t["role"], "content": t["content"]} for t in loaded]
        elif session_manager:
            session_id = session_manager.create_session()

        if session_manager and session_id:
            try:
                session_manager.append_turn(session_id, "user", prompt)
            except Exception as e:
                logger.warning("Failed to persist user turn: %s", e)

        conversation_history = None
        if raw_history:
            from langchain_core.messages import AIMessage, HumanMessage

            conversation_history = []
            for msg in raw_history:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "user":
                    conversation_history.append(HumanMessage(content=content))
                elif role == "assistant":
                    extra = {}
                    if "reasoning_content" in msg:
                        extra["reasoning_content"] = msg["reasoning_content"]
                    conversation_history.append(AIMessage(content=content, additional_kwargs=extra))

        def _system_appender():
            parts = []
            if system_message:
                parts.append(system_message)
            parts.append(
                "\n## Server Mode\n\n"
                "You are running as a backend API service via the admin portal. "
                "Do NOT use Ask — there is no human to respond. "
                "Make autonomous decisions. Be concise and direct. "
                "After a tool call completes, summarize the result briefly — "
                "do NOT repeat the tool input or arguments you already provided."
            )
            return "\n".join(parts)

        _tool_filter = None
        if tool_names:
            allowed = set(tool_names)

            async def _tool_filter(user_input, tools):
                return [t for t in tools if t.name in allowed]

        from cliver.permissions import PermissionMode

        if executor.permission_manager:
            executor.permission_manager.set_mode(PermissionMode.YOLO)
        executor.on_permission_prompt = lambda tool, args: "allow"

        uses_thinking = False
        try:
            from cliver.model_capabilities import ModelCapability

            effective_model = model or executor.default_model
            mc = executor._get_llm_model(effective_model) if effective_model else None
            if mc and ModelCapability.THINK_MODE in mc.get_capabilities():
                uses_thinking = True
        except Exception:
            pass

        async def generate():
            full_text = ""
            media_files = []
            stream_media = []
            try:
                async for chunk in executor._stream_user_input_async(
                    user_input=prompt,
                    model=model,
                    system_message_appender=_system_appender,
                    filter_tools=_tool_filter,
                    conversation_history=conversation_history,
                    outputs_dir=save_media_dir,
                ):
                    if hasattr(chunk, "content") and chunk.content:
                        text = str(chunk.content)
                        full_text += text
                        data = json.dumps({"type": "content", "content": text})
                        yield f"data: {data}\n\n".encode()
                    chunk_kwargs = getattr(chunk, "additional_kwargs", None) or {}
                    if "media_content" in chunk_kwargs:
                        stream_media.extend(chunk_kwargs["media_content"])

                if save_media_dir and stream_media:
                    try:
                        from cliver.media_handler import MultimediaResponse, MultimediaResponseHandler

                        handler = MultimediaResponseHandler(save_directory=save_media_dir)
                        multimedia = MultimediaResponse(media_content=stream_media)
                        media_files = handler.save_media_content(multimedia, prefix="chat")
                    except Exception as e:
                        logger.warning("Failed to save media: %s", e)

                # Persist assistant turn
                if session_manager and session_id:
                    try:
                        session_manager.append_turn(session_id, "assistant", full_text)
                    except Exception as e:
                        logger.warning("Failed to persist assistant turn: %s", e)

                done_data = {
                    "type": "done",
                    "content": full_text,
                    "media": media_files,
                    "media_files": media_files,
                    "session_id": session_id,
                }
                if uses_thinking:
                    done_data["reasoning_content"] = ""
                yield f"data: {json.dumps(done_data)}\n\n".encode()

            except Exception as e:
                logger.error("Chat streaming error: %s", e)
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n".encode()

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    return [Route("/admin/api/chat", handle_chat, methods=["POST"])]
