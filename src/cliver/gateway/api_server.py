"""
OpenAI-compatible API server for CLIver.

Exposes CLIver's agent capabilities via standard OpenAI API endpoints.
Stateless — each request runs a full Re-Act loop and returns the final answer.

Usage:
    from cliver.gateway.api_server import get_api_routes
    routes = get_api_routes(executor, get_status, api_key="secret")
"""

import json
import logging
import time
import uuid
from typing import Callable, Optional

from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level helpers (no per-instance state needed)
# ---------------------------------------------------------------------------


def _server_system_appender():
    return (
        "\n## Server Mode\n\n"
        "You are running as a backend API service. "
        "Do NOT use Ask — there is no human to respond. "
        "Make autonomous decisions. Be concise and direct."
    )


def _configure_server_mode(executor):
    """Set up executor for headless server operation."""
    if executor.permission_manager:
        from cliver.permissions import PermissionMode

        executor.permission_manager.set_mode(PermissionMode.YOLO)

    # Auto-allow any permission prompts (no human to respond)
    executor.on_permission_prompt = lambda tool, args: "allow"


def _parse_chat_request(body: dict, executor) -> dict:
    """Parse an OpenAI chat completion request body."""
    messages = body.get("messages")
    if not messages or not isinstance(messages, list):
        raise ValueError("'messages' field is required and must be a list")

    system_message = None
    conversation_history = []
    user_input = ""

    from langchain_core.messages import AIMessage, HumanMessage

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "system":
            system_message = content
        elif role == "user":
            if user_input:
                conversation_history.append(HumanMessage(content=user_input))
            user_input = content
        elif role == "assistant":
            if content:
                conversation_history.append(AIMessage(content=content))

    if not user_input:
        raise ValueError("No user message found in messages")

    options = {}
    for key in ("temperature", "max_tokens", "top_p", "frequency_penalty", "presence_penalty"):
        if key in body:
            options[key] = body[key]

    return {
        "user_input": user_input,
        "model": body.get("model") or executor.default_model,
        "stream": body.get("stream", False),
        "system_message": system_message,
        "conversation_history": conversation_history if conversation_history else None,
        "options": options if options else None,
    }


def _build_completion_response(
    request_id: str,
    model: str,
    content: str,
    input_tokens: int,
    output_tokens: int,
) -> dict:
    """Build a non-streaming chat completion response."""
    return {
        "id": request_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    }


def _build_stream_chunk(request_id: str, model: str, content: str) -> dict:
    """Build a single streaming chunk in OpenAI delta format."""
    return {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"content": content},
                "finish_reason": None,
            }
        ],
    }


def _build_system_appender(user_system_message: Optional[str] = None):
    """Build a combined system message appender with server mode note."""

    def appender():
        parts = []
        if user_system_message:
            parts.append(user_system_message)
        parts.append(_server_system_appender())
        return "\n".join(parts)

    return appender


def _build_models_response(executor) -> dict:
    """Build the /v1/models response."""
    models = []
    for name in executor.llm_models:
        models.append(
            {
                "id": name,
                "object": "model",
                "created": 0,
                "owned_by": "cliver",
            }
        )
    return {"object": "list", "data": models}


# ---------------------------------------------------------------------------
# Sync / streaming handlers (take executor as parameter)
# ---------------------------------------------------------------------------


async def _handle_sync(executor, request_id: str, parsed: dict):
    """Handle a non-streaming chat completion request."""
    system_appender = _build_system_appender(parsed.get("system_message"))

    try:
        response = await executor.process_user_input(
            user_input=parsed["user_input"],
            model=parsed["model"],
            options=parsed.get("options"),
            conversation_history=parsed.get("conversation_history"),
            system_message_appender=system_appender,
        )

        from cliver.media_handler import extract_response_text

        content = extract_response_text(response)

        input_tokens, output_tokens = 0, 0
        tracker = getattr(executor, "token_tracker", None)
        if tracker and hasattr(tracker, "last_usage") and tracker.last_usage:
            input_tokens = tracker.last_usage.input_tokens
            output_tokens = tracker.last_usage.output_tokens

        result = _build_completion_response(
            request_id,
            parsed["model"],
            content,
            input_tokens,
            output_tokens,
        )
        return JSONResponse(result)

    except Exception as e:
        logger.error(f"Chat completion error: {e}")
        return JSONResponse({"error": {"message": str(e)}}, status_code=500)


async def _handle_streaming(executor, request_id: str, parsed: dict):
    """Handle a streaming chat completion request via SSE."""
    system_appender = _build_system_appender(parsed.get("system_message"))

    async def generate():
        try:
            async for chunk in executor.stream_user_input(
                user_input=parsed["user_input"],
                model=parsed["model"],
                options=parsed.get("options"),
                conversation_history=parsed.get("conversation_history"),
                system_message_appender=system_appender,
            ):
                if hasattr(chunk, "content") and chunk.content:
                    data = _build_stream_chunk(request_id, parsed["model"], str(chunk.content))
                    yield f"data: {json.dumps(data)}\n\n".encode()

            yield b"data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            error_data = {"error": {"message": str(e)}}
            yield f"data: {json.dumps(error_data)}\n\n".encode()

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def get_api_routes(
    executor,
    get_status: Callable,
    api_key: Optional[str] = None,
) -> list:
    """Return OpenAI-compatible API routes as a list of Starlette Route objects.

    Args:
        executor: AgentCore instance
        get_status: callable returning {"uptime": int, "tasks_run": int, "platforms": [...]}
        api_key: optional API key for authentication

    Returns:
        list of starlette.routing.Route
    """

    # --- Auth closure ---

    def check_auth(request: Request) -> bool:
        if not api_key:
            return True
        auth = request.headers.get("Authorization", "")
        parts = auth.split(" ", 1)
        return len(parts) == 2 and parts[0] == "Bearer" and parts[1] == api_key

    # --- Route handlers (closures capturing executor, api_key, get_status) ---

    async def handle_health(request: Request):
        status = get_status()
        return JSONResponse({"status": "ok", **status})

    async def handle_list_models(request: Request):
        if not check_auth(request):
            return JSONResponse({"error": "Unauthorized"}, status_code=401)
        return JSONResponse(_build_models_response(executor))

    async def handle_chat_completions(request: Request):
        if not check_auth(request):
            return JSONResponse({"error": "Unauthorized"}, status_code=401)

        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": {"message": "Invalid JSON body"}}, status_code=400)

        try:
            parsed = _parse_chat_request(body, executor)
        except ValueError as e:
            return JSONResponse({"error": {"message": str(e)}}, status_code=400)

        model = parsed["model"]
        if model and model not in executor.llm_models:
            return JSONResponse({"error": {"message": f"Model '{model}' not found"}}, status_code=404)

        request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

        _configure_server_mode(executor)

        if parsed["stream"]:
            return await _handle_streaming(executor, request_id, parsed)
        else:
            return await _handle_sync(executor, request_id, parsed)

    # --- Return routes ---

    return [
        Route("/health", handle_health),
        Route("/v1/models", handle_list_models),
        Route("/v1/chat/completions", handle_chat_completions, methods=["POST"]),
    ]
