"""
OpenAI-compatible API server for CLIver.

Exposes CLIver's agent capabilities via standard OpenAI API endpoints.
Runs inside the Gateway daemon as an aiohttp web server.
Stateless — each request runs a full Re-Act loop and returns the final answer.
"""

import json
import logging
import time
import uuid
from typing import Optional

from cliver.config import APIServerConfig

logger = logging.getLogger(__name__)


def _server_system_appender():
    return (
        "\n## Server Mode\n\n"
        "You are running as a backend API service. "
        "Do NOT use Ask — there is no human to respond. "
        "Make autonomous decisions. Be concise and direct."
    )


class APIServer:
    """OpenAI-compatible HTTP API server backed by AgentCore."""

    def __init__(self, task_executor, config: APIServerConfig):
        self._executor = task_executor
        self._config = config
        self._app = None
        self._runner = None

    async def start(self) -> None:
        """Start the aiohttp web server."""
        try:
            from aiohttp import web
        except ImportError as e:
            raise ImportError("aiohttp is required for the API server. Install it with: pip install cliver[api]") from e

        self._app = web.Application()
        self._app.router.add_post("/v1/chat/completions", self._handle_chat_completions)
        self._app.router.add_get("/v1/models", self._handle_list_models)
        self._app.router.add_get("/health", self._handle_health)

        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self._config.host, self._config.port)
        await site.start()
        logger.info(f"API server started on http://{self._config.host}:{self._config.port}")

    async def stop(self) -> None:
        """Stop the web server."""
        if self._runner:
            await self._runner.cleanup()
            logger.info("API server stopped")

    # --- HTTP Handlers ---

    async def _handle_health(self, request):
        from aiohttp import web

        return web.json_response({"status": "ok"})

    async def _handle_list_models(self, request):
        from aiohttp import web

        if not self._check_auth(request.headers.get("Authorization")):
            return web.json_response({"error": "Unauthorized"}, status=401)

        return web.json_response(self._build_models_response())

    async def _handle_chat_completions(self, request):
        from aiohttp import web

        if not self._check_auth(request.headers.get("Authorization")):
            return web.json_response({"error": "Unauthorized"}, status=401)

        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": {"message": "Invalid JSON body"}}, status=400)

        try:
            parsed = self._parse_chat_request(body)
        except ValueError as e:
            return web.json_response({"error": {"message": str(e)}}, status=400)

        # Check model exists
        model = parsed["model"]
        if model and model not in self._executor.llm_models:
            return web.json_response({"error": {"message": f"Model '{model}' not found"}}, status=404)

        request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

        # Configure server mode: yolo permissions, no ask_user
        self._configure_server_mode()

        if parsed["stream"]:
            return await self._handle_streaming(request, request_id, parsed)
        else:
            return await self._handle_sync(request_id, parsed)

    def _configure_server_mode(self):
        """Set up executor for headless server operation."""
        # Set yolo permission mode so all tools auto-allow
        if self._executor.permission_manager:
            from cliver.permissions import PermissionMode

            self._executor.permission_manager.set_mode(PermissionMode.YOLO)

        # Auto-allow any permission prompts (no human to respond)
        self._executor.on_permission_prompt = lambda tool, args: "allow"

    async def _handle_sync(self, request_id: str, parsed: dict):
        """Handle a non-streaming chat completion request."""
        from aiohttp import web

        # Build combined system message appender
        system_appender = self._build_system_appender(parsed.get("system_message"))

        try:
            response = await self._executor.process_user_input(
                user_input=parsed["user_input"],
                model=parsed["model"],
                options=parsed.get("options"),
                conversation_history=parsed.get("conversation_history"),
                system_message_appender=system_appender,
            )

            content = str(response.content) if response and response.content else ""

            # Extract token usage
            input_tokens, output_tokens = 0, 0
            tracker = getattr(self._executor, "token_tracker", None)
            if tracker and hasattr(tracker, "last_usage") and tracker.last_usage:
                input_tokens = tracker.last_usage.input_tokens
                output_tokens = tracker.last_usage.output_tokens

            result = self._build_completion_response(
                request_id,
                parsed["model"],
                content,
                input_tokens,
                output_tokens,
            )
            return web.json_response(result)

        except Exception as e:
            logger.error(f"Chat completion error: {e}")
            return web.json_response({"error": {"message": str(e)}}, status=500)

    async def _handle_streaming(self, request, request_id: str, parsed: dict):
        """Handle a streaming chat completion request via SSE."""
        from aiohttp import web

        response = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )
        await response.prepare(request)

        # Build combined system message appender
        system_appender = self._build_system_appender(parsed.get("system_message"))

        try:
            async for chunk in self._executor.stream_user_input(
                user_input=parsed["user_input"],
                model=parsed["model"],
                options=parsed.get("options"),
                conversation_history=parsed.get("conversation_history"),
                system_message_appender=system_appender,
            ):
                if hasattr(chunk, "content") and chunk.content:
                    data = self._build_stream_chunk(request_id, parsed["model"], str(chunk.content))
                    await response.write(f"data: {json.dumps(data)}\n\n".encode())

            # Send [DONE]
            await response.write(b"data: [DONE]\n\n")

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            error_data = {"error": {"message": str(e)}}
            await response.write(f"data: {json.dumps(error_data)}\n\n".encode())

        return response

    # --- System Message ---

    @staticmethod
    def _build_system_appender(user_system_message: Optional[str] = None):
        """Build a combined system message appender with server mode note."""

        def appender():
            parts = []
            if user_system_message:
                parts.append(user_system_message)
            parts.append(_server_system_appender())
            return "\n".join(parts)

        return appender

    # --- Auth ---

    def _check_auth(self, auth_header: Optional[str]) -> bool:
        """Check authorization. Returns True if no api_key configured."""
        if not self._config.api_key:
            return True
        if not auth_header:
            return False
        # Expect "Bearer <key>"
        parts = auth_header.split(" ", 1)
        if len(parts) != 2 or parts[0] != "Bearer":
            return False
        return parts[1] == self._config.api_key

    # --- Request Parsing ---

    def _parse_chat_request(self, body: dict) -> dict:
        """Parse an OpenAI chat completion request body."""
        messages = body.get("messages")
        if not messages or not isinstance(messages, list):
            raise ValueError("'messages' field is required and must be a list")

        # Extract system message, conversation history, and user input
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
                # If we already have a user_input, the previous one is history
                if user_input:
                    conversation_history.append(HumanMessage(content=user_input))
                user_input = content
            elif role == "assistant":
                if content:
                    conversation_history.append(AIMessage(content=content))

        if not user_input:
            raise ValueError("No user message found in messages")

        # Extract options
        options = {}
        for key in ("temperature", "max_tokens", "top_p", "frequency_penalty", "presence_penalty"):
            if key in body:
                options[key] = body[key]

        return {
            "user_input": user_input,
            "model": body.get("model") or self._executor.default_model,
            "stream": body.get("stream", False),
            "system_message": system_message,
            "conversation_history": conversation_history if conversation_history else None,
            "options": options if options else None,
        }

    # --- Response Building ---

    def _build_models_response(self) -> dict:
        """Build the /v1/models response."""
        models = []
        for name in self._executor.llm_models:
            models.append(
                {
                    "id": name,
                    "object": "model",
                    "created": 0,
                    "owned_by": "cliver",
                }
            )
        return {"object": "list", "data": models}

    @staticmethod
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

    @staticmethod
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
