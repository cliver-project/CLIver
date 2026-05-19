"""Stream agent responses as SSE event dicts for lab cell chat sessions."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, AsyncIterator, Dict

if TYPE_CHECKING:
    from cliver.agent import Agent
    from cliver.session_manager import SessionManager

logger = logging.getLogger(__name__)


async def stream_chat_response(
    agent: "Agent",
    prompt: str,
    session_manager: "SessionManager",
    session_id: str,
    output_format: str = "text",
) -> AsyncIterator[Dict]:
    """Stream agent response as a sequence of SSE event dicts.

    Yields dicts with 'type' key: thinking, text, tool, tool_use,
    tool_result, status, done, error.
    """
    full_text: list[str] = []

    try:
        async for chunk in agent.stream(prompt):
            event = {"type": chunk.chunk_type}
            if chunk.chunk_type in ("thinking", "tool", "tool_use", "tool_result", "status"):
                event["content"] = chunk.text
            elif chunk.chunk_type == "text":
                if chunk.text:
                    full_text.append(chunk.text)
                event["content"] = chunk.text or ""
            elif chunk.chunk_type == "done":
                result_text = "".join(full_text)
                event["text"] = result_text
                # Persist assistant turn
                try:
                    session_manager.append_turn(session_id, "assistant", result_text)
                except Exception:
                    logger.warning("Failed to save assistant turn", exc_info=True)

                if output_format == "json":
                    try:
                        event["data"] = json.loads(result_text)
                    except (json.JSONDecodeError, TypeError):
                        pass

                if chunk.final_result and chunk.final_result.artifacts:
                    event["artifacts"] = [
                        {"path": a.path, "media_type": a.media_type, "size": a.size}
                        for a in chunk.final_result.artifacts
                    ]

            yield event

            if chunk.chunk_type == "done":
                return

    except Exception as e:
        logger.warning("Chat stream error: %s", e)
        yield {"type": "error", "message": str(e)}
