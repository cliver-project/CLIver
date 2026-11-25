"""Media capabilities as agent tools.

The tools are built in the app layer, after ``AgentCore`` exists: each closes
over the agent so calls reuse its same-purpose bindings and failover.  A tool
is advertised only when its purpose is configured.  Files are saved with the
CLI's own naming, and tool results carry text (saved paths or the transcript).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from cliver_core import ModelBindings
from cliver_core.policy import pinned_cwd, scope_violation
from cliver_core.provider import CLIverResponse
from cliver_core.tool import CLIverTool, tool

from cliver import media

__all__ = ["build_media_tools"]

_GENERATION = (
    (
        "image",
        "GenerateImage",
        "Generate an image from a text description and save it to the working directory. "
        "Returns the saved file path(s). Use when the user asks to create, draw, or generate an image.",
    ),
    (
        "video",
        "GenerateVideo",
        "Generate a video from a text description and save it to the working directory. "
        "Returns the saved file path(s). Use when the user asks to create or generate a video.",
    ),
    (
        "tts",
        "TextToSpeech",
        "Synthesize speech audio from text and save it to the working directory. "
        "Returns the saved file path(s). Use when the user asks for spoken audio or text-to-speech.",
    ),
)


def build_media_tools(agent: Any, bindings: ModelBindings | None) -> list[CLIverTool]:
    """Media tools for the purposes bound on *bindings* (configured only)."""
    tools: list[CLIverTool] = []
    for command, name, description in _GENERATION:
        purpose = media.bound_purpose(bindings, command)
        if purpose is not None:
            tools.append(_generation_tool(agent, command, purpose, name, description))
    return tools


def _generation_tool(agent: Any, command: str, purpose: str, name: str, description: str) -> CLIverTool:
    async def handler(prompt: str) -> list[dict]:
        response = await media.run_in_agent(agent, command, purpose, prompt)
        return _save_result(response, command)

    handler.__doc__ = description
    return tool(name=name, description=description, safety="mutating")(handler)


def _save_result(response: CLIverResponse, command: str) -> list[dict]:
    """Save generated media, or report why nothing was saved."""
    failure = media.failure_message(response)
    if failure is not None:
        return [{"error": failure}]
    base = pinned_cwd() or Path.cwd()
    violation = scope_violation(base, write=True)
    if violation is not None:
        return [{"error": violation}]
    saved = media.save_media(response, command, base)
    if not saved:
        detail = (response.message.content or "the provider returned no media").strip()
        return [{"error": detail}]
    paths = "\n".join(media.display_path(path) for path in saved)
    return [{"text": f"Saved {len(saved)} file(s):\n{paths}"}]
