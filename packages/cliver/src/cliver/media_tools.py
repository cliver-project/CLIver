"""Media capabilities as agent tools.

The tools are built in the app layer, after ``AgentCore`` exists: each closes
over the agent so calls reuse its same-purpose bindings and failover.  A tool
is advertised only when its purpose is configured.  Files are saved with the
CLI's own naming, and tool results carry text (saved paths or the transcript).
"""

from __future__ import annotations

import re
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

_ASR_DESCRIPTION = (
    "Transcribe a local audio file or an audio URL to text. "
    "Use when the user asks what an audio file says or wants a transcript. "
    "Sensitive files (e.g. .env, keys, credentials) are refused."
)

# Mirrors the sensitive-file guard in ``cliver_core.tools.read_file`` until core
# exposes a shared helper: transcription uploads the file's bytes to the
# provider, so the same names Read refuses must be refused here.
_SENSITIVE_PATTERNS = [
    r"\.env$",
    r"\.env\..+$",
    r"credentials\.json$",
    r"credentials\.yaml$",
    r"credentials\.yml$",
    r"\.secret$",
    r"\.secrets$",
    r"secrets\.json$",
    r"secrets\.yaml$",
    r"secrets\.yml$",
    r"\.pem$",
    r"\.key$",
    r"id_rsa$",
    r"id_ed25519$",
    r"id_dsa$",
    r"id_ecdsa$",
    r"\.p12$",
    r"\.pfx$",
    r"\.keystore$",
    r"\.jks$",
    r"htpasswd$",
    r"shadow$",
    r"\.netrc$",
    r"\.pgpass$",
    r"token\.json$",
    r"service[-_]?account.*\.json$",
    r"kubeconfig$",
]
_SENSITIVE_RE = re.compile("|".join(f"(?:{pattern})" for pattern in _SENSITIVE_PATTERNS), re.IGNORECASE)


def build_media_tools(agent: Any, bindings: ModelBindings | None) -> list[CLIverTool]:
    """Media tools for the purposes bound on *bindings* (configured only)."""
    tools: list[CLIverTool] = []
    for command, name, description in _GENERATION:
        purpose = media.bound_purpose(bindings, command)
        if purpose is not None:
            tools.append(_generation_tool(agent, command, purpose, name, description))
    purpose = media.bound_purpose(bindings, "asr")
    if purpose is not None:
        tools.append(_asr_tool(agent, purpose))
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


def _asr_tool(agent: Any, purpose: str) -> CLIverTool:
    async def handler(source: str, language: str = "") -> list[dict]:
        resolved = _resolve_source(source)
        if not _is_url(resolved):
            violation = scope_violation(resolved, write=False)
            if violation is not None:
                return [{"error": violation}]
            if _is_sensitive_file(resolved):
                return [{"error": f"Access denied: '{Path(resolved).name}' appears to be a sensitive file."}]
        response = await media.run_in_agent(agent, "asr", purpose, resolved, language=language or None)
        failure = media.failure_message(response)
        if failure is not None:
            return [{"error": failure}]
        transcript = (response.message.content or "").strip()
        if not transcript:
            return [{"error": "no transcript returned"}]
        return [{"text": transcript}]

    handler.__doc__ = _ASR_DESCRIPTION
    return tool(name="TranscribeAudio", description=_ASR_DESCRIPTION, safety="safe")(handler)


def _resolve_source(source: str) -> str:
    """Rebase a relative local path onto the pinned cwd; leave URLs alone."""
    if _is_url(source) or Path(source).is_absolute():
        return source
    return str((pinned_cwd() or Path.cwd()) / source)


def _is_url(value: str) -> bool:
    return value.startswith(("http://", "https://"))


def _is_sensitive_file(path: str) -> bool:
    return bool(_SENSITIVE_RE.search(Path(path).name))
