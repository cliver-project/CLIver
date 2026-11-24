"""Media commands shared by the one-shot CLI and the REPL.

One-shot commands have no agent (a media-only configuration needs no chat
model), so they call the standalone ``cliver_core.capabilities`` functions
with the configured primary binding.  The REPL runs through the agent, which
also gives same-purpose fallback.  Saving, naming, and failure reporting are
the same for both, so they live here.

Generated files are named by the CLI, not by core: ``<command>-<timestamp>-<n>
<ext>`` in the target directory.  Core's ``generated_<n>.<ext>`` would
overwrite between runs.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cliver_core import ModelBinding
from cliver_core.capabilities import (
    generate_image,
    generate_video,
    text_to_speech,
    transcribe_audio,
)
from cliver_core.provider import CLIverResponse

from cliver.config import CONFIG_FILENAME, CLIConfig
from cliver.paths import get_config_dir

if TYPE_CHECKING:
    from cliver_core import ModelBindings

__all__ = [
    "MEDIA_COMMANDS",
    "binding_for",
    "bound_purpose",
    "config_hint",
    "display_path",
    "failure_message",
    "not_configured_message",
    "run_in_agent",
    "run_once",
    "save_media",
    "status_line",
]

#: The first positional word that selects a media command.
MEDIA_COMMANDS = ("image", "video", "audio", "tts", "asr")

#: Config purposes that can serve each command, preferred first.  ``audio``
#: is the legacy TTS slot; it still works when no ``tts`` entry exists.
_PURPOSES = {
    "image": ("image",),
    "video": ("video",),
    "audio": ("tts", "audio"),
    "tts": ("tts", "audio"),
    "asr": ("asr",),
}


def binding_for(config: CLIConfig, command: str) -> tuple[str, ModelBinding]:
    """The primary binding for *command*'s purpose, from the JSON config only.

    One-shot media commands deliberately ignore ``CLIVER_*`` chat settings and
    the agent's failover list: they run the configured primary media binding.
    """
    for purpose in _PURPOSES[command]:
        bindings = config.models.get(purpose) or ()
        if bindings:
            return purpose, bindings[0]
    raise ValueError(not_configured_message(command, config_hint(config)))


def bound_purpose(bindings: ModelBindings | None, command: str) -> str | None:
    """The first purpose *command* can use on an agent's bindings."""
    if bindings is None:
        return None
    for purpose in _PURPOSES[command]:
        if bindings.get_all(purpose):
            return purpose
    return None


def config_hint(config: CLIConfig) -> str:
    """The file to name when a purpose is missing from the configuration."""
    if config.paths:
        return str(config.paths[0])
    root = config.config_dir or get_config_dir()
    return str(root / CONFIG_FILENAME)


def not_configured_message(command: str, hint: str = "") -> str:
    label = "audio" if command in {"audio", "tts"} else command
    key = "tts" if label == "audio" else command
    message = f'no {label} model configured: add "{key}" under "models"'
    if hint:
        message += f" in {hint}"
    return message


def status_line(command: str, source: str) -> str:
    """The stderr progress line printed before a media request."""
    if command == "asr":
        return f"[media] transcribing {source} ..."
    return f"[media] generating {command} ..."


def failure_message(response: CLIverResponse) -> str | None:
    """The error text of a swallowed media failure, or ``None`` on success.

    The provider pipeline reports wire failures as a normal response with
    ``retryable`` set, so exceptions alone are not enough to spot a failure.
    """
    if response.retryable is None:
        return None
    return response.message.content or "media request failed"


async def run_once(
    binding: ModelBinding,
    command: str,
    prompt: str,
    *,
    language: str | None = None,
) -> CLIverResponse:
    """Run one media command on a standalone binding (no agent, no failover)."""
    if command == "image":
        return await generate_image(prompt=prompt, binding=binding)
    if command == "video":
        return await generate_video(prompt=prompt, binding=binding)
    if command in {"audio", "tts"}:
        return await text_to_speech(text=prompt, binding=binding)
    return await transcribe_audio(audio=prompt, binding=binding, language=language)


async def run_in_agent(
    agent: Any,
    command: str,
    purpose: str,
    prompt: str,
    *,
    language: str | None = None,
) -> CLIverResponse:
    """Run one media command through the agent (same-purpose fallback)."""
    if command == "image":
        return await agent.image(prompt)
    if command == "video":
        return await agent.video(prompt)
    if command == "asr":
        return await agent.asr(prompt, language=language)
    if purpose == "tts":
        return await agent.tts(prompt)
    return await agent.audio(prompt)


def save_media(response: CLIverResponse, command: str, directory: Path) -> list[Path]:
    """Save every media item under a unique CLI-owned name; return the paths."""
    target_dir = Path(directory)
    target_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for index, item in enumerate(response.media or ()):
        target = unique_media_path(target_dir, command, index, item.get_file_extension())
        item.save(target)
        saved.append(target)
    return saved


def unique_media_path(directory: Path, command: str, index: int, extension: str) -> Path:
    stamp = _timestamp()
    candidate = directory / f"{command}-{stamp}-{index}{extension}"
    counter = 1
    while candidate.exists():
        candidate = directory / f"{command}-{stamp}-{index}-{counter}{extension}"
        counter += 1
    return candidate


def display_path(path: Path) -> str:
    """A path relative to the working directory when it lives there."""
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")
