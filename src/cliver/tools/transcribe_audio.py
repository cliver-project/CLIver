"""Built-in tool for transcribing audio files to text.

All inference is routed through AgentCore → LLMInferenceEngine, which
handles provider selection, rate limiting, and API calls. No direct
SDK usage here — the engine abstraction supports any provider with
AUDIO_TO_TEXT capability.
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from cliver.agent_profile import get_agent_core

logger = logging.getLogger(__name__)

# Supported audio formats
_SUPPORTED_FORMATS = {".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm", ".ogg", ".oga", ".flac"}


class TranscribeAudioInput(BaseModel):
    """Input schema for transcribe_audio."""

    file_path: str = Field(description="Path to the audio file to transcribe")
    language: Optional[str] = Field(
        default=None,
        description="Language hint (ISO 639-1 code, e.g., 'en', 'zh', 'ja'). Auto-detected if not set.",
    )


class TranscribeAudioTool(BaseTool):
    """Transcribe audio files to text using a configured model with AUDIO_TO_TEXT capability."""

    name: str = "Transcribe"
    description: str = (
        "Transcribe an audio or voice file to text. Supports mp3, wav, ogg, m4a, webm, flac. "
        "Use when you receive audio files or voice messages that need to be converted to text. "
        "Requires a model with audio_to_text capability configured in config.yaml."
    )
    args_schema: Type[BaseModel] = TranscribeAudioInput

    def _run(self, file_path: str, language: Optional[str] = None) -> str:
        path = Path(file_path)
        if not path.exists():
            return f"Error: File not found: {file_path}"

        suffix = path.suffix.lower()
        if suffix not in _SUPPORTED_FORMATS:
            return f"Error: Unsupported format '{suffix}'. Supported: {', '.join(sorted(_SUPPORTED_FORMATS))}"

        agent_core = get_agent_core()
        if not agent_core:
            return "Error: AgentCore not available."

        result = asyncio.run(agent_core.transcribe_audio(file_path, language))

        if result is None:
            return (
                "Error: No model with audio_to_text capability configured.\n"
                "Add one to config.yaml, e.g.:\n"
                "  whisper:\n"
                "    provider: openai\n"
                "    url: https://api.openai.com/v1\n"
                "    api_key: '{{ env.OPENAI_API_KEY }}'\n"
                "    name_in_provider: whisper-1\n"
                "    capabilities: [audio_to_text]"
            )

        if not result or not result.strip():
            return "(empty transcription — no speech detected)"

        return result


async def transcribe_voice_message(file_path: str, language: Optional[str] = None) -> Optional[str]:
    """Transcribe a voice message file. Returns text or None on failure.

    Used by the Gateway to auto-transcribe voice messages before passing
    them to AgentCore.
    """
    path = Path(file_path)
    if not path.exists():
        return None

    agent_core = get_agent_core()
    if not agent_core:
        logger.debug("AgentCore not available, skipping voice transcription")
        return None

    result = await agent_core.transcribe_audio(file_path, language)

    if not result or not result.strip():
        logger.debug("Voice transcription returned empty result")
        return None

    return result


transcribe_audio = TranscribeAudioTool()
