"""Built-in tool for transcribing audio files to text.

Uses whatever configured model has AUDIO_TO_TEXT capability — works with
OpenAI Whisper, Qwen-Audio, Groq Whisper, or any compatible provider.
No hardcoded provider — users configure the model in config.yaml.
"""

import logging
from pathlib import Path
from typing import Optional, Type

from langchain_core.tools import BaseTool
from openai import OpenAI
from pydantic import BaseModel, Field

from cliver.agent_profile import get_task_executor

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

        config = _find_audio_model()
        if not config:
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

        return _transcribe(path, config, language)


def _find_audio_model() -> Optional[dict]:
    """Find a configured model with AUDIO_TO_TEXT capability."""
    executor = get_task_executor()
    if not executor:
        return None

    from cliver.model_capabilities import ModelCapability

    for _name, model_config in executor.llm_models.items():
        caps = model_config.get_capabilities()
        if ModelCapability.AUDIO_TO_TEXT in caps:
            return {
                "api_key": model_config.get_api_key(),
                "url": model_config.url,
                "model_name": model_config.name_in_provider or model_config.name,
            }
    return None


def _transcribe(file_path: Path, config: dict, language: Optional[str] = None) -> str:
    """Transcribe using a configured model's API endpoint."""
    api_key = config.get("api_key")
    if not api_key:
        return "Error: Model has no API key configured."

    try:
        client_kwargs = {"api_key": api_key}
        base_url = config.get("url")
        if base_url:
            client_kwargs["base_url"] = base_url

        client = OpenAI(**client_kwargs)
        model_name = config.get("model_name", "whisper-1")

        with open(file_path, "rb") as f:
            kwargs = {"model": model_name, "file": f}
            if language:
                kwargs["language"] = language
            transcript = client.audio.transcriptions.create(**kwargs)

        text = transcript.text if hasattr(transcript, "text") else str(transcript)

        if not text or not text.strip():
            return "(empty transcription — no speech detected)"

        logger.info("Transcribed %s via %s: %d chars", file_path.name, model_name, len(text))
        return text

    except Exception as e:
        logger.warning(f"Transcription failed: {e}")
        return f"Error transcribing audio: {e}"


async def transcribe_voice_message(file_path: str, language: Optional[str] = None) -> Optional[str]:
    """Transcribe a voice message file. Returns text or None on failure.

    Used by the Gateway to auto-transcribe voice messages before passing
    them to AgentCore.
    """
    path = Path(file_path)
    if not path.exists():
        return None

    config = _find_audio_model()
    if not config:
        logger.debug("No audio_to_text model configured, skipping voice transcription")
        return None

    result = _transcribe(path, config, language)

    if result.startswith("Error:") or result.startswith("(empty"):
        logger.debug("Voice transcription failed or empty: %s", result)
        return None

    return result


transcribe_audio = TranscribeAudioTool()
