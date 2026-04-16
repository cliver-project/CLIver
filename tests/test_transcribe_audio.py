"""Tests for transcribe_audio tool and voice message transcription."""

import asyncio
from unittest.mock import MagicMock, patch

from cliver.tools.transcribe_audio import TranscribeAudioTool, transcribe_voice_message


def _mock_executor_with_audio_model():
    """Create a mock executor with a model that has AUDIO_TO_TEXT capability."""
    from cliver.model_capabilities import ModelCapability

    mock_config = MagicMock()
    mock_config.get_capabilities.return_value = {ModelCapability.AUDIO_TO_TEXT, ModelCapability.TEXT_TO_TEXT}
    mock_config.get_api_key.return_value = "test-key"
    mock_config.url = "https://api.example.com/v1"
    mock_config.name_in_provider = "whisper-1"
    mock_config.name = "whisper"

    mock_executor = MagicMock()
    mock_executor.llm_models = {"whisper": mock_config}
    return mock_executor


class TestTranscribeAudioTool:
    def test_tool_name(self):
        tool = TranscribeAudioTool()
        assert tool.name == "transcribe_audio"

    def test_file_not_found(self):
        tool = TranscribeAudioTool()
        result = tool._run(file_path="/nonexistent/audio.mp3")
        assert "not found" in result.lower()

    def test_unsupported_format(self, tmp_path):
        bad_file = tmp_path / "audio.xyz"
        bad_file.write_bytes(b"fake")
        tool = TranscribeAudioTool()
        result = tool._run(file_path=str(bad_file))
        assert "unsupported" in result.lower()

    def test_no_audio_model_configured(self, tmp_path):
        """Without a configured audio model, should show helpful error."""
        audio_file = tmp_path / "audio.mp3"
        audio_file.write_bytes(b"fake audio")

        with patch("cliver.tools.transcribe_audio.get_task_executor", return_value=None):
            tool = TranscribeAudioTool()
            result = tool._run(file_path=str(audio_file))

        assert "no model" in result.lower() or "audio_to_text" in result.lower()

    def test_successful_transcription(self, tmp_path):
        """With a configured audio model, should transcribe successfully."""
        audio_file = tmp_path / "audio.mp3"
        audio_file.write_bytes(b"fake audio data")

        mock_executor = _mock_executor_with_audio_model()

        mock_transcript = MagicMock()
        mock_transcript.text = "Hello, this is a test message."
        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = mock_transcript

        with (
            patch("cliver.tools.transcribe_audio.get_task_executor", return_value=mock_executor),
            patch("cliver.tools.transcribe_audio.OpenAI", return_value=mock_client),
        ):
            tool = TranscribeAudioTool()
            result = tool._run(file_path=str(audio_file))

        assert "Hello, this is a test message" in result

    def test_tool_in_core_toolset(self):
        from cliver.tool_registry import TOOLSETS

        assert "transcribe_audio" in TOOLSETS["core"]

    def test_tool_registered(self):
        from cliver.tool_registry import ToolRegistry

        registry = ToolRegistry()
        assert "transcribe_audio" in registry.tool_names


class TestTranscribeVoiceMessage:
    def test_nonexistent_file_returns_none(self):
        result = asyncio.run(transcribe_voice_message("/nonexistent.ogg"))
        assert result is None

    def test_no_model_returns_none(self, tmp_path):
        """Without a configured audio model, voice transcription returns None."""
        audio_file = tmp_path / "voice.ogg"
        audio_file.write_bytes(b"fake audio")

        with patch("cliver.tools.transcribe_audio.get_task_executor", return_value=None):
            result = asyncio.run(transcribe_voice_message(str(audio_file)))

        assert result is None

    def test_successful_transcription(self, tmp_path):
        audio_file = tmp_path / "voice.ogg"
        audio_file.write_bytes(b"fake audio")

        mock_executor = _mock_executor_with_audio_model()

        mock_transcript = MagicMock()
        mock_transcript.text = "Transcribed voice message"
        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = mock_transcript

        with (
            patch("cliver.tools.transcribe_audio.get_task_executor", return_value=mock_executor),
            patch("cliver.tools.transcribe_audio.OpenAI", return_value=mock_client),
        ):
            result = asyncio.run(transcribe_voice_message(str(audio_file)))

        assert result == "Transcribed voice message"
