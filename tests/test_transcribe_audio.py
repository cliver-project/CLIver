"""Tests for transcribe_audio tool and voice message transcription."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from cliver.tools.transcribe_audio import TranscribeAudioTool, transcribe_voice_message


def _mock_agent_core(transcribe_result="Hello, this is a test message."):
    """Create a mock AgentCore with transcribe_audio support."""
    mock = MagicMock()
    mock.transcribe_audio = AsyncMock(return_value=transcribe_result)
    return mock


class TestTranscribeAudioTool:
    def test_tool_name(self):
        tool = TranscribeAudioTool()
        assert tool.name == "Transcribe"

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

    def test_no_agent_core(self, tmp_path):
        audio_file = tmp_path / "audio.mp3"
        audio_file.write_bytes(b"fake audio")

        with patch("cliver.tools.transcribe_audio.get_agent_core", return_value=None):
            tool = TranscribeAudioTool()
            result = tool._run(file_path=str(audio_file))

        assert "agentcore not available" in result.lower()

    def test_no_audio_model_configured(self, tmp_path):
        """AgentCore returns None when no audio model is configured."""
        audio_file = tmp_path / "audio.mp3"
        audio_file.write_bytes(b"fake audio")

        mock_core = _mock_agent_core(transcribe_result=None)
        with patch("cliver.tools.transcribe_audio.get_agent_core", return_value=mock_core):
            tool = TranscribeAudioTool()
            result = tool._run(file_path=str(audio_file))

        assert "audio_to_text" in result.lower()

    def test_successful_transcription(self, tmp_path):
        audio_file = tmp_path / "audio.mp3"
        audio_file.write_bytes(b"fake audio data")

        mock_core = _mock_agent_core("Hello, this is a test message.")
        with patch("cliver.tools.transcribe_audio.get_agent_core", return_value=mock_core):
            tool = TranscribeAudioTool()
            result = tool._run(file_path=str(audio_file))

        assert "Hello, this is a test message" in result
        mock_core.transcribe_audio.assert_called_once()

    def test_tool_in_core_toolset(self):
        from cliver.tool_registry import TOOLSETS

        assert "Transcribe" in TOOLSETS["core"]

    def test_tool_registered(self):
        from cliver.tool_registry import ToolRegistry

        registry = ToolRegistry()
        assert "Transcribe" in registry.tool_names


class TestTranscribeVoiceMessage:
    def test_nonexistent_file_returns_none(self):
        result = asyncio.run(transcribe_voice_message("/nonexistent.ogg"))
        assert result is None

    def test_no_agent_core_returns_none(self, tmp_path):
        audio_file = tmp_path / "voice.ogg"
        audio_file.write_bytes(b"fake audio")

        with patch("cliver.tools.transcribe_audio.get_agent_core", return_value=None):
            result = asyncio.run(transcribe_voice_message(str(audio_file)))

        assert result is None

    def test_no_model_returns_none(self, tmp_path):
        """AgentCore returns None when no audio model configured."""
        audio_file = tmp_path / "voice.ogg"
        audio_file.write_bytes(b"fake audio")

        mock_core = _mock_agent_core(transcribe_result=None)
        with patch("cliver.tools.transcribe_audio.get_agent_core", return_value=mock_core):
            result = asyncio.run(transcribe_voice_message(str(audio_file)))

        assert result is None

    def test_successful_transcription(self, tmp_path):
        audio_file = tmp_path / "voice.ogg"
        audio_file.write_bytes(b"fake audio")

        mock_core = _mock_agent_core("Transcribed voice message")
        with patch("cliver.tools.transcribe_audio.get_agent_core", return_value=mock_core):
            result = asyncio.run(transcribe_voice_message(str(audio_file)))

        assert result == "Transcribed voice message"
