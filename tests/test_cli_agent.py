"""Tests for CliAgent subprocess base."""

import tempfile
from pathlib import Path
from typing import List
from unittest.mock import AsyncMock, patch

import pytest

from cliver.agents.cli_agent import CliAgent
from cliver.config import AgentConfig


class _TestAgent(CliAgent):
    """Minimal test subclass that implements abstract methods."""

    def _build_command(self, prompt: str) -> List[str]:
        return [self._resolved_command, *self._user_args, prompt]

    def _build_env(self) -> dict:
        return self._base_env()


@pytest.fixture
def basic_config():
    return AgentConfig(type="test", command="echo", args=[], timeout_s=30)


def test_claude_env_mapping():
    from cliver.agents.claude_agent import ClaudeAgent

    assert ClaudeAgent.ENV_MAPPING["api_key"] == "ANTHROPIC_API_KEY"
    assert ClaudeAgent.ENV_MAPPING["api_url"] == "ANTHROPIC_API_URL"


def test_gemini_env_mapping():
    from cliver.agents.gemini_agent import GeminiAgent

    assert GeminiAgent.ENV_MAPPING["api_key"] == "GEMINI_API_KEY"


def test_opencode_env_mapping():
    from cliver.agents.opencode_agent import OpenCodeAgent

    assert OpenCodeAgent.ENV_MAPPING["api_key"] == "OPENAI_API_KEY"
    assert OpenCodeAgent.ENV_MAPPING["api_url"] == "OPENAI_BASE_URL"


def test_claude_build_env_sets_model():
    from cliver.agents.claude_agent import ClaudeAgent
    from cliver.config import ModelConfig

    model = ModelConfig(name="anthropic/claude-sonnet-4-20250514", provider="anthropic")
    agent = ClaudeAgent(name="test", config=AgentConfig(type="claude"), model_config=model)
    env = agent._build_env()
    assert env["ANTHROPIC_MODEL"] == "claude-sonnet-4-20250514"


def test_build_env_no_provider(basic_config):
    agent = _TestAgent(name="test", config=basic_config)
    env = agent._build_env()
    assert isinstance(env, dict)
    assert "PATH" in env


def test_build_env_with_anthropic_provider(basic_config):
    from cliver.agents.claude_agent import ClaudeAgent
    from cliver.config import ModelConfig, ProviderConfig

    provider = ProviderConfig(
        name="anthropic",
        type="anthropic",
        api_url="https://api.anthropic.com",
        api_key="sk-ant-test123",
    )
    model = ModelConfig(name="anthropic/claude-sonnet-4-20250514", provider="anthropic")
    model._provider_config = provider

    agent = ClaudeAgent(
        name="test",
        config=basic_config,
        model_config=model,
        provider_config=provider,
    )
    env = agent._build_env()
    assert env["ANTHROPIC_API_KEY"] == "sk-ant-test123"
    assert env["ANTHROPIC_API_URL"] == "https://api.anthropic.com"
    assert env["ANTHROPIC_MODEL"] == "claude-sonnet-4-20250514"


def test_build_env_with_agent_env_override(basic_config):
    basic_config.env = {"MY_VAR": "my_value", "PATH": "/custom/path"}
    agent = _TestAgent(name="test", config=basic_config)
    env = agent._build_env()
    assert env["MY_VAR"] == "my_value"
    assert env["PATH"] == "/custom/path"


def test_snapshot_and_diff_artifacts():
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir)
        (p / "existing.txt").write_text("old")

        agent = _TestAgent(name="test", config=AgentConfig(type="test", command="echo"))
        snapshot = agent._snapshot_dir(p)
        assert len(snapshot) == 1

        (p / "new_image.png").write_bytes(b"\x89PNG")
        (p / "subdir").mkdir()
        (p / "subdir" / "report.pdf").write_bytes(b"%PDF")

        agent._pre_snapshot = snapshot
        artifacts = agent._diff_artifacts(p)
        paths = {a.path for a in artifacts}
        assert str(p / "new_image.png") in paths
        assert str(p / "subdir" / "report.pdf") in paths
        assert len(artifacts) == 2

        png = [a for a in artifacts if a.path.endswith(".png")][0]
        assert png.media_type == "image/png"
        assert png.size == 4


def test_diff_artifacts_no_snapshot():
    agent = _TestAgent(name="test", config=AgentConfig(type="test", command="echo"))
    assert agent._diff_artifacts(Path("/tmp")) == []


def test_parse_response_default():
    agent = _TestAgent(name="test", config=AgentConfig(type="test", command="echo"))
    result = agent._parse_response({"result": "Hello world"})
    assert result.text == "Hello world"
    assert result.status == "completed"


def test_parse_response_fallback_key():
    agent = _TestAgent(name="test", config=AgentConfig(type="test", command="echo"))
    result = agent._parse_response({"response": "From response key"})
    assert result.text == "From response key"


@pytest.mark.asyncio
async def test_initialize_checks_command():
    agent = _TestAgent(
        name="test",
        config=AgentConfig(type="test", command="nonexistent_command_xyz_12345"),
    )
    with pytest.raises(RuntimeError, match="not found"):
        await agent.initialize()


@pytest.mark.asyncio
async def test_initialize_creates_output_dir():
    agent = _TestAgent(
        name="test",
        config=AgentConfig(type="test", command="echo"),
    )
    await agent.initialize()
    assert agent._output_dir is not None
    assert agent._output_dir.exists()
    await agent.cleanup()
    assert not agent._output_dir.exists()


def _make_stdout_stream(lines: list[str]):
    """Build an async iterator over stdout lines (each with \\n appended)."""

    async def _iter():
        for line in lines:
            yield (line + "\n").encode()

    return _iter()


@pytest.mark.asyncio
async def test_do_run_success():
    lines = [
        '{"text": "test output"}',
    ]
    agent = _TestAgent(
        name="test",
        config=AgentConfig(type="test", command="echo", args=[]),
    )
    mock_proc = AsyncMock()
    mock_proc.returncode = 0
    mock_proc.stdout = _make_stdout_stream(lines)
    mock_proc.stderr = _make_stdout_stream([])

    with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=mock_proc)):
        result = await agent._do_run("test prompt")

    assert result.status == "completed"
    assert result.text == "test output"


@pytest.mark.asyncio
async def test_do_run_nonzero_exit():
    agent = _TestAgent(
        name="test",
        config=AgentConfig(type="test", command="false", args=[]),
    )
    mock_proc = AsyncMock()
    mock_proc.returncode = 1
    mock_proc.stdout = _make_stdout_stream([])
    mock_proc.stderr = _make_stdout_stream(["command failed"])

    with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=mock_proc)):
        result = await agent._do_run("test prompt")

    assert result.status == "error"
    assert "exited with code 1" in result.error


@pytest.mark.asyncio
async def test_do_run_invalid_json():
    lines = ["not json output"]
    agent = _TestAgent(
        name="test",
        config=AgentConfig(type="test", command="echo", args=[]),
    )
    mock_proc = AsyncMock()
    mock_proc.returncode = 0
    mock_proc.stdout = _make_stdout_stream(lines)
    mock_proc.stderr = _make_stdout_stream([])

    with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=mock_proc)):
        result = await agent._do_run("test prompt")

    # Non-JSON lines end up in the buffer and are eventually flushed as text
    assert result.status == "completed"
    assert "not json output" in result.text
