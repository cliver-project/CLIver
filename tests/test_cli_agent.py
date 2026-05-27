"""Tests for CliAgent subprocess base."""

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from cliver.config import AgentConfig


@pytest.fixture
def basic_config():
    return AgentConfig(type="test", command="echo", args=[], timeout_s=30)


def test_env_var_mapping_exists():
    from cliver.agents.cli_agent import ENV_VAR_MAPPING

    assert "anthropic" in ENV_VAR_MAPPING
    assert "openai" in ENV_VAR_MAPPING
    assert "google" in ENV_VAR_MAPPING
    assert "deepseek" in ENV_VAR_MAPPING


def test_build_env_no_provider(basic_config):
    from cliver.agents.cli_agent import CliAgent

    agent = CliAgent(name="test", config=basic_config)
    env = agent._build_env()
    assert isinstance(env, dict)
    assert "PATH" in env


def test_build_env_with_anthropic_provider(basic_config):
    from cliver.agents.cli_agent import CliAgent
    from cliver.config import ModelConfig, ProviderConfig

    provider = ProviderConfig(
        name="anthropic",
        type="anthropic",
        api_url="https://api.anthropic.com",
        api_key="sk-ant-test123",
    )
    model = ModelConfig(name="anthropic/claude-sonnet-4-20250514", provider="anthropic")
    model._provider_config = provider

    agent = CliAgent(
        name="test",
        config=basic_config,
        model_config=model,
        provider_config=provider,
    )
    env = agent._build_env()
    assert env["ANTHROPIC_API_KEY"] == "sk-ant-test123"
    assert env["ANTHROPIC_API_URL"] == "https://api.anthropic.com"


def test_build_env_with_agent_env_override(basic_config):
    from cliver.agents.cli_agent import CliAgent

    basic_config.env = {"MY_VAR": "my_value", "PATH": "/custom/path"}
    agent = CliAgent(name="test", config=basic_config)
    env = agent._build_env()
    assert env["MY_VAR"] == "my_value"
    assert env["PATH"] == "/custom/path"


def test_snapshot_and_diff_artifacts():
    from cliver.agents.cli_agent import CliAgent

    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir)
        (p / "existing.txt").write_text("old")

        agent = CliAgent(name="test", config=AgentConfig(type="test", command="echo"))
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
    from cliver.agents.cli_agent import CliAgent

    agent = CliAgent(name="test", config=AgentConfig(type="test", command="echo"))
    assert agent._diff_artifacts(Path("/tmp")) == []


def test_parse_response_default():
    from cliver.agents.cli_agent import CliAgent

    agent = CliAgent(name="test", config=AgentConfig(type="test", command="echo"))
    result = agent._parse_response({"result": "Hello world"})
    assert result.text == "Hello world"
    assert result.status == "completed"


def test_parse_response_fallback_key():
    from cliver.agents.cli_agent import CliAgent

    agent = CliAgent(name="test", config=AgentConfig(type="test", command="echo"))
    result = agent._parse_response({"response": "From response key"})
    assert result.text == "From response key"


@pytest.mark.asyncio
async def test_initialize_checks_command():
    from cliver.agents.cli_agent import CliAgent

    agent = CliAgent(
        name="test",
        config=AgentConfig(type="test", command="nonexistent_command_xyz_12345"),
    )
    with pytest.raises(RuntimeError, match="not found"):
        await agent.initialize()


@pytest.mark.asyncio
async def test_initialize_creates_output_dir():
    from cliver.agents.cli_agent import CliAgent

    agent = CliAgent(
        name="test",
        config=AgentConfig(type="test", command="echo"),
    )
    await agent.initialize()
    assert agent._output_dir is not None
    assert agent._output_dir.exists()
    await agent.cleanup()
    assert not agent._output_dir.exists()


@pytest.mark.asyncio
async def test_do_run_success():
    from cliver.agents.cli_agent import CliAgent

    response = {"result": "test output", "model": "test-model"}
    agent = CliAgent(
        name="test",
        config=AgentConfig(type="test", command="echo", args=[]),
    )
    mock_proc = AsyncMock()
    mock_proc.returncode = 0
    mock_proc.communicate = AsyncMock(return_value=(json.dumps(response).encode(), b""))

    with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
        result = await agent._do_run("test prompt")

    assert result.status == "completed"
    assert result.text == "test output"
    assert result.raw == response


@pytest.mark.asyncio
async def test_do_run_nonzero_exit():
    from cliver.agents.cli_agent import CliAgent

    agent = CliAgent(
        name="test",
        config=AgentConfig(type="test", command="false", args=[]),
    )
    mock_proc = AsyncMock()
    mock_proc.returncode = 1
    mock_proc.communicate = AsyncMock(return_value=(b"", b"command failed"))

    with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
        result = await agent._do_run("test prompt")

    assert result.status == "error"
    assert "exited with code 1" in result.error


@pytest.mark.asyncio
async def test_do_run_invalid_json():
    from cliver.agents.cli_agent import CliAgent

    agent = CliAgent(
        name="test",
        config=AgentConfig(type="test", command="echo", args=[]),
    )
    mock_proc = AsyncMock()
    mock_proc.returncode = 0
    mock_proc.communicate = AsyncMock(return_value=(b"not json output", b""))

    with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
        result = await agent._do_run("test prompt")

    assert result.status == "completed"
    assert result.text == "not json output"
