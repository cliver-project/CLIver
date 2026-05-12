"""Tests for AgentFactory and type registry."""

import pytest
from unittest.mock import MagicMock

from cliver.agent import Agent
from cliver.config import AgentConfig, AppConfig


def _make_config(**kwargs):
    return AppConfig(
        agents={
            "researcher": AgentConfig(type="cliver", model="test/model", role="Research assistant"),
            "coder": AgentConfig(type="claude", timeout_s=600),
            "custom": AgentConfig(type="aider", command="aider", args=["--message"]),
        },
        default_agent="researcher",
        **kwargs,
    )


def _make_factory(config=None):
    from cliver.agents.factory import AgentFactory

    config = config or _make_config()
    mock_core = MagicMock()
    return AgentFactory(config=config, agent_core=mock_core)


def test_registry_contains_builtin_types():
    from cliver.agents.factory import AGENT_REGISTRY

    assert "cliver" in AGENT_REGISTRY
    assert "claude" in AGENT_REGISTRY
    assert "gemini" in AGENT_REGISTRY
    assert "opencode" in AGENT_REGISTRY


def test_create_cliver_agent():
    factory = _make_factory()
    agent = factory.create("researcher")

    from cliver.agents.cliver_agent import CliverAgent
    assert isinstance(agent, CliverAgent)
    assert agent.name == "researcher"
    assert agent.config.role == "Research assistant"


def test_create_claude_agent():
    factory = _make_factory()
    agent = factory.create("coder")

    from cliver.agents.claude_agent import ClaudeAgent
    assert isinstance(agent, ClaudeAgent)
    assert agent.config.timeout_s == 600


def test_create_custom_type_falls_back_to_cli_agent():
    factory = _make_factory()
    agent = factory.create("custom")

    from cliver.agents.cli_agent import CliAgent
    assert isinstance(agent, CliAgent)
    assert agent._command == "aider"
    assert agent._args == ["--message"]


def test_create_default_agent():
    factory = _make_factory()
    agent = factory.create()

    assert agent.name == "researcher"


def test_create_unknown_name_returns_default_cliver():
    factory = _make_factory()
    agent = factory.create("nonexistent")

    from cliver.agents.cliver_agent import CliverAgent
    assert isinstance(agent, CliverAgent)


def test_agent_caching():
    factory = _make_factory()
    a1 = factory.create("researcher")
    a2 = factory.create("researcher")
    assert a1 is a2


@pytest.mark.asyncio
async def test_cleanup_all():
    factory = _make_factory()
    factory.create("researcher")
    factory.create("coder")
    assert len(factory._agents) == 2

    await factory.cleanup_all()
    assert len(factory._agents) == 0
