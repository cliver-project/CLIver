"""Tests for AgentFactory and agent creation."""

from unittest.mock import MagicMock

import pytest

from cliver.config import AgentConfig, AppConfig


def _make_config(**kwargs):
    return AppConfig(
        agents={
            "researcher": AgentConfig(type="cliver", model="test/model", role="Research assistant"),
            "coder": AgentConfig(model="test/model-2", description="Code-focused agent"),
        },
        default_agent="researcher",
        **kwargs,
    )


def _make_factory(config=None):
    from cliver.agents.factory import AgentFactory

    config = config or _make_config()
    mock_core = MagicMock()
    return AgentFactory(config=config, agent_core=mock_core)


def test_create_cliver_agent():
    factory = _make_factory()
    agent = factory.create("researcher")

    from cliver.agents.cliver_agent import CliverAgent

    assert isinstance(agent, CliverAgent)
    assert agent.name == "researcher"
    assert agent.config.role == "Research assistant"


def test_create_agent_with_role_and_model():
    """Agent is a CliverAgent configured with model, role, and description."""
    factory = _make_factory()
    agent = factory.create("coder")

    from cliver.agents.cliver_agent import CliverAgent

    assert isinstance(agent, CliverAgent)
    assert agent.name == "coder"
    assert agent.config.description == "Code-focused agent"


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
