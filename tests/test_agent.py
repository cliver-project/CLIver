"""Tests for Agent ABC, CliverAgent, and AgentFactory."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from cliver.agent import _AGENT_TYPES, Agent, AgentFactory, CliverAgent, register_agent_type
from cliver.agent_profile import get_agent_factory, set_agent_factory
from cliver.config import AgentConfig, AppConfig


@pytest.fixture
def mock_agent_core():
    core = MagicMock()
    core.process_user_input = AsyncMock(return_value=MagicMock(content="test response"))
    core.stream_user_input = AsyncMock()
    core.default_model = "test-model"
    return core


@pytest.fixture
def agent_config():
    return AgentConfig(
        type="cliver",
        description="Test agent",
        role="You are a test assistant",
        model="test-model",
        skills=["brainstorm"],
    )


@pytest.fixture
def app_config(agent_config):
    config = AppConfig()
    config.default_agent = "TestAgent"
    config.agents = {"TestAgent": agent_config}
    return config


class TestAgent:
    def test_agent_is_abstract(self):
        with pytest.raises(TypeError):
            Agent()

    def test_cliver_agent_is_concrete(self, agent_config, mock_agent_core):
        agent = CliverAgent(name="test", config=agent_config, agent_core=mock_agent_core)
        assert agent.name == "test"
        assert agent.description == "Test agent"

    def test_cliver_agent_from_config(self, agent_config, mock_agent_core):
        agent = CliverAgent.from_config(name="test", config=agent_config, agent_core=mock_agent_core)
        assert isinstance(agent, CliverAgent)
        assert agent.name == "test"


class TestCliverAgentRun:
    @pytest.mark.asyncio
    async def test_run_delegates_to_agent_core(self, agent_config, mock_agent_core):
        agent = CliverAgent(name="test", config=agent_config, agent_core=mock_agent_core)
        await agent.run("hello")
        mock_agent_core.process_user_input.assert_called_once()
        call_kwargs = mock_agent_core.process_user_input.call_args
        assert call_kwargs.kwargs["user_input"] == "hello"
        assert call_kwargs.kwargs["agent_config"] is agent_config

    @pytest.mark.asyncio
    async def test_run_uses_agent_model_as_default(self, agent_config, mock_agent_core):
        agent = CliverAgent(name="test", config=agent_config, agent_core=mock_agent_core)
        await agent.run("hello")
        call_kwargs = mock_agent_core.process_user_input.call_args
        assert call_kwargs.kwargs["model"] == "test-model"

    @pytest.mark.asyncio
    async def test_run_explicit_model_overrides_agent(self, agent_config, mock_agent_core):
        agent = CliverAgent(name="test", config=agent_config, agent_core=mock_agent_core)
        await agent.run("hello", model="override-model")
        call_kwargs = mock_agent_core.process_user_input.call_args
        assert call_kwargs.kwargs["model"] == "override-model"

    @pytest.mark.asyncio
    async def test_run_passes_kwargs_through(self, agent_config, mock_agent_core):
        agent = CliverAgent(name="test", config=agent_config, agent_core=mock_agent_core)
        history = [MagicMock()]
        await agent.run("hello", conversation_history=history, max_iterations=10)
        call_kwargs = mock_agent_core.process_user_input.call_args
        assert call_kwargs.kwargs["conversation_history"] is history
        assert call_kwargs.kwargs["max_iterations"] == 10


class TestAgentFactory:
    def test_create_default_agent(self, app_config, mock_agent_core):
        factory = AgentFactory(config=app_config, agent_core=mock_agent_core)
        agent = factory.create()
        assert isinstance(agent, CliverAgent)
        assert agent.name == "TestAgent"

    def test_create_named_agent(self, app_config, mock_agent_core):
        factory = AgentFactory(config=app_config, agent_core=mock_agent_core)
        agent = factory.create("TestAgent")
        assert isinstance(agent, CliverAgent)
        assert agent.name == "TestAgent"

    def test_create_unknown_type_raises(self, mock_agent_core):
        config = AppConfig()
        config.default_agent = "Bad"
        config.agents = {"Bad": AgentConfig(type="nonexistent")}
        factory = AgentFactory(config=config, agent_core=mock_agent_core)
        with pytest.raises(ValueError, match="Unknown agent type"):
            factory.create("Bad")

    def test_agent_core_property(self, app_config, mock_agent_core):
        factory = AgentFactory(config=app_config, agent_core=mock_agent_core)
        assert factory.agent_core is mock_agent_core


class TestTypeRegistry:
    def test_cliver_type_registered(self):
        assert "cliver" in _AGENT_TYPES
        assert _AGENT_TYPES["cliver"] is CliverAgent

    def test_register_custom_type(self, mock_agent_core):
        class CustomAgent(Agent):
            def __init__(self, **kwargs):
                self.name = "custom"
                self.description = None

            async def run(self, prompt, **kwargs):
                return None

            async def stream(self, prompt, **kwargs):
                yield None

            @classmethod
            def from_config(cls, name, config, agent_core):
                return cls()

        register_agent_type("custom", CustomAgent)
        assert _AGENT_TYPES["custom"] is CustomAgent
        del _AGENT_TYPES["custom"]


class TestAgentConfigType:
    def test_default_type_is_cliver(self):
        config = AgentConfig()
        assert config.type == "cliver"

    def test_explicit_type(self):
        config = AgentConfig(type="external")
        assert config.type == "external"

    def test_existing_fields_unchanged(self):
        config = AgentConfig(
            description="test",
            role="helper",
            system_prompt="be helpful",
            model="gpt-4",
            skills=["brainstorm"],
        )
        assert config.description == "test"
        assert config.role == "helper"
        assert config.model == "gpt-4"


class TestGlobalFactory:
    def test_set_and_get_factory(self, app_config, mock_agent_core):
        factory = AgentFactory(config=app_config, agent_core=mock_agent_core)
        set_agent_factory(factory)
        assert get_agent_factory() is factory
        set_agent_factory(None)

    def test_get_factory_returns_none_by_default(self):
        set_agent_factory(None)
        assert get_agent_factory() is None
