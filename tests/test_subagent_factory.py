"""Tests for SubAgentFactory — creates isolated AgentCore per step."""

import pytest
from unittest.mock import MagicMock, patch

from cliver.workflow.subagent_factory import SubAgentFactory
from cliver.workflow.workflow_models import AgentConfig


@pytest.fixture
def mock_app_config():
    config = MagicMock()
    model_config = MagicMock()
    model_config.name = "qwen"
    model_config.provider = "openai"
    model_config._provider_config = None
    config.models = {"qwen": model_config}
    return config


@pytest.fixture
def mock_skill_manager():
    return MagicMock()


class TestSubAgentFactory:
    def test_create_returns_agent_core(self, mock_app_config, mock_skill_manager):
        factory = SubAgentFactory(mock_app_config, mock_skill_manager)
        agent_config = AgentConfig(model="qwen")

        with patch("cliver.workflow.subagent_factory.AgentCore") as MockCore:
            MockCore.return_value = MagicMock()
            core = factory.create(agent_config)
            assert core is not None
            MockCore.assert_called_once()

    def test_create_passes_model(self, mock_app_config, mock_skill_manager):
        factory = SubAgentFactory(mock_app_config, mock_skill_manager)
        agent_config = AgentConfig(model="qwen")

        with patch("cliver.workflow.subagent_factory.AgentCore") as MockCore:
            MockCore.return_value = MagicMock()
            factory.create(agent_config)
            call_kwargs = MockCore.call_args[1]
            assert call_kwargs["default_model"] == "qwen"

    def test_create_filters_tools(self, mock_app_config, mock_skill_manager):
        factory = SubAgentFactory(mock_app_config, mock_skill_manager)
        agent_config = AgentConfig(model="qwen", tools=["read_file", "write_file"])

        with patch("cliver.workflow.subagent_factory.AgentCore") as MockCore:
            MockCore.return_value = MagicMock()
            factory.create(agent_config)
            call_kwargs = MockCore.call_args[1]
            assert call_kwargs["enabled_toolsets"] == ["read_file", "write_file"]

    def test_create_no_tools_means_all(self, mock_app_config, mock_skill_manager):
        factory = SubAgentFactory(mock_app_config, mock_skill_manager)
        agent_config = AgentConfig(model="qwen")

        with patch("cliver.workflow.subagent_factory.AgentCore") as MockCore:
            MockCore.return_value = MagicMock()
            factory.create(agent_config)
            call_kwargs = MockCore.call_args[1]
            assert call_kwargs.get("enabled_toolsets") is None
