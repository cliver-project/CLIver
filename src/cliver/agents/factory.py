"""AgentFactory — creates and caches Agent instances by name from config.

Each agent is a CliverAgent configured with a model, role, system prompt,
and skills — composition over inheritance.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, Dict

from cliver.agent import Agent
from cliver.agents.cliver_agent import CliverAgent

if TYPE_CHECKING:
    from cliver.config import AgentConfig, AppConfig
    from cliver.llm.new_agent import AgentCore as NewAgentCore

logger = logging.getLogger(__name__)


class AgentFactory:
    """Creates and caches Agent instances by name from config."""

    def __init__(
        self,
        config: "AppConfig",
        agent_core_factory: Callable[[str], "NewAgentCore"],
    ):
        self._config = config
        self._agent_core_factory = agent_core_factory
        self._agents: Dict[str, Agent] = {}

    def create(self, name: str = None) -> Agent:
        name = name or self._config.default_agent or "default"

        if name in self._agents:
            return self._agents[name]

        agent_config = self._resolve_agent_config(name)
        model_name = agent_config.model or self._config.default_model
        agent_core = self._agent_core_factory(model_name)

        agent = CliverAgent(
            name=name,
            config=agent_config,
            agent_core=agent_core,
        )

        self._agents[name] = agent
        return agent

    def _resolve_agent_config(self, name: str) -> "AgentConfig":
        from cliver.config import AgentConfig

        agents = self._config.agents or {}
        if name in agents:
            return agents[name]
        return AgentConfig(name="default", model=self._config.default_model)

    async def cleanup_all(self) -> None:
        for agent in self._agents.values():
            await agent.cleanup()
        self._agents.clear()
