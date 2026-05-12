"""AgentFactory — creates and caches Agent instances by name from config."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, Type

from cliver.agent import Agent
from cliver.agents.claude_agent import ClaudeAgent
from cliver.agents.cli_agent import CliAgent
from cliver.agents.cliver_agent import CliverAgent
from cliver.agents.gemini_agent import GeminiAgent
from cliver.agents.opencode_agent import OpenCodeAgent

if TYPE_CHECKING:
    from cliver.config import AgentConfig, AppConfig
    from cliver.llm.llm import AgentCore

logger = logging.getLogger(__name__)

AGENT_REGISTRY: Dict[str, Type[Agent]] = {
    "cliver": CliverAgent,
    "claude": ClaudeAgent,
    "gemini": GeminiAgent,
    "opencode": OpenCodeAgent,
}


class AgentFactory:
    """Creates and caches Agent instances by name from config."""

    def __init__(self, config: "AppConfig", agent_core: "AgentCore"):
        self._config = config
        self._agent_core = agent_core
        self._agents: Dict[str, Agent] = {}

        from cliver.llm.rate_limiter import RateLimiter, parse_period

        self._rate_limiter = RateLimiter()
        for _name, provider in self._config.providers.items():
            if provider.rate_limit:
                key = f"{provider.api_url}|{provider.api_key or ''}"
                period_s = parse_period(provider.rate_limit.period)
                self._rate_limiter.configure(
                    key,
                    requests=provider.rate_limit.requests,
                    period_seconds=period_s,
                    margin=provider.rate_limit.margin,
                )

    def create(self, name: str = None) -> Agent:
        name = name or self._config.default_agent or "default"

        if name in self._agents:
            return self._agents[name]

        agent_config = self._resolve_agent_config(name)
        model_config, provider_config = self._resolve_model(agent_config.model)

        agent_cls = AGENT_REGISTRY.get(agent_config.type, CliAgent)

        kwargs = dict(
            name=name,
            config=agent_config,
            model_config=model_config,
            provider_config=provider_config,
            rate_limiter=self._rate_limiter,
        )
        if agent_config.type == "cliver":
            kwargs["agent_core"] = self._agent_core

        agent = agent_cls(**kwargs)
        self._agents[name] = agent
        return agent

    def _resolve_agent_config(self, name: str) -> "AgentConfig":
        from cliver.config import AgentConfig

        agents = self._config.agents or {}
        if name in agents:
            return agents[name]
        return AgentConfig(type="cliver", model=self._config.default_model)

    def _resolve_model(self, model_name: str = None):
        if not model_name:
            model_name = self._config.default_model
        if not model_name:
            return None, None
        model_config = self._config.models.get(model_name)
        provider_config = (
            model_config._provider_config if model_config and hasattr(model_config, "_provider_config") else None
        )
        return model_config, provider_config

    async def cleanup_all(self) -> None:
        for agent in self._agents.values():
            await agent.cleanup()
        self._agents.clear()
