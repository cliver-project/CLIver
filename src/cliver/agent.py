"""Agent — universal entry point for all LLM operations.

All callers (CLI, gateway, tasks, workflows) go through Agent.run()
or Agent.stream() — never AgentCore directly. AgentFactory creates
agents by name from AppConfig.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, AsyncIterator, Dict, Optional, Type

from langchain_core.messages import BaseMessage

if TYPE_CHECKING:
    from cliver.config import AgentConfig, AppConfig
    from cliver.llm import AgentCore

logger = logging.getLogger(__name__)

_AGENT_TYPES: Dict[str, Type[Agent]] = {}


def register_agent_type(type_name: str, cls: Type[Agent]) -> None:
    """Register an agent class for a type name."""
    _AGENT_TYPES[type_name] = cls


class Agent(ABC):
    """Abstract base for all agent types."""

    name: str
    description: Optional[str]

    @abstractmethod
    async def run(self, prompt: str, **kwargs) -> BaseMessage:
        """Execute a prompt and return the response."""

    @abstractmethod
    async def stream(self, prompt: str, **kwargs) -> AsyncIterator[BaseMessage]:
        """Stream a prompt execution. Yields BaseMessage chunks."""

    @classmethod
    @abstractmethod
    def from_config(cls, name: str, config: "AgentConfig", agent_core: "AgentCore") -> "Agent":
        """Construct from AgentConfig and shared AgentCore."""


class CliverAgent(Agent):
    """Agent backed by a shared AgentCore instance."""

    def __init__(self, name: str, config: "AgentConfig", agent_core: "AgentCore"):
        self.name = name
        self.description = config.description
        self.config = config
        self._agent_core = agent_core
        fallback = config.auto_fallback
        self._auto_fallback = fallback if fallback is not None else agent_core.model_auto_fallback

    async def run(self, prompt: str, **kwargs) -> BaseMessage:
        kwargs.setdefault("model", self.config.model)
        kwargs.setdefault("agent_config", self.config)
        kwargs.setdefault("auto_fallback", self._auto_fallback)
        skill_name = kwargs.pop("skill_name", None)
        if skill_name:
            return await self._agent_core.process_skill(
                skill_name=skill_name,
                user_input=prompt,
                **kwargs,
            )
        return await self._agent_core.process_user_input(
            user_input=prompt,
            **kwargs,
        )

    async def stream(self, prompt: str, **kwargs) -> AsyncIterator[BaseMessage]:
        kwargs.setdefault("model", self.config.model)
        kwargs.setdefault("agent_config", self.config)
        kwargs.setdefault("auto_fallback", self._auto_fallback)
        skill_name = kwargs.pop("skill_name", None)
        if skill_name:
            gen = self._agent_core.stream_skill(
                skill_name=skill_name,
                user_input=prompt,
                **kwargs,
            )
        else:
            gen = self._agent_core.stream_user_input(
                user_input=prompt,
                **kwargs,
            )
        async for chunk in gen:
            yield chunk

    @classmethod
    def from_config(cls, name: str, config: "AgentConfig", agent_core: "AgentCore") -> "CliverAgent":
        return cls(name=name, config=config, agent_core=agent_core)


class AgentFactory:
    """Creates Agent instances by name from config.

    Holds shared AppConfig and AgentCore so callers just need a name.
    """

    def __init__(self, config: "AppConfig", agent_core: "AgentCore"):
        self._config = config
        self._agent_core = agent_core

    def create(self, name: Optional[str] = None) -> Agent:
        """Create an agent by name. None = default agent."""
        agent_config = self._config.get_agent(name)
        resolved_name = name or self._config.default_agent or "CLIver"
        agent_type = agent_config.type

        cls = _AGENT_TYPES.get(agent_type)
        if cls is None:
            raise ValueError(f"Unknown agent type: {agent_type!r}. Available: {list(_AGENT_TYPES.keys())}")
        return cls.from_config(
            name=resolved_name,
            config=agent_config,
            agent_core=self._agent_core,
        )

    @property
    def agent_core(self) -> "AgentCore":
        """Access the underlying AgentCore (for direct use by internals)."""
        return self._agent_core


register_agent_type("cliver", CliverAgent)
