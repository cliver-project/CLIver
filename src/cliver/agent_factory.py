"""AgentCore factory — creates AgentCore from configuration."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cliver.config import ConfigManager, ModelConfig
    from cliver.events import EventHandler
    from cliver.llm.agent_core import AgentCore
    from cliver.mcp import MCPClient
    from cliver.tool import CLIverTool

logger = logging.getLogger(__name__)


def create_agent_core(
    model_config: "ModelConfig",
    builtin_tools: list["CLIverTool"],
    *,
    mcp_client: "MCPClient | None" = None,
    tool_filter: set[str] | None = None,
    on_event: "EventHandler | None" = None,
    max_consecutive_errors: int = 5,
    user_agent: str | None = "CLIver",
    agent_name: str = "CLIver",
    models: dict | None = None,
    agents: dict | None = None,
) -> "AgentCore":
    """Create an AgentCore from a model config and tool list.

    The builtin system prompt (model inventory, self-awareness, tools) is
    generated automatically and always included.  Callers only need to
    pass persona / extra context via ``system_prompt`` on ``chat()`` / ``stream()``.

    Args:
        model_config: The ModelConfig for the desired model.
        builtin_tools: Full list of available builtin tools.
        mcp_client: Shared MCP client (one per process).
        tool_filter: If set, only include tools whose names are in this set.
        on_event: Optional handler for ToolEvent/InferenceEvent callbacks.
        max_consecutive_errors: Max consecutive tool errors before stopping.
        user_agent: User-Agent header for LLM provider HTTP requests.
        agent_name: Name for the identity section of the system prompt.
        models: All configured models dict (for system prompt inventory).
        agents: All agent profiles dict (for system prompt inventory).

    Returns:
        A configured AgentCore ready for ``chat()`` / ``stream()``.
    """
    from cliver.llm.agent_core import AgentCore
    from cliver.provider.providers import create_provider
    from cliver.system_prompt import build as build_system_prompt

    provider = create_provider(
        api_key=model_config.get_api_key() or "",
        base_url=model_config.get_resolved_url() or "",
        protocol=model_config.get_provider_type(),
        user_agent=user_agent,
    )

    tools = builtin_tools
    if tool_filter is not None:
        tools = [t for t in builtin_tools if t.name in tool_filter]

    builtin_sp = build_system_prompt(
        agent_name=agent_name,
        available_tools={t.name for t in tools},
        models=models,
        agents=agents,
        current_model=model_config.api_model_name,
        current_provider=model_config.get_provider_type(),
    )

    return AgentCore(
        provider=provider,
        model=model_config.api_model_name,
        builtin_tools=tools,
        mcp_client=mcp_client,
        on_event=on_event,
        max_consecutive_errors=max_consecutive_errors,
        builtin_system_prompt=builtin_sp,
    )


def resolve_model(
    model_name: str | None,
    config_manager: "ConfigManager",
) -> "ModelConfig | None":
    """Resolve a model name to a ModelConfig, using the default if None.

    Returns None if no model is configured at all.
    """
    if model_name:
        mc = _resolve_name(model_name, config_manager)
        if mc:
            return mc

    default = config_manager.get_llm_model()
    if default:
        return default

    # Fallback: first available text model
    text_models = config_manager.config.models.get("text", {})
    if text_models:
        return next(iter(text_models.values()))

    return None


def _resolve_name(name: str, config_manager: "ConfigManager") -> "ModelConfig | None":
    all_models = config_manager.all_models()
    if name in all_models:
        return all_models[name]
    matches = [mc for key, mc in all_models.items() if key.endswith(f"/{name}") or key == name]
    if len(matches) == 1:
        return matches[0]
    return None
