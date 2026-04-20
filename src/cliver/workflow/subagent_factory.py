"""
SubAgentFactory — creates isolated AgentCore instances for workflow steps.

Each LLM step in a workflow is a subagent with its own model, tools,
permission scope, and system message. The factory uses the main AppConfig
to resolve model configurations and MCP servers.
"""

import logging
from typing import TYPE_CHECKING, Optional

from cliver.llm import AgentCore
from cliver.workflow.workflow_models import AgentConfig

if TYPE_CHECKING:
    from cliver.permissions import PermissionManager
    from cliver.skill_manager import SkillManager

logger = logging.getLogger(__name__)


class SubAgentFactory:
    """Creates isolated AgentCore instances from workflow agent profiles."""

    def __init__(self, app_config, skill_manager: "SkillManager"):
        self.app_config = app_config
        self.skill_manager = skill_manager

    def create(self, agent_config: AgentConfig) -> AgentCore:
        """Create a fresh AgentCore with the specified configuration."""
        from cliver.config import ConfigManager

        config_manager = ConfigManager.__new__(ConfigManager)
        config_manager.config = self.app_config

        llm_models = config_manager.list_llm_models()
        mcp_servers = config_manager.list_mcp_servers_for_mcp_caller()

        permission_manager = None
        if agent_config.permissions:
            permission_manager = self._build_permissions(agent_config.permissions)

        core = AgentCore(
            llm_models=llm_models,
            mcp_servers=mcp_servers,
            default_model=agent_config.model,
            agent_name=f"subagent-{agent_config.model or 'default'}",
            permission_manager=permission_manager,
            enabled_toolsets=agent_config.tools,
        )

        return core

    @staticmethod
    def _build_permissions(permissions_config) -> Optional["PermissionManager"]:
        """Build a PermissionManager from workflow permission config."""
        from cliver.permissions import PermissionManager, PermissionMode

        pm = PermissionManager()

        if isinstance(permissions_config, dict):
            mode = permissions_config.get("mode")
            if mode:
                pm.set_mode(PermissionMode(mode))

            rules = permissions_config.get("rules", [])
            for rule in rules:
                from cliver.permissions import PermissionAction

                pm.add_rule(
                    tool_pattern=rule.get("tool", ".*"),
                    action=PermissionAction(rule.get("action", "allow")),
                    resource_pattern=rule.get("resource"),
                )

        return pm
