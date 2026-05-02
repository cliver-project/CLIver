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

    def __init__(self, app_config, skill_manager: "SkillManager", agent_name: str = "CLIver"):
        self.app_config = app_config
        self.skill_manager = skill_manager
        self.agent_name = agent_name

    def create(self, agent_config: AgentConfig) -> AgentCore:
        """Create a fresh AgentCore with the specified configuration."""
        from cliver.config import ConfigManager

        config_manager = ConfigManager.__new__(ConfigManager)
        config_manager.config = self.app_config

        llm_models = config_manager.list_llm_models()
        mcp_servers = config_manager.list_mcp_servers_for_mcp_caller()

        # Use agent's model if specified, otherwise fall back to app default
        default_model = agent_config.model
        if not default_model:
            default_llm = config_manager.get_llm_model()
            if default_llm:
                default_model = default_llm.name

        permission_manager = (
            self._build_permissions(agent_config.permissions)
            if agent_config.permissions
            else self._build_auto_allow_permissions()
        )

        core = AgentCore(
            llm_models=llm_models,
            mcp_servers=mcp_servers,
            default_model=default_model,
            agent_name=self.agent_name,
            permission_manager=permission_manager,
            on_permission_prompt=lambda tool, args: ("allow", ""),
            enabled_toolsets=agent_config.tools,
            model_auto_fallback=True,
        )

        return core

    @staticmethod
    def _build_auto_allow_permissions() -> "PermissionManager":
        """Build a PermissionManager that auto-allows everything (headless mode)."""
        from cliver.permissions import PermissionManager, PermissionMode

        pm = PermissionManager()
        pm.set_mode(PermissionMode.YOLO)
        return pm

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
