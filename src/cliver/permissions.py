"""
Permission system for CLIver tool execution.

Provides resource-aware permission checking with layered rules:
- Persistent rules from cliver-settings.yaml (global + local project)
- Session-scoped grants (in-memory, cleared on exit)
- Task-scoped permissions (pushed/popped during workflow execution)

Each tool has an action kind (safe/read/write/execute/fetch) and resource type
(path/url/command/none). Rules use regex for tool matching and fnmatch globs
for resource matching.
"""

import logging
import re
from enum import Enum
from fnmatch import fnmatch
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from pydantic import BaseModel

logger = logging.getLogger(__name__)

SETTINGS_FILENAME = "cliver-settings.yaml"


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ActionKind(str, Enum):
    """Side-effect level of a tool."""

    SAFE = "safe"
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    FETCH = "fetch"


class ResourceType(str, Enum):
    """Type of resource a tool operates on."""

    NONE = "none"
    PATH = "path"
    URL = "url"
    COMMAND = "command"


class PermissionMode(str, Enum):
    """Global permission mode controlling baseline behavior."""

    DEFAULT = "default"
    AUTO_EDIT = "auto-edit"
    YOLO = "yolo"


class PermissionAction(str, Enum):
    ALLOW = "allow"
    DENY = "deny"


class PermissionDecision(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"


# ---------------------------------------------------------------------------
# Tool metadata
# ---------------------------------------------------------------------------


class ToolMeta:
    """Metadata about a tool's permission characteristics."""

    def __init__(
        self,
        action_kind: ActionKind,
        resource_type: ResourceType,
        resource_param: Optional[str] = None,
    ):
        self.action_kind = action_kind
        self.resource_type = resource_type
        self.resource_param = resource_param


TOOL_META_REGISTRY: Dict[str, ToolMeta] = {
    # Safe tools (no side effects)
    "skill": ToolMeta(ActionKind.SAFE, ResourceType.NONE),
    "todo_read": ToolMeta(ActionKind.SAFE, ResourceType.NONE),
    "todo_write": ToolMeta(ActionKind.SAFE, ResourceType.NONE),
    "memory_read": ToolMeta(ActionKind.SAFE, ResourceType.NONE),
    "memory_write": ToolMeta(ActionKind.SAFE, ResourceType.NONE),
    "identity_update": ToolMeta(ActionKind.SAFE, ResourceType.NONE),
    "ask_user_question": ToolMeta(ActionKind.SAFE, ResourceType.NONE),
    # Read tools
    "read_file": ToolMeta(ActionKind.READ, ResourceType.PATH, "file_path"),
    "list_directory": ToolMeta(ActionKind.READ, ResourceType.PATH, "path"),
    "grep_search": ToolMeta(ActionKind.READ, ResourceType.PATH, "path"),
    # Write tools
    "write_file": ToolMeta(ActionKind.WRITE, ResourceType.PATH, "file_path"),
    "create_workflow": ToolMeta(ActionKind.WRITE, ResourceType.NONE),
    # Execute tools
    "run_shell_command": ToolMeta(ActionKind.EXECUTE, ResourceType.COMMAND, "command"),
    "docker_run": ToolMeta(ActionKind.EXECUTE, ResourceType.COMMAND, "image"),
    "setup_docker": ToolMeta(ActionKind.EXECUTE, ResourceType.NONE),
    # Fetch tools
    "web_fetch": ToolMeta(ActionKind.FETCH, ResourceType.URL, "url"),
    "web_search": ToolMeta(ActionKind.FETCH, ResourceType.NONE),
}


def get_tool_meta(tool_name: str) -> ToolMeta:
    """Get metadata for a builtin tool by bare name.

    For MCP tools (not in registry), defaults to EXECUTE (safest).
    Callers should strip the 'server#' prefix before calling.
    """
    return TOOL_META_REGISTRY.get(tool_name, ToolMeta(ActionKind.EXECUTE, ResourceType.NONE))


# ---------------------------------------------------------------------------
# Rule models
# ---------------------------------------------------------------------------


class PermissionRule(BaseModel):
    """A single permission rule with regex tool matching and glob resource matching."""

    tool: str  # Regex pattern for tool identity
    resource: Optional[str] = None  # fnmatch glob pattern for resource
    action: PermissionAction

    def matches_tool(self, tool_identity: str) -> bool:
        """Match tool identity using regex (re.fullmatch)."""
        try:
            return bool(re.fullmatch(self.tool, tool_identity))
        except re.error:
            return self.tool == tool_identity

    def matches_resource(self, resource: Optional[str]) -> bool:
        """Match resource using fnmatch glob. No resource pattern = matches all."""
        if self.resource is None:
            return True
        if resource is None:
            return False
        return fnmatch(resource, self.resource)

    def matches(self, tool_identity: str, resource: Optional[str]) -> bool:
        """Check if this rule matches a tool call."""
        return self.matches_tool(tool_identity) and self.matches_resource(resource)


class TaskPermissions(BaseModel):
    """Permission overrides for a task/workflow/step execution."""

    mode: Optional[PermissionMode] = None
    rules: List[PermissionRule] = []


# ---------------------------------------------------------------------------
# Permission manager
# ---------------------------------------------------------------------------


class PermissionManager:
    """Evaluates tool calls against layered permission rules.

    Resolution order:
    1. Deny rules (all layers) — deny always wins
    2. Task-scope permissions (push/pop stack)
    3. Session grants (in-memory)
    4. Persistent rules (from config files)
    5. Mode-based defaults
    6. Fallback → ASK
    """

    def __init__(
        self,
        global_config_dir: Optional[Path] = None,
        local_dir: Optional[Path] = None,
    ):
        self.mode: PermissionMode = PermissionMode.DEFAULT
        self.rules: List[PermissionRule] = []
        self._session_grants: Dict[str, PermissionAction] = {}
        self._task_stack: List[TaskPermissions] = []
        if global_config_dir is not None:
            self._load_settings(global_config_dir, local_dir)

    def _load_settings(self, global_dir: Path, local_dir: Optional[Path]) -> None:
        """Load and merge settings files in priority order.

        Global is loaded first, then local overrides (last mode wins,
        rules accumulate).
        """
        for d in [global_dir, local_dir]:
            if d is None:
                continue
            path = Path(d) / SETTINGS_FILENAME
            if not path.exists():
                continue
            try:
                data = yaml.safe_load(path.read_text()) or {}
            except Exception as e:
                logger.warning(f"Failed to load {path}: {e}")
                continue
            if "permission_mode" in data:
                try:
                    self.mode = PermissionMode(data["permission_mode"])
                except ValueError:
                    logger.warning(f"Invalid permission_mode '{data['permission_mode']}' in {path}")
            for rule_dict in data.get("permissions", []):
                try:
                    self.rules.append(PermissionRule(**rule_dict))
                except Exception as e:
                    logger.warning(f"Invalid permission rule in {path}: {rule_dict} — {e}")

    def check(self, tool_identity: str, args: dict) -> PermissionDecision:
        """Check if a tool call is allowed, denied, or needs prompting.

        Args:
            tool_identity: Full tool identity — "read_file" for builtins,
                          "github#create_issue" for MCP server tools.
            args: Tool call arguments dict.
        """
        bare_name = tool_identity.split("#")[-1] if "#" in tool_identity else tool_identity
        meta = get_tool_meta(bare_name)
        resource = self._extract_resource(meta, args)

        # 1. Deny rules always win (all layers)
        if self._matches_deny(self.rules, tool_identity, resource):
            return PermissionDecision.DENY
        if self._matches_deny(self._all_task_rules(), tool_identity, resource):
            return PermissionDecision.DENY

        # 2. Task-scope allow rules
        if self._matches_allow(self._all_task_rules(), tool_identity, resource):
            return PermissionDecision.ALLOW

        # 3. Session grants (keyed by full tool_identity)
        if tool_identity in self._session_grants:
            if self._session_grants[tool_identity] == PermissionAction.ALLOW:
                return PermissionDecision.ALLOW
            return PermissionDecision.DENY

        # 4. Persistent allow rules from config
        if self._matches_allow(self.rules, tool_identity, resource):
            return PermissionDecision.ALLOW

        # 5. Mode-based default
        effective_mode = self._effective_mode()
        if meta.action_kind == ActionKind.SAFE:
            return PermissionDecision.ALLOW
        if effective_mode == PermissionMode.YOLO:
            return PermissionDecision.ALLOW
        if effective_mode == PermissionMode.AUTO_EDIT:
            if meta.action_kind in (ActionKind.READ, ActionKind.WRITE):
                return PermissionDecision.ALLOW

        # 6. Must ask user
        return PermissionDecision.ASK

    # --- Rule matching (with constrained-allow semantics) ---

    @staticmethod
    def _matches_deny(rules: List[PermissionRule], tool_identity: str, resource: Optional[str]) -> bool:
        return any(r.action == PermissionAction.DENY and r.matches(tool_identity, resource) for r in rules)

    @staticmethod
    def _matches_allow(rules: List[PermissionRule], tool_identity: str, resource: Optional[str]) -> bool:
        """Check allow rules with constrained-allow semantics.

        If a tool has specific resource-constrained rules (not wildcards),
        only matching resources are allowed. Wildcards cannot override
        tool-specific resource constraints.
        """
        has_specific_resource_rules = any(
            r.action == PermissionAction.ALLOW
            and r.resource is not None
            and r.matches_tool(tool_identity)
            and r.tool != ".*"
            and not r.tool.endswith("#.*")
            for r in rules
        )

        if has_specific_resource_rules:
            # Only specific resource matches count
            return any(
                r.action == PermissionAction.ALLOW and r.resource is not None and r.matches(tool_identity, resource)
                for r in rules
            )
        else:
            return any(r.action == PermissionAction.ALLOW and r.matches(tool_identity, resource) for r in rules)

    # --- Mode resolution ---

    def _effective_mode(self) -> PermissionMode:
        """Get the effective mode, considering task-scope overrides."""
        for task_perms in reversed(self._task_stack):
            if task_perms.mode is not None:
                return task_perms.mode
        return self.mode

    # --- Resource extraction ---

    @staticmethod
    def _extract_resource(meta: ToolMeta, args: dict) -> Optional[str]:
        """Extract the resource value from tool args based on metadata."""
        if meta.resource_param and meta.resource_param in args:
            return str(args[meta.resource_param])
        return None

    # --- Task scope management ---

    def push_task_scope(self, permissions: TaskPermissions):
        """Push a task/workflow/step permission scope onto the stack."""
        self._task_stack.append(permissions)

    def pop_task_scope(self):
        """Pop the innermost task permission scope."""
        if self._task_stack:
            self._task_stack.pop()

    def _all_task_rules(self) -> List[PermissionRule]:
        """Collect all task-scope rules from all stack levels."""
        return [r for tp in self._task_stack for r in tp.rules]

    # --- Session grants ---

    def grant_session(self, tool_identity: str, action: PermissionAction):
        """Grant or deny a tool for the rest of the session."""
        self._session_grants[tool_identity] = action

    def set_mode(self, mode: PermissionMode):
        """Change permission mode for this session."""
        self.mode = mode

    def clear_session_grants(self):
        """Clear all session-level grants."""
        self._session_grants.clear()
