"""Tests for the permission system."""

from pathlib import Path

import pytest
import yaml

from cliver.permissions import (
    ActionKind,
    PermissionAction,
    PermissionDecision,
    PermissionManager,
    PermissionMode,
    PermissionRule,
    ResourceType,
    TaskPermissions,
    get_tool_meta,
)

# ---------------------------------------------------------------------------
# PermissionRule matching
# ---------------------------------------------------------------------------


class TestPermissionRuleMatching:
    """Test regex tool matching and fnmatch resource matching."""

    def test_exact_builtin_tool_match(self):
        rule = PermissionRule(tool="read_file", action=PermissionAction.ALLOW)
        assert rule.matches_tool("read_file")
        assert not rule.matches_tool("write_file")

    def test_mcp_tool_exact_match(self):
        rule = PermissionRule(tool="github#create_issue", action=PermissionAction.ALLOW)
        assert rule.matches_tool("github#create_issue")
        assert not rule.matches_tool("github#list_repos")

    def test_mcp_server_wildcard(self):
        rule = PermissionRule(tool="github#.*", action=PermissionAction.ALLOW)
        assert rule.matches_tool("github#create_issue")
        assert rule.matches_tool("github#list_repos")
        assert not rule.matches_tool("ocp#get_pods")

    def test_mcp_server_specific_tools(self):
        rule = PermissionRule(tool="ocp#(get_pods|get_nodes|describe_.*)", action=PermissionAction.ALLOW)
        assert rule.matches_tool("ocp#get_pods")
        assert rule.matches_tool("ocp#get_nodes")
        assert rule.matches_tool("ocp#describe_pod")
        assert not rule.matches_tool("ocp#delete_pod")

    def test_catch_all_wildcard(self):
        rule = PermissionRule(tool=".*", action=PermissionAction.ALLOW)
        assert rule.matches_tool("read_file")
        assert rule.matches_tool("github#create_issue")

    def test_all_mcp_tools_wildcard(self):
        rule = PermissionRule(tool=".*#.*", action=PermissionAction.ALLOW)
        assert rule.matches_tool("github#create_issue")
        assert rule.matches_tool("ocp#get_pods")
        assert not rule.matches_tool("read_file")  # No '#' separator

    def test_resource_glob_path(self):
        rule = PermissionRule(tool="read_file", resource="/data/reports/**", action=PermissionAction.ALLOW)
        assert rule.matches_resource("/data/reports/q1.csv")
        assert rule.matches_resource("/data/reports/2024/jan.csv")
        assert not rule.matches_resource("/etc/passwd")

    def test_resource_glob_url(self):
        rule = PermissionRule(tool="web_fetch", resource="https://docs.python.org/**", action=PermissionAction.ALLOW)
        assert rule.matches_resource("https://docs.python.org/3/library/os.html")
        assert not rule.matches_resource("https://evil.com/malware")

    def test_resource_glob_command(self):
        rule = PermissionRule(tool="run_shell_command", resource="git *", action=PermissionAction.ALLOW)
        assert rule.matches_resource("git status")
        assert rule.matches_resource("git diff HEAD")
        assert not rule.matches_resource("rm -rf /")

    def test_no_resource_pattern_matches_all(self):
        rule = PermissionRule(tool="read_file", action=PermissionAction.ALLOW)
        assert rule.matches_resource(None)
        assert rule.matches_resource("/any/path")

    def test_resource_pattern_with_none_resource(self):
        rule = PermissionRule(tool="read_file", resource="/data/**", action=PermissionAction.ALLOW)
        assert not rule.matches_resource(None)

    def test_full_match(self):
        rule = PermissionRule(tool="read_file", resource="/data/**", action=PermissionAction.ALLOW)
        assert rule.matches("read_file", "/data/file.txt")
        assert not rule.matches("read_file", "/etc/passwd")
        assert not rule.matches("write_file", "/data/file.txt")

    def test_regex_fullmatch_not_partial(self):
        """Ensure regex uses fullmatch, not search/match."""
        rule = PermissionRule(tool="read", action=PermissionAction.ALLOW)
        assert rule.matches_tool("read")
        assert not rule.matches_tool("read_file")  # fullmatch, not prefix


# ---------------------------------------------------------------------------
# ToolMeta registry
# ---------------------------------------------------------------------------


class TestToolMeta:
    def test_builtin_tool_lookup(self):
        meta = get_tool_meta("read_file")
        assert meta.action_kind == ActionKind.READ
        assert meta.resource_type == ResourceType.PATH
        assert meta.resource_param == "file_path"

    def test_safe_tool(self):
        meta = get_tool_meta("skill")
        assert meta.action_kind == ActionKind.SAFE
        assert meta.resource_type == ResourceType.NONE

    def test_execute_tool(self):
        meta = get_tool_meta("run_shell_command")
        assert meta.action_kind == ActionKind.EXECUTE
        assert meta.resource_type == ResourceType.COMMAND
        assert meta.resource_param == "command"

    def test_fetch_tool(self):
        meta = get_tool_meta("web_fetch")
        assert meta.action_kind == ActionKind.FETCH
        assert meta.resource_type == ResourceType.URL

    def test_unknown_tool_defaults_to_execute(self):
        meta = get_tool_meta("unknown_mcp_tool")
        assert meta.action_kind == ActionKind.EXECUTE
        assert meta.resource_type == ResourceType.NONE


# ---------------------------------------------------------------------------
# PermissionManager.check() — mode-based defaults
# ---------------------------------------------------------------------------


class TestPermissionManagerModes:
    def _make_manager(self, mode: PermissionMode) -> PermissionManager:
        pm = PermissionManager.__new__(PermissionManager)
        pm.mode = mode
        pm.rules = []
        pm._session_grants = {}
        pm._task_stack = []
        return pm

    def test_safe_tools_always_allowed(self):
        """Safe tools are auto-allowed in all modes."""
        for mode in PermissionMode:
            pm = self._make_manager(mode)
            assert pm.check("skill", {}) == PermissionDecision.ALLOW
            assert pm.check("todo_read", {}) == PermissionDecision.ALLOW

    def test_default_mode_asks_for_non_safe(self):
        pm = self._make_manager(PermissionMode.DEFAULT)
        assert pm.check("read_file", {"file_path": "/tmp/x"}) == PermissionDecision.ASK
        assert pm.check("write_file", {"file_path": "/tmp/x"}) == PermissionDecision.ASK
        assert pm.check("run_shell_command", {"command": "ls"}) == PermissionDecision.ASK
        assert pm.check("web_fetch", {"url": "https://x.com"}) == PermissionDecision.ASK

    def test_auto_edit_allows_read_write(self):
        pm = self._make_manager(PermissionMode.AUTO_EDIT)
        assert pm.check("read_file", {"file_path": "/tmp/x"}) == PermissionDecision.ALLOW
        assert pm.check("write_file", {"file_path": "/tmp/x"}) == PermissionDecision.ALLOW
        assert pm.check("run_shell_command", {"command": "ls"}) == PermissionDecision.ASK
        assert pm.check("web_fetch", {"url": "https://x.com"}) == PermissionDecision.ASK

    def test_yolo_allows_everything(self):
        pm = self._make_manager(PermissionMode.YOLO)
        assert pm.check("read_file", {"file_path": "/tmp/x"}) == PermissionDecision.ALLOW
        assert pm.check("write_file", {"file_path": "/tmp/x"}) == PermissionDecision.ALLOW
        assert pm.check("run_shell_command", {"command": "rm -rf /"}) == PermissionDecision.ALLOW
        assert pm.check("web_fetch", {"url": "https://x.com"}) == PermissionDecision.ALLOW

    def test_mcp_tool_defaults_to_execute(self):
        pm = self._make_manager(PermissionMode.DEFAULT)
        assert pm.check("github#create_issue", {"title": "bug"}) == PermissionDecision.ASK

    def test_mcp_tool_yolo_allows(self):
        pm = self._make_manager(PermissionMode.YOLO)
        assert pm.check("github#create_issue", {"title": "bug"}) == PermissionDecision.ALLOW


# ---------------------------------------------------------------------------
# PermissionManager.check() — persistent rules
# ---------------------------------------------------------------------------


class TestPermissionManagerRules:
    def _make_manager(self, rules: list, mode=PermissionMode.DEFAULT) -> PermissionManager:
        pm = PermissionManager.__new__(PermissionManager)
        pm.mode = mode
        pm.rules = [PermissionRule(**r) for r in rules]
        pm._session_grants = {}
        pm._task_stack = []
        return pm

    def test_allow_rule_matches(self):
        pm = self._make_manager(
            [
                {"tool": "read_file", "resource": "/data/**", "action": "allow"},
            ]
        )
        assert pm.check("read_file", {"file_path": "/data/x.csv"}) == PermissionDecision.ALLOW

    def test_deny_rule_wins_over_allow(self):
        pm = self._make_manager(
            [
                {"tool": "read_file", "action": "allow"},
                {"tool": "read_file", "resource": "/etc/**", "action": "deny"},
            ]
        )
        assert pm.check("read_file", {"file_path": "/etc/passwd"}) == PermissionDecision.DENY
        assert pm.check("read_file", {"file_path": "/tmp/x"}) == PermissionDecision.ALLOW

    def test_mcp_server_wildcard_allow(self):
        pm = self._make_manager(
            [
                {"tool": "github#.*", "action": "allow"},
            ]
        )
        assert pm.check("github#create_issue", {}) == PermissionDecision.ALLOW
        assert pm.check("github#list_repos", {}) == PermissionDecision.ALLOW
        assert pm.check("ocp#get_pods", {}) == PermissionDecision.ASK

    def test_catch_all_wildcard(self):
        pm = self._make_manager(
            [
                {"tool": ".*", "action": "allow"},
            ]
        )
        assert pm.check("read_file", {"file_path": "/any"}) == PermissionDecision.ALLOW
        assert pm.check("github#create_issue", {}) == PermissionDecision.ALLOW

    def test_constrained_allow_denies_unmatched_resource(self):
        """When a tool has specific resource rules, non-matching resources are denied."""
        pm = self._make_manager(
            [
                {"tool": "read_file", "resource": "/data/reports/**", "action": "allow"},
                {"tool": ".*", "action": "allow"},
            ]
        )
        # Matches resource constraint
        assert pm.check("read_file", {"file_path": "/data/reports/q1.csv"}) == PermissionDecision.ALLOW
        # Does NOT match — constrained, so wildcard cannot override
        assert pm.check("read_file", {"file_path": "/etc/passwd"}) == PermissionDecision.ASK
        # Other tools still allowed by wildcard
        assert pm.check("run_shell_command", {"command": "ls"}) == PermissionDecision.ALLOW

    def test_constrained_allow_multiple_resources(self):
        pm = self._make_manager(
            [
                {"tool": "web_fetch", "resource": "https://api.github.com/**", "action": "allow"},
                {"tool": "web_fetch", "resource": "https://docs.python.org/**", "action": "allow"},
                {"tool": ".*", "action": "allow"},
            ]
        )
        assert pm.check("web_fetch", {"url": "https://api.github.com/repos"}) == PermissionDecision.ALLOW
        assert pm.check("web_fetch", {"url": "https://docs.python.org/3/"}) == PermissionDecision.ALLOW
        assert pm.check("web_fetch", {"url": "https://evil.com"}) == PermissionDecision.ASK

    def test_server_wide_wildcard_not_treated_as_specific(self):
        """server#.* rules should not trigger constrained-allow for individual tools."""
        pm = self._make_manager(
            [
                {"tool": "github#.*", "action": "allow"},
            ]
        )
        # This should allow all github tools without resource constraints
        assert pm.check("github#create_issue", {}) == PermissionDecision.ALLOW


# ---------------------------------------------------------------------------
# Session grants
# ---------------------------------------------------------------------------


class TestSessionGrants:
    def _make_manager(self) -> PermissionManager:
        pm = PermissionManager.__new__(PermissionManager)
        pm.mode = PermissionMode.DEFAULT
        pm.rules = []
        pm._session_grants = {}
        pm._task_stack = []
        return pm

    def test_session_grant_allow(self):
        pm = self._make_manager()
        pm.grant_session("read_file", PermissionAction.ALLOW)
        assert pm.check("read_file", {"file_path": "/any"}) == PermissionDecision.ALLOW

    def test_session_grant_deny(self):
        pm = self._make_manager()
        pm.grant_session("run_shell_command", PermissionAction.DENY)
        assert pm.check("run_shell_command", {"command": "ls"}) == PermissionDecision.DENY

    def test_session_grant_mcp_tool(self):
        pm = self._make_manager()
        pm.grant_session("github#create_issue", PermissionAction.ALLOW)
        assert pm.check("github#create_issue", {}) == PermissionDecision.ALLOW
        # Different tool from same server still asks
        assert pm.check("github#delete_repo", {}) == PermissionDecision.ASK

    def test_clear_session_grants(self):
        pm = self._make_manager()
        pm.grant_session("read_file", PermissionAction.ALLOW)
        pm.clear_session_grants()
        assert pm.check("read_file", {"file_path": "/tmp/x"}) == PermissionDecision.ASK

    def test_deny_rule_wins_over_session_grant(self):
        """Deny rules from config always win over session grants."""
        pm = self._make_manager()
        pm.rules = [PermissionRule(tool="read_file", resource="/etc/**", action=PermissionAction.DENY)]
        pm.grant_session("read_file", PermissionAction.ALLOW)
        assert pm.check("read_file", {"file_path": "/etc/passwd"}) == PermissionDecision.DENY


# ---------------------------------------------------------------------------
# Task-scope permissions
# ---------------------------------------------------------------------------


class TestTaskScopePermissions:
    def _make_manager(self) -> PermissionManager:
        pm = PermissionManager.__new__(PermissionManager)
        pm.mode = PermissionMode.DEFAULT
        pm.rules = []
        pm._session_grants = {}
        pm._task_stack = []
        return pm

    def test_task_scope_allow(self):
        pm = self._make_manager()
        pm.push_task_scope(
            TaskPermissions(
                rules=[
                    PermissionRule(tool="read_file", action=PermissionAction.ALLOW),
                ]
            )
        )
        assert pm.check("read_file", {"file_path": "/any"}) == PermissionDecision.ALLOW
        pm.pop_task_scope()
        assert pm.check("read_file", {"file_path": "/any"}) == PermissionDecision.ASK

    def test_task_scope_mode_override(self):
        pm = self._make_manager()
        pm.push_task_scope(TaskPermissions(mode=PermissionMode.YOLO))
        assert pm.check("run_shell_command", {"command": "rm -rf /"}) == PermissionDecision.ALLOW
        pm.pop_task_scope()
        assert pm.check("run_shell_command", {"command": "rm -rf /"}) == PermissionDecision.ASK

    def test_nested_task_scopes_accumulate(self):
        pm = self._make_manager()
        pm.push_task_scope(
            TaskPermissions(
                rules=[
                    PermissionRule(tool="read_file", action=PermissionAction.ALLOW),
                ]
            )
        )
        pm.push_task_scope(
            TaskPermissions(
                rules=[
                    PermissionRule(tool="write_file", action=PermissionAction.ALLOW),
                ]
            )
        )
        # Both should be allowed
        assert pm.check("read_file", {"file_path": "/any"}) == PermissionDecision.ALLOW
        assert pm.check("write_file", {"file_path": "/any"}) == PermissionDecision.ALLOW
        # Pop inner scope
        pm.pop_task_scope()
        assert pm.check("read_file", {"file_path": "/any"}) == PermissionDecision.ALLOW
        assert pm.check("write_file", {"file_path": "/any"}) == PermissionDecision.ASK

    def test_innermost_mode_wins(self):
        pm = self._make_manager()
        pm.push_task_scope(TaskPermissions(mode=PermissionMode.YOLO))
        pm.push_task_scope(TaskPermissions(mode=PermissionMode.AUTO_EDIT))
        # Inner AUTO_EDIT wins over outer YOLO
        assert pm.check("run_shell_command", {"command": "ls"}) == PermissionDecision.ASK
        assert pm.check("read_file", {"file_path": "/any"}) == PermissionDecision.ALLOW

    def test_task_scope_deny_wins(self):
        pm = self._make_manager()
        pm.push_task_scope(
            TaskPermissions(
                rules=[
                    PermissionRule(tool=".*", action=PermissionAction.ALLOW),
                    PermissionRule(tool="run_shell_command", resource="rm *", action=PermissionAction.DENY),
                ]
            )
        )
        assert pm.check("read_file", {"file_path": "/any"}) == PermissionDecision.ALLOW
        assert pm.check("run_shell_command", {"command": "rm -rf /"}) == PermissionDecision.DENY

    def test_task_scope_constrained_allow(self):
        """Constrained-allow applies to task rules too."""
        pm = self._make_manager()
        pm.push_task_scope(
            TaskPermissions(
                rules=[
                    PermissionRule(tool="read_file", resource="/data/**", action=PermissionAction.ALLOW),
                    PermissionRule(tool=".*", action=PermissionAction.ALLOW),
                ]
            )
        )
        assert pm.check("read_file", {"file_path": "/data/x.csv"}) == PermissionDecision.ALLOW
        assert pm.check("read_file", {"file_path": "/etc/passwd"}) == PermissionDecision.ASK
        assert pm.check("write_file", {"file_path": "/any"}) == PermissionDecision.ALLOW

    def test_task_grant_all(self):
        """A task with mode: yolo grants everything."""
        pm = self._make_manager()
        pm.push_task_scope(TaskPermissions(mode=PermissionMode.YOLO))
        assert pm.check("run_shell_command", {"command": "anything"}) == PermissionDecision.ALLOW
        assert pm.check("github#delete_repo", {}) == PermissionDecision.ALLOW

    def test_pop_empty_stack_is_safe(self):
        pm = self._make_manager()
        pm.pop_task_scope()  # Should not raise


# ---------------------------------------------------------------------------
# set_mode
# ---------------------------------------------------------------------------


class TestSetMode:
    def test_set_mode(self):
        pm = PermissionManager.__new__(PermissionManager)
        pm.mode = PermissionMode.DEFAULT
        pm.rules = []
        pm._session_grants = {}
        pm._task_stack = []

        pm.set_mode(PermissionMode.YOLO)
        assert pm.check("run_shell_command", {"command": "ls"}) == PermissionDecision.ALLOW

        pm.set_mode(PermissionMode.DEFAULT)
        assert pm.check("run_shell_command", {"command": "ls"}) == PermissionDecision.ASK


# ---------------------------------------------------------------------------
# AgentCore integration
# ---------------------------------------------------------------------------


class TestAgentCorePermissionGate:
    """Test that PermissionManager integrates with AgentCore._check_permission."""

    def _make_executor(self, pm, on_prompt=None):
        from cliver.llm.llm import AgentCore

        te = AgentCore.__new__(AgentCore)
        te.permission_manager = pm
        te.on_permission_prompt = on_prompt
        te.on_tool_event = None
        return te

    def _make_pm(self, mode=PermissionMode.DEFAULT, rules=None):
        pm = PermissionManager.__new__(PermissionManager)
        pm.mode = mode
        pm.rules = [PermissionRule(**r) for r in (rules or [])]
        pm._session_grants = {}
        pm._task_stack = []
        return pm

    @pytest.mark.asyncio
    async def test_allow_returns_none(self):
        pm = self._make_pm(mode=PermissionMode.YOLO)
        te = self._make_executor(pm)
        result = await te._check_permission("read_file", {"file_path": "/x"}, "id1", [], None, None)
        assert result is None  # Proceed

    @pytest.mark.asyncio
    async def test_deny_returns_continue(self):
        pm = self._make_pm(rules=[{"tool": "run_shell_command", "action": "deny"}])
        te = self._make_executor(pm)
        messages = []
        result = await te._check_permission("run_shell_command", {"command": "rm"}, "id1", messages, None, None)
        assert result == (False, None)  # Continue loop, don't stop
        # Should have appended AIMessage + ToolMessage
        assert len(messages) == 2
        assert "Permission denied" in messages[1].content

    @pytest.mark.asyncio
    async def test_ask_user_allows(self):
        pm = self._make_pm()
        te = self._make_executor(pm, on_prompt=lambda name, args: "allow")
        result = await te._check_permission("read_file", {"file_path": "/x"}, "id1", [], None, None)
        assert result is None

    @pytest.mark.asyncio
    async def test_ask_user_denies(self):
        pm = self._make_pm()
        te = self._make_executor(pm, on_prompt=lambda name, args: "deny")
        messages = []
        result = await te._check_permission("read_file", {"file_path": "/x"}, "id1", messages, None, None)
        assert result == (False, None)
        assert len(messages) == 2

    @pytest.mark.asyncio
    async def test_ask_user_allow_always_grants_session(self):
        pm = self._make_pm()
        te = self._make_executor(pm, on_prompt=lambda name, args: "allow_always")
        await te._check_permission("read_file", {"file_path": "/x"}, "id1", [], None, None)
        # Should now be session-granted
        assert pm.check("read_file", {"file_path": "/any"}) == PermissionDecision.ALLOW

    @pytest.mark.asyncio
    async def test_ask_user_deny_always_grants_session(self):
        pm = self._make_pm()
        te = self._make_executor(pm, on_prompt=lambda name, args: "deny_always")
        await te._check_permission("read_file", {"file_path": "/x"}, "id1", [], None, None)
        assert pm.check("read_file", {"file_path": "/any"}) == PermissionDecision.DENY

    @pytest.mark.asyncio
    async def test_safe_tools_skip_prompt(self):
        pm = self._make_pm()
        prompt_called = []
        te = self._make_executor(pm, on_prompt=lambda n, a: prompt_called.append(1) or "allow")
        result = await te._check_permission("skill", {}, "id1", [], None, None)
        assert result is None
        assert len(prompt_called) == 0  # Never prompted

    @pytest.mark.asyncio
    async def test_no_permission_manager_falls_back_to_confirm(self):
        """When no PermissionManager, legacy confirm_tool_exec is used."""
        te = self._make_executor(pm=None)
        te.permission_manager = None
        result = await te._check_permission(
            "read_file", {"file_path": "/x"}, "id1", [], None, confirm_tool_exec=lambda prompt: False
        )
        assert result == (True, "Stopped at tool execution: read_file")

    @pytest.mark.asyncio
    async def test_no_permission_manager_no_confirm_auto_allows(self):
        te = self._make_executor(pm=None)
        te.permission_manager = None
        result = await te._check_permission("read_file", {"file_path": "/x"}, "id1", [], None, None)
        assert result is None


# ---------------------------------------------------------------------------
# Settings file loading
# ---------------------------------------------------------------------------


class TestSettingsLoading:
    def _write_settings(self, path: Path, data: dict):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.dump(data))

    def test_load_global_settings(self, tmp_path):
        global_dir = tmp_path / "global"
        self._write_settings(
            global_dir / "cliver-settings.yaml",
            {
                "permission_mode": "auto-edit",
                "permissions": [
                    {"tool": "read_file", "action": "allow"},
                ],
            },
        )
        pm = PermissionManager(global_config_dir=global_dir)
        assert pm.mode == PermissionMode.AUTO_EDIT
        assert len(pm.rules) == 1
        assert pm.rules[0].tool == "read_file"

    def test_load_local_settings(self, tmp_path):
        local_dir = tmp_path / "local"
        self._write_settings(
            local_dir / "cliver-settings.yaml",
            {
                "permission_mode": "yolo",
                "permissions": [
                    {"tool": ".*", "action": "allow"},
                ],
            },
        )
        pm = PermissionManager(global_config_dir=tmp_path / "empty", local_dir=local_dir)
        assert pm.mode == PermissionMode.YOLO
        assert len(pm.rules) == 1

    def test_local_mode_overrides_global(self, tmp_path):
        global_dir = tmp_path / "global"
        local_dir = tmp_path / "local"
        self._write_settings(
            global_dir / "cliver-settings.yaml",
            {
                "permission_mode": "default",
                "permissions": [
                    {"tool": "read_file", "action": "allow"},
                ],
            },
        )
        self._write_settings(
            local_dir / "cliver-settings.yaml",
            {
                "permission_mode": "yolo",
                "permissions": [
                    {"tool": "write_file", "action": "allow"},
                ],
            },
        )
        pm = PermissionManager(global_config_dir=global_dir, local_dir=local_dir)
        # Local mode wins
        assert pm.mode == PermissionMode.YOLO
        # Rules accumulate (global first, then local)
        assert len(pm.rules) == 2
        assert pm.rules[0].tool == "read_file"
        assert pm.rules[1].tool == "write_file"

    def test_missing_files_graceful(self, tmp_path):
        """No settings files → defaults (mode=default, no rules)."""
        pm = PermissionManager(global_config_dir=tmp_path / "nonexistent")
        assert pm.mode == PermissionMode.DEFAULT
        assert pm.rules == []

    def test_missing_local_dir_graceful(self, tmp_path):
        global_dir = tmp_path / "global"
        self._write_settings(
            global_dir / "cliver-settings.yaml",
            {
                "permission_mode": "auto-edit",
            },
        )
        pm = PermissionManager(global_config_dir=global_dir, local_dir=None)
        assert pm.mode == PermissionMode.AUTO_EDIT
        assert pm.rules == []

    def test_empty_settings_file(self, tmp_path):
        global_dir = tmp_path / "global"
        (global_dir).mkdir(parents=True)
        (global_dir / "cliver-settings.yaml").write_text("")
        pm = PermissionManager(global_config_dir=global_dir)
        assert pm.mode == PermissionMode.DEFAULT
        assert pm.rules == []

    def test_settings_only_mode(self, tmp_path):
        """Settings with mode but no permissions list."""
        global_dir = tmp_path / "global"
        self._write_settings(
            global_dir / "cliver-settings.yaml",
            {
                "permission_mode": "yolo",
            },
        )
        pm = PermissionManager(global_config_dir=global_dir)
        assert pm.mode == PermissionMode.YOLO
        assert pm.rules == []

    def test_settings_only_rules(self, tmp_path):
        """Settings with rules but no mode → default mode."""
        global_dir = tmp_path / "global"
        self._write_settings(
            global_dir / "cliver-settings.yaml",
            {
                "permissions": [
                    {"tool": "github#.*", "action": "allow"},
                ],
            },
        )
        pm = PermissionManager(global_config_dir=global_dir)
        assert pm.mode == PermissionMode.DEFAULT
        assert len(pm.rules) == 1

    def test_workflow_permissions_from_dict(self, tmp_path):
        """TaskPermissions can be constructed from a dict (as loaded from YAML)."""
        perms_dict = {
            "mode": "yolo",
            "rules": [{"tool": "read_file", "action": "allow"}],
        }
        perms = TaskPermissions(**perms_dict)
        assert perms.mode == PermissionMode.YOLO
        assert len(perms.rules) == 1

    def test_task_definition_with_permissions(self):
        """TaskDefinition model accepts permissions field."""
        from cliver.task_manager import TaskDefinition

        td = TaskDefinition(
            name="daily",
            prompt="generate report",
            permissions={"mode": "yolo"},
        )
        assert td.permissions is not None

    def test_nested_workflow_permission_accumulation(self):
        """Simulates workflow -> step nested permission scopes."""
        pm = PermissionManager()
        assert pm.check("read_file", {"file_path": "/x"}) == PermissionDecision.ASK
        assert pm.check("write_file", {"file_path": "/x"}) == PermissionDecision.ASK

        # Workflow scope: allow read_file
        pm.push_task_scope(
            TaskPermissions(
                rules=[
                    PermissionRule(tool="read_file", action=PermissionAction.ALLOW),
                ]
            )
        )
        assert pm.check("read_file", {"file_path": "/x"}) == PermissionDecision.ALLOW
        assert pm.check("write_file", {"file_path": "/x"}) == PermissionDecision.ASK

        # Step scope: allow write_file too
        pm.push_task_scope(
            TaskPermissions(
                rules=[
                    PermissionRule(tool="write_file", action=PermissionAction.ALLOW),
                ]
            )
        )
        assert pm.check("read_file", {"file_path": "/x"}) == PermissionDecision.ALLOW
        assert pm.check("write_file", {"file_path": "/x"}) == PermissionDecision.ALLOW

        # Pop step scope
        pm.pop_task_scope()
        assert pm.check("write_file", {"file_path": "/x"}) == PermissionDecision.ASK

        # Pop workflow scope
        pm.pop_task_scope()
        assert pm.check("read_file", {"file_path": "/x"}) == PermissionDecision.ASK

    def test_save_rule_to_global(self, tmp_path):
        global_dir = tmp_path / "global"
        global_dir.mkdir()
        pm = PermissionManager(global_config_dir=global_dir)

        rule = PermissionRule(tool="read_file", resource="/data/**", action=PermissionAction.ALLOW)
        pm.save_rule(rule, "global")

        assert len(pm.rules) == 1
        # Verify written to file
        data = yaml.safe_load((global_dir / "cliver-settings.yaml").read_text())
        assert len(data["permissions"]) == 1
        assert data["permissions"][0]["tool"] == "read_file"

    def test_save_rule_to_local(self, tmp_path):
        global_dir = tmp_path / "global"
        local_dir = tmp_path / "local"
        global_dir.mkdir()
        pm = PermissionManager(global_config_dir=global_dir, local_dir=local_dir)

        rule = PermissionRule(tool=".*", action=PermissionAction.ALLOW)
        pm.save_rule(rule, "local")

        assert len(pm.rules) == 1
        data = yaml.safe_load((local_dir / "cliver-settings.yaml").read_text())
        assert data["permissions"][0]["tool"] == ".*"

    def test_remove_rule(self, tmp_path):
        global_dir = tmp_path / "global"
        self._write_settings(
            global_dir / "cliver-settings.yaml",
            {
                "permissions": [
                    {"tool": "read_file", "action": "allow"},
                    {"tool": "write_file", "action": "allow"},
                ],
            },
        )
        pm = PermissionManager(global_config_dir=global_dir)
        assert len(pm.rules) == 2

        pm.remove_rule(0)
        assert len(pm.rules) == 1
        assert pm.rules[0].tool == "write_file"

        # Verify file updated
        data = yaml.safe_load((global_dir / "cliver-settings.yaml").read_text())
        assert len(data["permissions"]) == 1
        assert data["permissions"][0]["tool"] == "write_file"

    def test_save_mode(self, tmp_path):
        global_dir = tmp_path / "global"
        global_dir.mkdir()
        pm = PermissionManager(global_config_dir=global_dir)
        assert pm.mode == PermissionMode.DEFAULT

        pm.save_mode(PermissionMode.YOLO, "global")
        assert pm.mode == PermissionMode.YOLO

        data = yaml.safe_load((global_dir / "cliver-settings.yaml").read_text())
        assert data["permission_mode"] == "yolo"

    def test_loaded_rules_are_functional(self, tmp_path):
        """End-to-end: loaded rules actually affect check() decisions."""
        global_dir = tmp_path / "global"
        self._write_settings(
            global_dir / "cliver-settings.yaml",
            {
                "permission_mode": "default",
                "permissions": [
                    {"tool": "read_file", "resource": "/data/**", "action": "allow"},
                    {"tool": "run_shell_command", "resource": "git *", "action": "allow"},
                    {"tool": "run_shell_command", "resource": "rm *", "action": "deny"},
                ],
            },
        )
        pm = PermissionManager(global_config_dir=global_dir)
        assert pm.check("read_file", {"file_path": "/data/x.csv"}) == PermissionDecision.ALLOW
        assert pm.check("read_file", {"file_path": "/etc/passwd"}) == PermissionDecision.ASK
        assert pm.check("run_shell_command", {"command": "git status"}) == PermissionDecision.ALLOW
        assert pm.check("run_shell_command", {"command": "rm -rf /"}) == PermissionDecision.DENY
