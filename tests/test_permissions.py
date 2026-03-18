"""Tests for the permission system."""

import pytest

from cliver.permissions import (
    ActionKind,
    PermissionAction,
    PermissionDecision,
    PermissionManager,
    PermissionMode,
    PermissionRule,
    ResourceType,
    TaskPermissions,
    ToolMeta,
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
        pm = self._make_manager([
            {"tool": "read_file", "resource": "/data/**", "action": "allow"},
        ])
        assert pm.check("read_file", {"file_path": "/data/x.csv"}) == PermissionDecision.ALLOW

    def test_deny_rule_wins_over_allow(self):
        pm = self._make_manager([
            {"tool": "read_file", "action": "allow"},
            {"tool": "read_file", "resource": "/etc/**", "action": "deny"},
        ])
        assert pm.check("read_file", {"file_path": "/etc/passwd"}) == PermissionDecision.DENY
        assert pm.check("read_file", {"file_path": "/tmp/x"}) == PermissionDecision.ALLOW

    def test_mcp_server_wildcard_allow(self):
        pm = self._make_manager([
            {"tool": "github#.*", "action": "allow"},
        ])
        assert pm.check("github#create_issue", {}) == PermissionDecision.ALLOW
        assert pm.check("github#list_repos", {}) == PermissionDecision.ALLOW
        assert pm.check("ocp#get_pods", {}) == PermissionDecision.ASK

    def test_catch_all_wildcard(self):
        pm = self._make_manager([
            {"tool": ".*", "action": "allow"},
        ])
        assert pm.check("read_file", {"file_path": "/any"}) == PermissionDecision.ALLOW
        assert pm.check("github#create_issue", {}) == PermissionDecision.ALLOW

    def test_constrained_allow_denies_unmatched_resource(self):
        """When a tool has specific resource rules, non-matching resources are denied."""
        pm = self._make_manager([
            {"tool": "read_file", "resource": "/data/reports/**", "action": "allow"},
            {"tool": ".*", "action": "allow"},
        ])
        # Matches resource constraint
        assert pm.check("read_file", {"file_path": "/data/reports/q1.csv"}) == PermissionDecision.ALLOW
        # Does NOT match — constrained, so wildcard cannot override
        assert pm.check("read_file", {"file_path": "/etc/passwd"}) == PermissionDecision.ASK
        # Other tools still allowed by wildcard
        assert pm.check("run_shell_command", {"command": "ls"}) == PermissionDecision.ALLOW

    def test_constrained_allow_multiple_resources(self):
        pm = self._make_manager([
            {"tool": "web_fetch", "resource": "https://api.github.com/**", "action": "allow"},
            {"tool": "web_fetch", "resource": "https://docs.python.org/**", "action": "allow"},
            {"tool": ".*", "action": "allow"},
        ])
        assert pm.check("web_fetch", {"url": "https://api.github.com/repos"}) == PermissionDecision.ALLOW
        assert pm.check("web_fetch", {"url": "https://docs.python.org/3/"}) == PermissionDecision.ALLOW
        assert pm.check("web_fetch", {"url": "https://evil.com"}) == PermissionDecision.ASK

    def test_server_wide_wildcard_not_treated_as_specific(self):
        """server#.* rules should not trigger constrained-allow for individual tools."""
        pm = self._make_manager([
            {"tool": "github#.*", "action": "allow"},
        ])
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
        pm.push_task_scope(TaskPermissions(rules=[
            PermissionRule(tool="read_file", action=PermissionAction.ALLOW),
        ]))
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
        pm.push_task_scope(TaskPermissions(rules=[
            PermissionRule(tool="read_file", action=PermissionAction.ALLOW),
        ]))
        pm.push_task_scope(TaskPermissions(rules=[
            PermissionRule(tool="write_file", action=PermissionAction.ALLOW),
        ]))
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
        pm.push_task_scope(TaskPermissions(rules=[
            PermissionRule(tool=".*", action=PermissionAction.ALLOW),
            PermissionRule(tool="run_shell_command", resource="rm *", action=PermissionAction.DENY),
        ]))
        assert pm.check("read_file", {"file_path": "/any"}) == PermissionDecision.ALLOW
        assert pm.check("run_shell_command", {"command": "rm -rf /"}) == PermissionDecision.DENY

    def test_task_scope_constrained_allow(self):
        """Constrained-allow applies to task rules too."""
        pm = self._make_manager()
        pm.push_task_scope(TaskPermissions(rules=[
            PermissionRule(tool="read_file", resource="/data/**", action=PermissionAction.ALLOW),
            PermissionRule(tool=".*", action=PermissionAction.ALLOW),
        ]))
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
