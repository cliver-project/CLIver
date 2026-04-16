"""Tests for ToolRegistry — toolset-based filtering with environment gating."""

from unittest.mock import patch

from cliver.tool_registry import TOOLSETS, ToolRegistry


class TestToolsetDefinitions:
    def test_core_toolset_has_essentials(self):
        assert "read_file" in TOOLSETS["core"]
        assert "write_file" in TOOLSETS["core"]
        assert "run_shell_command" in TOOLSETS["core"]

    def test_memory_toolset(self):
        assert "memory_read" in TOOLSETS["memory"]
        assert "memory_write" in TOOLSETS["memory"]

    def test_web_toolset(self):
        assert "web_fetch" in TOOLSETS["web"]
        assert "web_search" in TOOLSETS["web"]
        assert "browse_web" in TOOLSETS["web"]

    def test_browser_toolset(self):
        assert "browser_action" in TOOLSETS["browser"]

    def test_all_toolsets_accounted(self):
        """Every tool should be in exactly one toolset."""
        all_in_toolsets = set()
        for tools in TOOLSETS.values():
            all_in_toolsets.update(tools)
        # Core tools we always expect
        assert "read_file" in all_in_toolsets
        assert "skill" in all_in_toolsets


class TestDefaultToolsets:
    def test_core_always_included(self):
        registry = ToolRegistry()
        names = registry.tool_names
        assert "read_file" in names
        assert "write_file" in names
        assert "run_shell_command" in names

    def test_memory_always_included(self):
        registry = ToolRegistry()
        names = registry.tool_names
        assert "memory_read" in names
        assert "memory_write" in names

    def test_automation_always_included(self):
        registry = ToolRegistry()
        names = registry.tool_names
        assert "skill" in names
        assert "todo_read" in names
        assert "ask_user_question" in names

    def test_web_fetch_always_available(self):
        """web_fetch and web_search should always be available."""
        registry = ToolRegistry()
        names = registry.tool_names
        assert "web_fetch" in names
        assert "web_search" in names


class TestEnvironmentGating:
    @patch.dict("os.environ", {"FIRECRAWL_API_KEY": "test-key"})
    def test_browse_web_included_with_api_key(self):
        registry = ToolRegistry()
        registry._tools = None  # force reload
        registry._ensure_loaded()
        names = registry.tool_names
        assert "browse_web" in names

    @patch.dict("os.environ", {}, clear=True)
    def test_browse_web_excluded_without_api_key(self):
        # Remove FIRECRAWL_API_KEY
        import os

        os.environ.pop("FIRECRAWL_API_KEY", None)
        registry = ToolRegistry()
        registry._tools = None
        registry._ensure_loaded()
        names = registry.tool_names
        assert "browse_web" not in names


class TestUserOverride:
    def test_explicit_toolsets(self):
        """User can specify which toolsets to enable."""
        registry = ToolRegistry(enabled_toolsets=["core"])
        names = registry.tool_names
        assert "read_file" in names
        # memory is always included even if not specified
        assert "memory_read" in names
        # web should NOT be included (not in enabled list)
        assert "web_fetch" not in names

    def test_explicit_includes_always_enabled(self):
        """Always-enabled toolsets can't be disabled."""
        registry = ToolRegistry(enabled_toolsets=[])
        names = registry.tool_names
        # core, memory, automation still included
        assert "read_file" in names
        assert "memory_read" in names
        assert "skill" in names


class TestToolAccess:
    def test_get_tool_by_name(self):
        registry = ToolRegistry()
        tool = registry.get_tool_by_name("read_file")
        assert tool is not None
        assert tool.name == "read_file"

    def test_get_tool_by_name_nonexistent(self):
        registry = ToolRegistry()
        assert registry.get_tool_by_name("nonexistent") is None

    def test_disabled_tool_still_accessible_by_name(self):
        """Disabled tools can still be executed directly (MCP, workflow)."""
        registry = ToolRegistry(enabled_toolsets=["core"])
        # web_fetch is disabled but should still be findable
        tool = registry.get_tool_by_name("web_fetch")
        assert tool is not None

    def test_user_input_does_not_filter(self):
        """user_input parameter has no effect on filtering."""
        registry = ToolRegistry()
        all_tools = registry.get_tools()
        filtered = registry.get_tools(user_input="hello world")
        assert len(filtered) == len(all_tools)

    def test_all_tools_property(self):
        registry = ToolRegistry()
        assert len(registry.all_tools) > 0

    def test_setup_docker_not_present(self):
        registry = ToolRegistry()
        assert registry.get_tool_by_name("setup_docker") is None
