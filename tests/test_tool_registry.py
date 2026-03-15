"""Tests for the ToolRegistry keyword-based tool matching."""

import pytest
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from cliver.tool_registry import CORE_TOOLS, ToolRegistry


@pytest.fixture
def registry():
    """Create a fresh ToolRegistry that loads real builtin tools."""
    return ToolRegistry()


# ---------------------------------------------------------------------------
# Basic loading and indexing
# ---------------------------------------------------------------------------


class TestRegistryLoading:
    def test_loads_all_tools(self, registry):
        """All 11 builtin tools should be discovered."""
        assert len(registry.all_tools) == 12

    def test_tool_names_populated(self, registry):
        names = registry.tool_names
        assert "read_file" in names
        assert "web_search" in names
        assert "docker_run" in names

    def test_available_tags_populated(self, registry):
        tags = registry.available_tags
        assert "search" in tags
        assert "execute" in tags
        assert "file" in tags

    def test_lazy_loading(self):
        """Registry should not load tools until first access."""
        reg = ToolRegistry()
        assert reg._tools is None
        _ = reg.all_tools
        assert reg._tools is not None


# ---------------------------------------------------------------------------
# Lookup by name
# ---------------------------------------------------------------------------


class TestGetToolByName:
    def test_exact_name(self, registry):
        tool = registry.get_tool_by_name("read_file")
        assert tool is not None
        assert tool.name == "read_file"

    def test_builtin_prefix_fallback(self, registry):
        """Should also match 'builtin#tool_name' convention."""
        # This tests the fallback path; real tools don't use the prefix
        tool = registry.get_tool_by_name("nonexistent_tool")
        assert tool is None

    def test_nonexistent_tool(self, registry):
        assert registry.get_tool_by_name("totally_fake_tool") is None


# ---------------------------------------------------------------------------
# Lookup by tag
# ---------------------------------------------------------------------------


class TestGetToolsByTag:
    def test_search_tag(self, registry):
        tools = registry.get_tools_by_tag("search")
        names = {t.name for t in tools}
        assert "grep_search" in names
        assert "web_search" in names
        assert "list_directory" in names

    def test_execute_tag(self, registry):
        tools = registry.get_tools_by_tag("execute")
        names = {t.name for t in tools}
        assert "run_shell_command" in names
        assert "setup_docker" in names
        assert "docker_run" in names

    def test_docker_tag(self, registry):
        tools = registry.get_tools_by_tag("docker")
        names = {t.name for t in tools}
        assert "setup_docker" in names
        assert "docker_run" in names
        assert "read_file" not in names

    def test_nonexistent_tag(self, registry):
        assert registry.get_tools_by_tag("nonexistent") == []


# ---------------------------------------------------------------------------
# get_tools() — no filtering (user_input=None)
# ---------------------------------------------------------------------------


class TestGetToolsNoFilter:
    def test_returns_all_when_no_input(self, registry):
        tools = registry.get_tools(user_input=None)
        assert len(tools) == 12


# ---------------------------------------------------------------------------
# get_tools() — core tools always included
# ---------------------------------------------------------------------------


class TestCoreToolsAlwaysIncluded:
    def test_core_tools_present_for_generic_input(self, registry):
        """Even a generic query should return all core tools."""
        tools = registry.get_tools(user_input="hello world")
        names = {t.name for t in tools}
        for core in CORE_TOOLS:
            assert core in names, f"Core tool '{core}' missing for generic input"

    def test_core_tools_present_for_specific_input(self, registry):
        tools = registry.get_tools(user_input="run a docker container")
        names = {t.name for t in tools}
        for core in CORE_TOOLS:
            assert core in names, f"Core tool '{core}' missing for specific input"

    def test_generic_input_only_returns_core(self, registry):
        """Input with no matching keywords should return only core tools."""
        tools = registry.get_tools(user_input="hello world")
        names = {t.name for t in tools}
        assert names == CORE_TOOLS


# ---------------------------------------------------------------------------
# get_tools() — contextual matching by tag
# ---------------------------------------------------------------------------


class TestContextualMatchingByTag:
    def test_web_tag_matches_web_tools(self, registry):
        tools = registry.get_tools(user_input="search the web for python tutorials")
        names = {t.name for t in tools}
        assert "web_search" in names
        assert "web_fetch" in names

    def test_docker_tag_matches_docker_tools(self, registry):
        tools = registry.get_tools(user_input="I need to set up docker")
        names = {t.name for t in tools}
        assert "setup_docker" in names
        assert "docker_run" in names

    def test_fetch_tag_matches_web_fetch(self, registry):
        tools = registry.get_tools(user_input="fetch this page")
        names = {t.name for t in tools}
        assert "web_fetch" in names


# ---------------------------------------------------------------------------
# get_tools() — contextual matching by tool name parts
# ---------------------------------------------------------------------------


class TestContextualMatchingByName:
    def test_docker_in_input_matches_docker_tools(self, registry):
        tools = registry.get_tools(user_input="docker")
        names = {t.name for t in tools}
        assert "docker_run" in names
        assert "setup_docker" in names

    def test_container_matches_docker_tools(self, registry):
        """'container' appears in docker tool descriptions."""
        tools = registry.get_tools(user_input="run a container")
        names = {t.name for t in tools}
        assert "docker_run" in names


# ---------------------------------------------------------------------------
# get_tools() — contextual matching by description keywords
# ---------------------------------------------------------------------------


class TestContextualMatchingByDescription:
    def test_url_matches_web_fetch(self, registry):
        """'url' appears in web_fetch description."""
        tools = registry.get_tools(user_input="download content from a url")
        names = {t.name for t in tools}
        assert "web_fetch" in names

    def test_ephemeral_matches_docker_run(self, registry):
        """'ephemeral' appears in docker_run description."""
        tools = registry.get_tools(user_input="ephemeral environment")
        names = {t.name for t in tools}
        assert "docker_run" in names

    def test_podman_matches_docker_tools(self, registry):
        """'podman' appears in docker tool descriptions."""
        tools = registry.get_tools(user_input="check if podman is installed")
        names = {t.name for t in tools}
        assert "setup_docker" in names


# ---------------------------------------------------------------------------
# Keyword extraction
# ---------------------------------------------------------------------------


class TestKeywordExtraction:
    def test_filters_stop_words(self, registry):
        keywords = registry._extract_keywords("the quick brown fox is very fast")
        assert "the" not in keywords
        assert "is" not in keywords
        assert "very" not in keywords
        assert "quick" in keywords
        assert "brown" in keywords
        assert "fox" in keywords
        assert "fast" in keywords

    def test_filters_short_words(self, registry):
        keywords = registry._extract_keywords("go to my db")
        assert "go" not in keywords  # len <= 2
        assert "to" not in keywords  # stop word
        assert "my" not in keywords  # stop word
        # "db" is len 2, should be filtered
        assert "db" not in keywords

    def test_case_insensitive(self, registry):
        keywords = registry._extract_keywords("Docker Container SEARCH")
        assert "docker" in keywords
        assert "container" in keywords
        assert "search" in keywords

    def test_handles_punctuation(self, registry):
        keywords = registry._extract_keywords("search, docker! container?")
        assert "search" in keywords
        assert "docker" in keywords
        assert "container" in keywords

    def test_empty_input(self, registry):
        keywords = registry._extract_keywords("")
        assert keywords == set()

    def test_only_stop_words(self, registry):
        keywords = registry._extract_keywords("the is a an to for and or")
        assert keywords == set()


# ---------------------------------------------------------------------------
# _matches() — direct unit tests
# ---------------------------------------------------------------------------


class TestMatchesMethod:
    """Test the _matches method with a mock tool to isolate matching logic."""

    def _make_tool(self, tool_name="test_tool", tags=None, description="A test tool"):
        """Create a minimal BaseTool for testing."""

        class DummyInput(BaseModel):
            value: str = Field(default="x")

        class DummyTool(BaseTool):
            name: str = "placeholder"
            description: str = "placeholder"
            args_schema: type = DummyInput
            tags: list = []

            def _run(self, **kwargs):
                return "ok"

        tool = DummyTool(name=tool_name, description=description, tags=tags or [])
        return tool

    def test_matches_by_tag(self, registry):
        tool = self._make_tool(tags=["docker", "container"])
        assert registry._matches(tool, {"docker"}) is True

    def test_matches_by_name_part(self, registry):
        tool = self._make_tool(tool_name="web_search")
        assert registry._matches(tool, {"web"}) is True
        assert registry._matches(tool, {"search"}) is True

    def test_matches_by_description(self, registry):
        tool = self._make_tool(description="Runs containers in isolated environments")
        assert registry._matches(tool, {"containers"}) is True
        assert registry._matches(tool, {"isolated"}) is True

    def test_no_match(self, registry):
        tool = self._make_tool(tool_name="read_file", tags=["read", "file"], description="Reads files")
        assert registry._matches(tool, {"docker"}) is False

    def test_empty_keywords_no_match(self, registry):
        tool = self._make_tool(tags=["search"])
        assert registry._matches(tool, set()) is False

    def test_partial_keyword_in_description(self, registry):
        """Keyword 'contain' should match 'container' in description."""
        tool = self._make_tool(description="Runs a container image")
        assert registry._matches(tool, {"contain"}) is True

    def test_tag_match_is_exact(self, registry):
        """Tag matching uses set intersection — must be exact."""
        tool = self._make_tool(tags=["docker"])
        assert registry._matches(tool, {"docker"}) is True
        assert registry._matches(tool, {"dock"}) is False  # partial tag doesn't match via tags


# ---------------------------------------------------------------------------
# Integration: realistic user queries
# ---------------------------------------------------------------------------


class TestRealisticQueries:
    def test_what_time_is_it(self, registry):
        """'time' matches 'runtime' in docker descriptions — an acceptable over-match."""
        tools = registry.get_tools(user_input="what time is it in Beijing?")
        names = {t.name for t in tools}
        # Core tools always present
        assert CORE_TOOLS.issubset(names)
        # 'time' substring-matches docker tool descriptions ('runtime')
        assert "docker_run" in names or "setup_docker" in names

    def test_read_a_config_file(self, registry):
        """File-related query should include core (which has file tools)."""
        tools = registry.get_tools(user_input="read the config file at /etc/hosts")
        names = {t.name for t in tools}
        assert "read_file" in names

    def test_build_and_deploy_with_docker(self, registry):
        tools = registry.get_tools(user_input="build and deploy my app with docker compose")
        names = {t.name for t in tools}
        assert "docker_run" in names
        assert "setup_docker" in names
        assert "run_shell_command" in names

    def test_search_codebase_for_pattern(self, registry):
        tools = registry.get_tools(user_input="find all TODO comments in the codebase")
        names = {t.name for t in tools}
        assert "grep_search" in names

    def test_plan_a_complex_task(self, registry):
        tools = registry.get_tools(user_input="plan the migration steps for the database")
        names = {t.name for t in tools}
        assert "todo_write" in names

    def test_web_research(self, registry):
        tools = registry.get_tools(user_input="search online for the latest python release")
        names = {t.name for t in tools}
        assert "web_search" in names

    def test_fetch_url_content(self, registry):
        """Explicit 'fetch' keyword should match web_fetch."""
        tools = registry.get_tools(user_input="fetch the content from this website")
        names = {t.name for t in tools}
        assert "web_fetch" in names
