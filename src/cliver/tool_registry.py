"""
Tool Registry for CLIver builtin tools.

Tools are organized into toolsets and filtered by environment availability.
Default toolsets are auto-determined — no config required. Users can override
via config if they want fewer/more tools.

Toolsets:
- core: file ops, shell, grep (always included)
- memory: memory, identity, session search (always included)
- automation: skill, todo, parallel_tasks, ask_user (always included)
- web: web_fetch, web_search, browse_web (browse_web requires FIRECRAWL_API_KEY)
- browser: browser_action (requires playwright installed)
- container: docker_run (requires docker/podman)
"""

import inspect
import logging
import os
import shutil
from typing import Dict, List, Optional, Set

from langchain_core.tools import BaseTool

import cliver.tools

logger = logging.getLogger(__name__)

# Toolset definitions: toolset_name → set of tool names
TOOLSETS: Dict[str, Set[str]] = {
    "core": {
        "read_file",
        "write_file",
        "list_directory",
        "grep_search",
        "run_shell_command",
        "execute_code",
        "transcribe_audio",
    },
    "memory": {
        "memory_read",
        "memory_write",
        "identity_update",
        "search_sessions",
    },
    "automation": {
        "skill",
        "todo_read",
        "todo_write",
        "ask_user_question",
        "parallel_tasks",
    },
    "web": {
        "web_fetch",
        "web_search",
        "browse_web",
    },
    "browser": {
        "browser_action",
    },
    "container": {
        "docker_run",
    },
}

# Toolsets always included regardless of environment
_ALWAYS_ENABLED = {"core", "memory", "automation"}

# Environment checks for optional toolsets.
# Returns True if the toolset's dependencies are available.
_TOOLSET_CHECKS: Dict[str, callable] = {
    "web": lambda: True,  # web_fetch/web_search always work; browse_web checked per-tool
    "browser": lambda: _is_importable("playwright"),
    "container": lambda: shutil.which("docker") is not None or shutil.which("podman") is not None,
}

# Per-tool environment checks for tools within enabled toolsets.
# If a tool's check fails, it's excluded even if its toolset is enabled.
_TOOL_CHECKS: Dict[str, callable] = {
    "browse_web": lambda: bool(os.environ.get("FIRECRAWL_API_KEY")),
}


def _is_importable(module_name: str) -> bool:
    """Check if a Python module is importable without actually importing it."""
    import importlib.util

    return importlib.util.find_spec(module_name) is not None


class ToolRegistry:
    """Registry that indexes builtin tools by name and filters by toolset.

    Default behavior (no config):
    - core + memory + automation: always included
    - web: included (browse_web excluded if no FIRECRAWL_API_KEY)
    - browser: included if playwright is installed
    - container: included if docker/podman is available

    Users can override via enabled_toolsets parameter.
    """

    def __init__(self, enabled_toolsets: Optional[List[str]] = None):
        self._tools: Optional[List[BaseTool]] = None
        self._tools_by_name: Optional[Dict[str, BaseTool]] = None
        self._enabled_toolsets = enabled_toolsets

    def _ensure_loaded(self):
        """Lazy-load and index all tools on first access."""
        if self._tools is not None:
            return

        # Discover all tools from the module
        all_tools_by_name: Dict[str, BaseTool] = {}
        for _name, obj in inspect.getmembers(cliver.tools):
            if isinstance(obj, BaseTool):
                all_tools_by_name[obj.name] = obj

        # Resolve which toolsets are enabled
        enabled = self._resolve_enabled_toolsets()

        # Collect enabled tool names
        enabled_tool_names: Set[str] = set()
        for toolset_name in enabled:
            if toolset_name in TOOLSETS:
                enabled_tool_names.update(TOOLSETS[toolset_name])

        # Apply per-tool environment checks
        final_tool_names = set()
        for name in enabled_tool_names:
            check = _TOOL_CHECKS.get(name)
            if check and not check():
                logger.debug(f"Tool '{name}' excluded (environment check failed)")
                continue
            final_tool_names.add(name)

        # Build filtered tool list
        self._tools = []
        self._tools_by_name = {}
        for name, tool in all_tools_by_name.items():
            if name in final_tool_names:
                self._tools.append(tool)
                self._tools_by_name[name] = tool
            else:
                # Still index for execute_tool (MCP/direct calls) but don't send to LLM
                self._tools_by_name[name] = tool

        logger.info(
            "Tool registry: %d tools enabled (from %d total). Toolsets: %s",
            len(self._tools), len(all_tools_by_name), ", ".join(sorted(enabled)),
        )

    def _resolve_enabled_toolsets(self) -> Set[str]:
        """Determine which toolsets are enabled."""
        if self._enabled_toolsets is not None:
            # User-specified override
            return set(self._enabled_toolsets) | _ALWAYS_ENABLED

        # Auto-detect from environment
        enabled = set(_ALWAYS_ENABLED)
        for toolset_name, check_fn in _TOOLSET_CHECKS.items():
            try:
                if check_fn():
                    enabled.add(toolset_name)
                    logger.debug(f"Toolset '{toolset_name}' enabled (env check passed)")
                else:
                    logger.debug(f"Toolset '{toolset_name}' disabled (env check failed)")
            except Exception as e:
                logger.debug(f"Toolset '{toolset_name}' disabled (check error: {e})")

        return enabled

    @property
    def all_tools(self) -> List[BaseTool]:
        """Get all enabled tools (sent to LLM)."""
        self._ensure_loaded()
        return list(self._tools)

    @property
    def tool_names(self) -> List[str]:
        """Get all enabled tool names."""
        self._ensure_loaded()
        return [t.name for t in self._tools]

    def get_tool_by_name(self, name: str) -> Optional[BaseTool]:
        """Look up a tool by exact name (includes disabled tools for direct execution)."""
        self._ensure_loaded()
        return self._tools_by_name.get(name) or self._tools_by_name.get(f"builtin#{name}")

    def get_tools(self, user_input: Optional[str] = None) -> List[BaseTool]:
        """Get enabled tools for LLM. user_input accepted for API compat but unused."""
        self._ensure_loaded()
        return list(self._tools)

    def execute_tool(self, tool_name: str, args=None) -> list:
        """Execute a builtin tool by name (works for both enabled and disabled tools)."""
        if args is None:
            args = {}

        tool = self.get_tool_by_name(tool_name)
        if tool is None:
            return [{"error": f"Tool '{tool_name}' not found in registry"}]

        try:
            result = tool.invoke(input=args)
            if isinstance(result, dict):
                return [result]
            elif isinstance(result, list):
                return result
            elif result is None:
                return [{}]
            else:
                return [{"tool_result": str(result)}]
        except Exception as e:
            logger.error(f"Failed to execute tool {tool_name}", exc_info=e)
            return [{"error": str(e)}]
