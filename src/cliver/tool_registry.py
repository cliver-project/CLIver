"""
Tool Registry for CLIver builtin tools.

Tools are organized into toolsets and filtered by environment availability.
Default toolsets are auto-determined — no config required. Users can override
via config if they want fewer/more tools.

Toolsets:
- core: Read, Write, LS, Grep, Bash, Exec, Transcribe, ImageGenerate (always included)
- memory: MemoryRead, MemoryWrite, Identity, SearchSessions (always included)
- automation: Skill, TodoRead, TodoWrite, Ask, Parallel (always included)
- web: WebFetch, WebSearch, Browse (Browse requires FIRECRAWL_API_KEY)
- browser: Browser (requires playwright installed)
- container: Docker (requires docker/podman)
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
        "Read",
        "Write",
        "LS",
        "Grep",
        "Bash",
        "Exec",
        "Transcribe",
        "ImageGenerate",
    },
    "memory": {
        "MemoryRead",
        "MemoryWrite",
        "Identity",
        "SearchSessions",
    },
    "automation": {
        "Skill",
        "TodoRead",
        "TodoWrite",
        "Ask",
        "Parallel",
        "WorkflowValidate",
        "CliverHelp",
    },
    "web": {
        "WebFetch",
        "WebSearch",
        "Browse",
    },
    "browser": {
        "Browser",
    },
    "container": {
        "Docker",
    },
}

# Toolsets always included regardless of environment
_ALWAYS_ENABLED = {"core", "memory", "automation"}

# Environment checks for optional toolsets.
# Returns True if the toolset's dependencies are available.
_TOOLSET_CHECKS: Dict[str, callable] = {
    "web": lambda: True,  # web_fetch/web_search always work; browse_web checked per-tool
    "browser": lambda: False,  # opt-in only via enabled_toolsets config
    "container": lambda: shutil.which("docker") is not None or shutil.which("podman") is not None,
}


# Per-tool environment checks for tools within enabled toolsets.
# If a tool's check fails, it's excluded even if its toolset is enabled.
def _playwright_ready() -> bool:
    """Check if playwright is installed AND browsers are downloaded."""
    try:
        from playwright._impl._driver import compute_driver_executable

        executable = compute_driver_executable()
        return os.path.isfile(executable)
    except Exception:
        return False


_TOOL_CHECKS: Dict[str, callable] = {
    "Browse": lambda: bool(os.environ.get("FIRECRAWL_API_KEY")),
    "Browser": _playwright_ready,
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
            len(self._tools),
            len(all_tools_by_name),
            ", ".join(sorted(enabled)),
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
            from pydantic import ValidationError

            if isinstance(e, ValidationError) or "ValidationError" in type(e).__name__:
                schema = getattr(tool, "args_schema", None)
                hint = ""
                if schema:
                    fields = schema.model_fields
                    required = [f for f, info in fields.items() if info.is_required()]
                    optional = [f for f, info in fields.items() if not info.is_required()]
                    hint = f" Required: {', '.join(required)}." if required else ""
                    hint += f" Optional: {', '.join(optional)}." if optional else ""
                logger.warning("Tool '%s' called with wrong arguments: %s", tool_name, e)
                return [{"error": f"Wrong arguments for '{tool_name}'.{hint} Got: {list(args.keys())}"}]
            logger.error(f"Failed to execute tool {tool_name}", exc_info=e)
            return [{"error": str(e)}]
