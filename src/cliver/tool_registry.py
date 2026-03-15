"""
Tool Registry for CLIver builtin tools.

Provides keyword-based filtering so only relevant tools are sent to the LLM,
reducing token usage as the tool count grows.

Tools are categorized as:
- Core: always included regardless of user input (e.g., read_file, write_file)
- Contextual: included only when user input matches their tags/name/description
"""

import inspect
import logging
import re
from typing import Dict, List, Optional, Set

from langchain_core.tools import BaseTool

import cliver.tools

logger = logging.getLogger(__name__)

# Tools that are always sent to the LLM regardless of user input.
# These are fundamental tools that the LLM needs for most tasks.
CORE_TOOLS: Set[str] = {
    "read_file",
    "write_file",
    "list_directory",
    "grep_search",
    "run_shell_command",
    "todo_read",
    "todo_write",
    "ask_user_question",
    "skill",
}


class ToolRegistry:
    """
    Registry that indexes builtin tools by name and tags for efficient lookup.

    Usage:
        registry = ToolRegistry()
        # Get all tools (no filtering)
        all_tools = registry.get_tools()
        # Get core + contextually matched tools
        matched = registry.get_tools(user_input="search the web for python docs")
        # Execute a tool by name
        result = registry.execute_tool("read_file", {"file_path": "/tmp/test.txt"})
    """

    def __init__(self):
        self._tools: Optional[List[BaseTool]] = None
        self._tools_by_name: Optional[Dict[str, BaseTool]] = None
        self._tools_by_tag: Optional[Dict[str, List[BaseTool]]] = None

    def _ensure_loaded(self):
        """Lazy-load and index all tools on first access."""
        if self._tools is not None:
            return

        self._tools = []
        self._tools_by_name = {}
        self._tools_by_tag = {}

        for _name, obj in inspect.getmembers(cliver.tools):
            if isinstance(obj, BaseTool):
                logger.debug(f"Registry: indexing tool '{obj.name}'")
                self._tools.append(obj)
                self._tools_by_name[obj.name] = obj

                # Index by tags
                tags = getattr(obj, "tags", [])
                for tag in tags:
                    if tag not in self._tools_by_tag:
                        self._tools_by_tag[tag] = []
                    self._tools_by_tag[tag].append(obj)

    @property
    def all_tools(self) -> List[BaseTool]:
        """Get all registered tools."""
        self._ensure_loaded()
        return list(self._tools)

    @property
    def tool_names(self) -> List[str]:
        """Get all registered tool names."""
        self._ensure_loaded()
        return list(self._tools_by_name.keys())

    @property
    def available_tags(self) -> List[str]:
        """Get all available tags."""
        self._ensure_loaded()
        return list(self._tools_by_tag.keys())

    def get_tool_by_name(self, name: str) -> Optional[BaseTool]:
        """Look up a tool by exact name."""
        self._ensure_loaded()
        return self._tools_by_name.get(name) or self._tools_by_name.get(f"builtin#{name}")

    def get_tools_by_tag(self, tag: str) -> List[BaseTool]:
        """Get all tools with a specific tag."""
        self._ensure_loaded()
        return list(self._tools_by_tag.get(tag, []))

    def get_tools(self, user_input: Optional[str] = None) -> List[BaseTool]:
        """
        Get tools filtered by relevance to user input.

        If user_input is None, returns all tools.
        Otherwise, returns core tools + contextually matched tools.

        Args:
            user_input: The user's query/input to match against tool tags

        Returns:
            List of relevant BaseTool instances
        """
        self._ensure_loaded()

        if user_input is None:
            return list(self._tools)

        # Start with core tools
        result: Dict[str, BaseTool] = {}
        for tool in self._tools:
            if tool.name in CORE_TOOLS:
                result[tool.name] = tool

        # Match contextual tools by checking user input against tags, name, and description
        keywords = self._extract_keywords(user_input)
        for tool in self._tools:
            if tool.name in result:
                continue
            if self._matches(tool, keywords):
                result[tool.name] = tool

        return list(result.values())

    def execute_tool(self, tool_name: str, args=None) -> list:
        """
        Execute a builtin tool by name.

        Args:
            tool_name: Name of the tool to execute
            args: Arguments to pass to the tool

        Returns:
            List of result dicts
        """
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

    def _extract_keywords(self, text: str) -> Set[str]:
        """Extract meaningful keywords from user input."""
        # Normalize and split
        words = re.findall(r"[a-zA-Z]+", text.lower())
        # Filter out very short words and common stop words
        stop_words = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "it",
            "its",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "and",
            "or",
            "not",
            "no",
            "do",
            "does",
            "did",
            "can",
            "could",
            "will",
            "would",
            "shall",
            "should",
            "may",
            "might",
            "this",
            "that",
            "these",
            "those",
            "with",
            "from",
            "by",
            "as",
            "if",
            "then",
            "else",
            "when",
            "what",
            "how",
            "why",
            "who",
            "where",
            "which",
            "me",
            "my",
            "i",
            "you",
            "your",
            "we",
            "our",
            "he",
            "she",
            "they",
            "his",
            "her",
            "him",
            "them",
            "us",
            "all",
            "some",
            "any",
            "each",
            "every",
            "just",
            "also",
            "very",
            "too",
            "so",
            "up",
            "out",
            "about",
            "please",
            "help",
            "want",
            "need",
            "like",
            "use",
            "using",
            "get",
            "make",
            "let",
            "have",
            "has",
            "had",
        }
        return {w for w in words if len(w) > 2 and w not in stop_words}

    def _matches(self, tool: BaseTool, keywords: Set[str]) -> bool:
        """Check if a tool matches any of the extracted keywords."""
        tags = set(getattr(tool, "tags", []))
        tool_name_parts = set(tool.name.lower().split("_"))

        # Direct tag match
        if keywords & tags:
            return True

        # Tool name match
        if keywords & tool_name_parts:
            return True

        # Keyword-in-description match (check if any keyword appears in description)
        desc_lower = tool.description.lower()
        for kw in keywords:
            if kw in desc_lower:
                return True

        return False
