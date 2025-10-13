"""Built-in tools that are always available for the LLM to use."""
import inspect
from typing import List, Dict, Any, Union
from langchain_core.tools import BaseTool
import cliver.tools
import logging

logger = logging.getLogger(__name__)

# The builtin tools should be defined in the module: 'cliver.tools'
# All builtin tools should have a global unique name either like: 'tool_name' or 'builtin#tool_name'
def get_builtin_tools() -> List[BaseTool]:
    """
    Return a list of builtin tools that should always be available.

    Returns:
        List of BaseTool objects representing builtin tools
    """

    tools: List[BaseTool] = []
    for name, obj in inspect.getmembers(cliver.tools):
        if isinstance(obj, BaseTool):
            logger.debug(f"Register builtin tool: {name}")
            tools.append(obj)
    return tools


class BuiltinTools:
    """
    A class to manage built-in tools with caching and execution capabilities.
    """

    def __init__(self):
        self._tools = None  # Cache for builtin tools

    @property
    def tools(self) -> List[BaseTool]:
        """Get the list of builtin tools, caching them on first access."""
        if self._tools is None:
            self._tools = get_builtin_tools()
        return self._tools

    def execute_tool(self, tool_name: str, args: Union[str, Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Execute a builtin tool by name with the provided arguments.

        Args:
            tool_name (str): The name of the tool to execute
            args (Dict[str, Any]): Arguments to pass to the tool

        Returns:
            List[Dict[str, Any]]: The result of the tool execution
        """
        if args is None:
            args = {}

        # Find the tool by name
        tool = None
        for t in self.tools:
            if t.name == tool_name or t.name == f"builtin#{tool_name}":
                tool = t
                break

        if tool is None:
            return [{"error": f"Tool '{tool_name}' not found in builtin tools"}]

        try:
            # Execute the tool
            result = tool.invoke(input=args)

            # Handle different result types
            if isinstance(result, dict):
                return [result]
            elif isinstance(result, list):
                return result
            elif result is None:
                return [{}]
            else:
                return [{"tool_result": str(result)}]

        except Exception as e:
            logger.error(f"Failed to execute builtin tool {tool_name}", exc_info=e)
            return [{"error": str(e)}]
