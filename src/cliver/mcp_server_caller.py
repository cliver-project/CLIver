from typing import Dict, Optional, Any, Union, get_type_hints
from langchain_mcp_adapters.client import MultiServerMCPClient
import logging
from langchain_mcp_adapters.sessions import (
    StdioConnection,
    SSEConnection,
    StreamableHttpConnection,
    WebsocketConnection,
)
from langchain_core.documents.base import Blob
from langchain_core.tools import BaseTool
from langchain_core.messages import AIMessage, HumanMessage
from mcp.types import CallToolResult

logger = logging.getLogger(__name__)

def filter_dict_for_typed_dict(source: Dict, typed_dict_type: type) -> Dict:
    """Helper method to extract values from a source dict"""
    keys = get_type_hints(typed_dict_type).keys()
    return {k: source[k] for k in keys if k in source}


def _get_mcp_server_connection(server_config: Dict) -> StdioConnection | SSEConnection | StreamableHttpConnection | WebsocketConnection:
    """Get the connection configuration for an MCP server."""
    if not "transport" in server_config:
        raise ValueError(f"Transport not defined in {str(server_config)}")
    transport = server_config["transport"]
    if transport == "stdio":
        stdio_dict = filter_dict_for_typed_dict(server_config, StdioConnection)
        return StdioConnection(**stdio_dict)
    elif transport == "sse":
        sse_dict = filter_dict_for_typed_dict(server_config, SSEConnection)
        return SSEConnection(**sse_dict)
    elif transport == "streamable_http":
        stream_dict = filter_dict_for_typed_dict(server_config, StreamableHttpConnection)
        return StreamableHttpConnection(**stream_dict)
    elif transport == "websocket":
        websocket_dict = filter_dict_for_typed_dict(server_config, WebsocketConnection)
        return WebsocketConnection(**websocket_dict)
    else:
        raise ValueError(f"Transport: {transport} is not supported")


class MCPServersCaller:
    """
    The central place to interact with MP Servers
    """

    def __init__(self, mcp_servers: Dict[str, Dict]):
        self.mcp_servers = mcp_servers
        self.mcp_client = MultiServerMCPClient(
            connections={
                server_name: _get_mcp_server_connection(server_config)
                for server_name, server_config in mcp_servers.items()
            }
        )

    async def get_mcp_resource(
        self, server: str, resource_path: str = None
    ) -> Union[Dict[str, str] | list[Blob]]:
        """Call the MCP server to get resources using langchain_mcp_adapters."""
        try:
            return await self.mcp_client.get_resources(
                server_name=server, uris=resource_path
            )
        except Exception as e:
            return {"error": str(e)}

    async def get_mcp_tools(
        self, server: Optional[str] = None
    ) -> list[BaseTool]:
        """
        Call the MCP server to get tools using langchain_mcp_adapters and convert to BaseTool to be used in langchain
        """
        tools: list[BaseTool] = []
        try:
            from langchain_mcp_adapters.tools import load_mcp_tools

            server_connections = {}
            if server:
                if server not in self.mcp_client.connections:
                    raise ValueError(
                        f"Couldn't find a server with name '{server}', expected one of '{list(self.mcp_client.connections.keys())}'"
                    )
                server_connections[server] = self.mcp_client.connections[server]
            else:
                server_connections = self.mcp_client.connections

            for s_name, connection in server_connections.items():
                server_tools = await load_mcp_tools(None, connection=connection)
                for tool in server_tools:
                    tool.name = f"{s_name}#{tool.name}"
                tools.extend(server_tools)
            return tools
        except Exception as e:
            logger.error("Failed to load MCP tools", exc_info=e)
            return tools

    async def get_mcp_prompt(
        self, server: str, prompt_name: str, arguments: dict[str, Any] | None = None
    ) -> list[HumanMessage | AIMessage]:
        """Call the MCP server to get prompt using langchain_mcp_adapters."""
        return await self.mcp_client.get_prompt(
            server_name=server, prompt_name=prompt_name, arguments=arguments
        )

    async def call_mcp_server_tool(
        self, server: str, tool_name: str, args: Dict[str, Any] = None
    ) -> list[Dict[str, Any]]:
        """Call an MCP tool using langchain_mcp_adapters."""
        try:
            if server not in self.mcp_servers:
                return [
                    {"error": f"Server '{server}' not found in configured MCP servers"}
                ]

            async with self.mcp_client.session(server_name=server) as mcp_session:
                result: CallToolResult = await mcp_session.call_tool(
                    name=tool_name, arguments=args
                )
                if result.isError:
                    return [
                        {
                            "error": f"Failed to call tool {tool_name} in mcp server {server}"
                        }
                    ]
                return [c.model_dump() for c in result.content]

        except Exception as e:
            return [{"error": str(e)}]
