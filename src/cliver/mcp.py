from typing import Dict, List, Optional, Any, Union, Literal, get_type_hints
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.sessions import StdioConnection, SSEConnection, StreamableHttpConnection, WebsocketConnection
from langchain_core.documents.base import Blob
from langchain_core.tools import BaseTool
from langchain_core.messages import AIMessage, HumanMessage
from mcp.types import CallToolResult


def filter_dict_for_typed_dict(source: Dict, typed_dict_type: type) -> Dict:
    """Helper method to extract values from a source dict
    """
    keys = get_type_hints(typed_dict_type).keys()
    return {k: source[k] for k in keys if k in source}


class MCPServersCaller:
    """
    The central place to interact with MP Servers
    """

    def __init__(self, mcp_servers: Dict[str, Dict], default_server: Optional[str] = None):
        self.mcp_servers = mcp_servers
        self.default_server = default_server
        self.mcp_client = MultiServerMCPClient({
            server_name: self._get_mcp_server_connection(server_config) for server_name, server_config in mcp_servers.items()
        })

    def _get_mcp_server_connection(self, server_config: Dict) -> Dict[str, Any]:
        """Get the connection configuration for an MCP server."""
        if not "transport" in server_config:
            raise ValueError(f"Transport not defined in {str(server_config)}")
        transport = server_config["transport"]
        if transport == "stdio":
            return filter_dict_for_typed_dict(server_config, StdioConnection)
        elif transport == "sse":
            return filter_dict_for_typed_dict(server_config, SSEConnection)
        elif transport == "streamable_http":
            return filter_dict_for_typed_dict(server_config, StreamableHttpConnection)
        elif transport == "websocket":
            return filter_dict_for_typed_dict(server_config, WebsocketConnection)
        else:
            raise ValueError(f"Transport: {transport} is not supported")

    async def get_mcp_resource(self, server: str, resource_path: str = None) -> list[Blob]:
        """Call the MCP server to get resources using langchain_mcp_adapters."""
        try:
            return await self.mcp_client.get_resources(server_name=server, uris=resource_path)
        except Exception as e:
            return {"error": str(e)}

    async def get_mcp_tools(self, server: Optional[str] = None) -> list[BaseTool]:
        """
        Call the MCP server to get tools using langchain_mcp_adapters and convert to BaseTool to be used in langchain
        """
        try:
            from langchain_mcp_adapters.tools import load_mcp_tools
            tools: list[BaseTool] = []
            server_connections = {}
            if server:
                if server not in self.mcp_client.connections:
                    raise ValueError(
                        f"Couldn't find a server with name '{server}', expected one of '{list(self.mcp_client.connections.keys())}'"
                    )
                server_connections[server] = self.mcp_client.connections[server]
            else:
                server_connections = self.mcp_client.connections

            tools = []
            for s_name, connection in server_connections.items():
                server_tools = await load_mcp_tools(None, connection=connection)
                for tool in server_tools:
                    tool.name = f"{s_name}#{tool.name}"
                tools.extend(server_tools)
            return tools
        except Exception as e:
            return {"error": str(e)}

    async def get_mcp_prompt(self, server: str) -> list[HumanMessage | AIMessage]:
        """Call the MCP server to get prompt using langchain_mcp_adapters."""
        try:
            return await self.mcp_client.get_prompt(server_name=server)
        except Exception as e:
            return {"error": str(e)}

    async def call_mcp_server_tool(self, server: str, tool_name: str, args: Dict[str, Any] = None) -> list[Dict[str, Any]]:
        """Call an MCP tool using langchain_mcp_adapters."""
        try:
            if server not in self.mcp_servers:
                return [{"error": f"Server '{server}' not found in configured MCP servers"}]

            async with self.mcp_client.session(server_name=server) as mcp_session:
                result: CallToolResult = await mcp_session.call_tool(name=tool_name, arguments=args)
                if result.isError:
                    return [{"error": f"Failed to call tool {tool_name} in mcp server {server}"}]
                return [c.model_dump() for c in result.content]

        except Exception as e:
            return [{"error": str(e)}]
