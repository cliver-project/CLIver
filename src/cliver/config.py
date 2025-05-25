"""
Configuration module for Cliver client.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Literal

import json
from pydantic import BaseModel, Field


class MCPServerBase(BaseModel):
    type: str
    name: str


class MCPServerStdio(MCPServerBase):
    type: Literal["stdio"]
    command: str
    args: Optional[List[str]] = None
    env: Optional[Dict[str, str]] = None

    def info(self) -> str:
        return f"Command: {self.command} {' '.join(self.args)}\nEnvironment: {self.env}"


class MCPServerSSE(MCPServerBase):
    type: Literal["sse"]
    url: str
    headers: Optional[Dict[str, str]] = None

    def info(self) -> str:
        return f"URL: {self.url}\nHeaders: {self.headers}"


MCPServer = Union[MCPServerStdio, MCPServerSSE]


class ModelOptions(BaseModel):
    temperature: float = Field(0.7, description="Sampling temperature")
    top_p: float = Field(1.0, description="Top-p sampling cutoff")
    max_tokens: int = Field(1000, description="Maximum number of tokens")
    # special class-level variable to allow extra fields
    model_config = {"extra": "allow"}


class ModelConfig(BaseModel):
    name: str
    provider: str
    type: str
    url: str
    name_in_provider: Optional[str] = Field(
        None, description="Internal name used by provider"
    )
    api_key: Optional[str] = Field(None, description="API key for the model")
    options: Optional[ModelOptions] = None


class SecretsConfig(BaseModel):
    vault_path: str
    references: Dict[str, str]


class AppConfig(BaseModel):
    mcpServers: Dict[str, MCPServer] = {}
    default_server: Optional[str] = None
    models: Dict[str, ModelConfig] = {}
    default_model: Optional[str] = None
    secrets: Optional[SecretsConfig] = None


class ConfigManager:
    """Configuration manager for Cliver client."""

    def __init__(self, config_dir: Path):
        """Initialize the configuration manager.

        Args:
            config_dir: Configuration directory
        """
        self.config_dir = config_dir
        self.config_file = self.config_dir / "config.json"
        self.config = self._load_config()

    def _load_config(self) -> AppConfig:
        """Load configuration from file.

        Returns:
            Cliver configuration
        """
        if not self.config_file.exists():
            return AppConfig()

        try:
            with open(self.config_file, "r") as f:
                file_content = f.read()
                # Check if the file is empty
                if not file_content or not file_content.strip():
                    return AppConfig()
                config_data = json.loads(file_content)
                # Ensure each MCPServer has its name set from the key
                if "mcpServers" in config_data and isinstance(
                    config_data["mcpServers"], dict
                ):
                    for name, server in config_data["mcpServers"].items():
                        if isinstance(server, dict):
                            server["name"] = name
                # Ensure each ModelConfig has its name set from the key
                if "models" in config_data and isinstance(config_data["models"], dict):
                    for name, model in config_data["models"].items():
                        if isinstance(model, dict):
                            model["name"] = name
                config = AppConfig(**config_data)
                return config
        except Exception as e:
            print(f"Error loading configuration: {e}")
            return AppConfig()

    def _save_config(self) -> None:
        """Save configuration to file.

        Args:
            config: Client configuration
        """
        try:
            if not self.config_dir.exists():
                self.config_dir.mkdir(parents=True, exist_ok=True)
            with open(self.config_file, "w") as f:
                json.dump(self.config.model_dump(), f)

        except Exception as e:
            print(f"Error saving configuration: {e}")

    def add_or_update_stdio_mcp_server(
        self,
        name: str,
        command: str = None,
        args: List[str] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> None:
        """Add a stdio server to the configuration.

        Args:
            name: Server name
            command: Command to run
            args: Command arguments
            env: Environment variables
        """
        # Create server config
        server = MCPServerStdio(
            name=name, type="stdio", command=command, args=args, env=env
        )
        self.add_or_update_server(server)

    def add_or_update_server(self, server: MCPServer) -> None:
        """Add or update a server in the configuration.

        Args:
            server: Server configuration
        """
        # Check if server already exists
        if server.name in self.config.mcpServers:
            server_in_config = self.config.mcpServers[server.name]
            if server_in_config.type != server.type:
                raise ValueError(
                    f"Server with name {server.name} already exists with a different type."
                )
            # Update existing server
            if server.type == "stdio":
                if server.command:
                    server_in_config.command = server.command
                if server.args:
                    server_in_config.args = server.args
                if server.env:
                    server_in_config.env = server.env
            elif server.type == "sse":
                if server.url:
                    server_in_config.url = server.url
                if server.headers:
                    server_in_config.headers = server.headers
        else:
            # Add new server
            self.config.mcpServers[server.name] = server
            # Set as default if first server
            if not self.config.default_server:
                self.config.default_server = server.name

        # Save config
        self._save_config()

    def add_or_update_sse_mcp_server(
        self, name: str, url: str = None, headers: Optional[Dict[str, str]] = None
    ) -> None:
        """Add a SSE server to the configuration.

        Args:
            name: Server name
            url: Server URL
            headers: Headers for the server
        """
        # Create server config
        server = MCPServerSSE(name=name, type="sse", url=url, headers=headers)
        self.add_or_update_server(server)

    def remove_mcp_server(self, name: str) -> bool:
        """Remove a server from the configuration.

        Args:
            name: Server name

        Returns:
            True if server was removed, False otherwise
        """
        # Find server
        if name in self.config.mcpServers:
            # Remove server
            self.config.mcpServers.pop(name)

            # Update default server if needed
            if self.config.default_server == name:
                self.config.default_server = (
                    self.config.mcpServers.keys()[0]
                    if self.config.mcpServers
                    else None
                )

            # Save config
            self._save_config()
            return True

        return False

    def get_mcp_server(self, name: Optional[str] = None) -> Optional[MCPServer]:
        """Get a server configuration.

        Args:
            name: Server name (defaults to default server)

        Returns:
            Server configuration if found, None otherwise
        """
        # Use default server if name not specified
        if not name:
            name = self.config.default_server
        if self.config.mcpServers:
            return self.config.mcpServers.get(name)
        return None

    def set_default_mcp_server(self, name: str) -> bool:
        """Set the default server.

        Args:
            name: Server name

        Returns:
            True if default server was set, False otherwise
        """
        # Check if server exists
        if name in self.config.mcpServers:
            # Check if server is already default
            if self.config.default_server == name:
                return True
            self.config.default_server = name
            # Save config
            self._save_config()
            return True

        return False

    def list_mcp_servers(self) -> List[MCPServer]:
        """List all mcp servers.

        Returns:
            List of mcp server information
        """
        if self.config.mcpServers:
            return [server for _, server in self.config.mcpServers.items()]
        return []
