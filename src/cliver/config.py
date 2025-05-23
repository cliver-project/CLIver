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
    args: List[str]
    env: Optional[Dict[str, str]] = None


class MCPServerSSE(MCPServerBase):
    type: Literal["sse"]
    url: str
    headers: Optional[Dict[str, str]] = None


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
    mcpServers: Dict[str, MCPServer]
    default_server: Optional[str] = None
    models: Dict[str, ModelConfig]
    default_model: Optional[str] = None
    secrets: SecretsConfig


class ConfigManager:
    """Configuration manager for Cliver client."""

    def __init__(self, config_dir: Path):
        """Initialize the configuration manager.

        Args:
            config_dir: Configuration directory
        """
        self.config_dir = config_dir
        self.config_file = self.config_dir / "config.json"
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.config = self._load_config()

    def _load_config(self) -> AppConfig:
        """Load configuration from file.

        Returns:
            Client configuration
        """
        if not self.config_file.exists():
            return AppConfig()

        try:
            with open(self.config_file, "r") as f:
                config_data = json.load(f)
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

    def _save_config(self, config: AppConfig) -> None:
        """Save configuration to file.

        Args:
            config: Client configuration
        """
        try:
            with open(self.config_file, "w") as f:
                json.dump(config, f, indent=4)

        except Exception as e:
            print(f"Error saving configuration: {e}")

    def add_stdio_mcp_server(
        self,
        name: str,
        command: str,
        args: List[str],
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
        self._add_or_update_server(server)

    def _add_or_update_server(self, server: MCPServer) -> None:
        """Add or update a server in the configuration.

        Args:
            server: Server configuration
        """
        # Check if server already exists
        for i, existing in enumerate(self.config.mcpServers):
            if existing.name == server.name:
                # Replace existing server
                self.config.mcpServers[i] = server
                break
        else:
            # Add new server
            self.config.mcpServers.append(server)

        # Set as default if first server
        if not self.config.default_server:
            self.config.default_server = server.name

        # Save config
        self._save_config(self.config)

    def add_sse_mcp_server(
        self, name: str, url: str, headers: Optional[Dict[str, str]] = None
    ) -> None:
        """Add a SSE server to the configuration.

        Args:
            name: Server name
            url: Server URL
            headers: Headers for the server
        """
        # Create server config
        server = MCPServerSSE(name=name, type="sse", url=url, headers=headers)
        self._add_or_update_server(server)

    def remove_mcp_server(self, name: str) -> bool:
        """Remove a server from the configuration.

        Args:
            name: Server name

        Returns:
            True if server was removed, False otherwise
        """
        # Find server
        for i, server in enumerate(self.config.mcpServers):
            if server.name == name:
                # Remove server
                self.config.mcpServers.pop(i)

                # Update default server if needed
                if self.config.default_server == name:
                    self.config.default_server = (
                        self.config.mcpServers[0].name
                        if self.config.mcpServers
                        else None
                    )

                # Save config
                self._save_config(self.config)
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

        if not name:
            return None

        # Find server
        for server in self.config.mcpServers:
            if server.name == name:
                return server

        return None

    def set_default_mcp_server(self, name: str) -> bool:
        """Set the default server.

        Args:
            name: Server name

        Returns:
            True if default server was set, False otherwise
        """
        # Check if server exists
        for server in self.config.mcpServers:
            if server.name == name:
                # Set default server
                self.config.default_server = name

                # Save config
                self._save_config(self.config)
                return True

        return False

    def list_mcp_servers(self) -> List[MCPServer]:
        """List all mcp servers.

        Returns:
            List of mcp server information
        """
        return [server for _, server in self.config.mcpServers.items()]
