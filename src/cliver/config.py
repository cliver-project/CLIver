"""
Configuration module for Cliver client.
"""

from pathlib import Path
from typing import Dict, List, Optional

import json
from pydantic import BaseModel, Field

class ModelOptions(BaseModel):
    temperature: float = Field(0.9, description="Sampling temperature")
    top_p: float = Field(1.0, description="Top-p sampling cutoff")
    max_tokens: int = Field(4096, description="Maximum number of tokens")
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
    mcpServers: Dict[str, Dict] = {}
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
                json.dump(self.config.model_dump(), f, indent=4, sort_keys=True)

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
        server = {
            "transport": "stdio",
            "command": command,
            "args": args,
            "env": env,
        }
        self.add_or_update_server(name, server)

    def add_or_update_server(self, name: str, server: Dict) -> None:
        """Add or update a server in the configuration.

        Args:
            name: Server name
            server: Server configuration
        """
        # Check if server already exists
        if name in self.config.mcpServers:
            server_in_config = self.config.mcpServers[name]
            if server_in_config.get("transport") != server.get("transport"):
                raise ValueError(
                    f"Server with name {name} already exists with a different type."
                )
            # Update existing server
            if server.get("transport") == "stdio":
                if server.get("command"):
                    server_in_config["command"] = server.get("command")
                if server.get("args"):
                    server_in_config["args"] = server.get("args")
                if server.get("env"):
                    server_in_config["env"] = server.get("env")
            elif server.get("transport") == "sse":
                if server.get("url"):
                    server_in_config["url"] = server.get("url")
                if server.get("headers"):
                    server_in_config["headers"] = server.get("headers")
        else:
            # Add new server
            self.config.mcpServers[name] = server
            # Set as default if first server
            if not self.config.default_server:
                self.config.default_server = name

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
        server = {
            "transport": "sse",
            "url": url,
            "headers": headers,
        }
        self.add_or_update_server(name, server)

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

    def get_mcp_server(self, name: Optional[str] = None) -> Optional[Dict]:
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

    def list_mcp_servers(self) -> Dict[str, Dict]:
        """List all mcp servers.

        Returns:
            List of mcp server information
        """
        return self.config.mcpServers

    def list_llm_models(self) -> Dict[str, ModelConfig]:
        """List all LLM Models
        """
        return self.config.models

    def add_or_update_llm_model(self, name: str, provider: str, api_key: str, url: str, options: str, name_in_provider: str, type: str = "TEXT_TO_TEXT") -> None:
        if not self.config.models:
            self.config.models = {}
        if name in self.config.models:
            llm = self.config.models[name]
            if provider:
                llm.provider = provider
            if url:
                llm.url = url
            if type:
                llm.type = type
            if name_in_provider:
                llm.name_in_provider = name_in_provider
        else:
            llm = ModelConfig(name=name, provider=provider, type=type, url=url)
            self.config.models[name] = llm
            if self.config.default_model is None:
                self.config.default_model = name
            if name_in_provider:
                llm.name_in_provider = name_in_provider
            else:
                llm.name_in_provider = name

        if api_key:
            llm.api_key = api_key
        if options:
            try:
                options_json = json.loads(options)
                llm.options = ModelOptions(**options_json)
            except:
                # fall backs to default
                llm.options = ModelOptions(
                    temperature=0.7,
                    top_p=0.9,
                    max_tokens=4096
                )
        self._save_config()

    def remove_llm_model(self, name: str) -> bool:
        if name in self.config.models:
            # Remove model
            self.config.models.pop(name)

            # Update default model if needed
            if self.config.default_model == name:
                self.config.default_model = (
                    self.config.models.keys()[0]
                    if self.config.models
                    else None
                )

            # Save config
            self._save_config()
            return True

        return False

    def get_llm_model(self, name: Optional[str] = None) -> Optional[ModelConfig]:
        if not name:
            name = self.config.default_model
        if self.config.models:
            return self.config.models.get(name)
        return None