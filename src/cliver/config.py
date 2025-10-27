"""
Configuration module for Cliver client.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from pydantic import BaseModel, Field

# Import model capabilities
from cliver.model_capabilities import (
    ModelCapabilities,
    ModelCapability,
    ModelCapabilityDetector,
)

logger = logging.getLogger(__name__)


class ModelOptions(BaseModel):
    temperature: float = Field(default=0.7, description="Sampling temperature")
    top_p: float = Field(default=0.3, description="Top-p sampling cutoff")
    max_tokens: int = Field(default=4096, description="Maximum number of tokens")
    frequency_penalty: float = Field(default=0.1, description="Model’s tendency to repeat tokens")
    # special class-level variable to allow extra fields
    model_config = {"extra": "allow"}


class ModelConfig(BaseModel):
    name: str
    provider: str
    url: str
    name_in_provider: Optional[str] = Field(default=None, description="Internal name used by provider")
    api_key: Optional[str] = Field(default=None, description="API key for the model")
    options: Optional[ModelOptions] = Field(default=None, description="Options for model")
    capabilities: Optional[Set[ModelCapability]] = Field(default=None, description="Model capabilities")

    model_config = {"extra": "allow"}

    def get_capabilities(self) -> Set[ModelCapability]:
        """
        Get the model's capabilities. If not explicitly set, detect based on
        provider and model name.

        Returns:
            Set of ModelCapability enums representing the model's capabilities
        """
        if self.capabilities is not None:
            return self.capabilities

        # If capabilities not explicitly set, detect them
        capabilities = self.get_model_capabilities()
        return capabilities.capabilities

    def get_model_capabilities(self) -> ModelCapabilities:
        detector = ModelCapabilityDetector()
        capabilities = detector.detect_capabilities(self.provider, self.name)
        return capabilities

    # we need to override this for persistence purpose to skip null values on saving
    # as we already have the name as the key, we don't want to persistent the name to the config json
    def model_dump(self, **kwargs):
        """Override to exclude name field and null values."""
        data = super().model_dump(**kwargs)
        # Remove name field since it's redundant (key in models dict)
        data.pop("name", None)
        # Remove null values
        result = {k: v for k, v in data.items() if v is not None}

        # Handle capabilities serialization
        if "capabilities" in result and result["capabilities"]:
            # Convert set of ModelCapability enums to list of strings
            result["capabilities"] = [cap.value for cap in result["capabilities"]]

        return result


class MCPServerConfig(BaseModel):
    """Base class for MCP server configurations."""

    name: str
    transport: str

    def model_dump(self, **kwargs):
        """Override to exclude name field and null values."""
        data = super().model_dump(**kwargs)
        # Remove name field since it's redundant (key in mcpServers dict)
        data.pop("name", None)
        # Remove null values
        return {k: v for k, v in data.items() if v is not None}


class StdioMCPServerConfig(MCPServerConfig):
    """Configuration for stdio MCP servers."""

    transport: str = "stdio"
    command: str
    args: Optional[List[str]] = Field(default=None, description="Arguments to start the stdio mcp server")
    env: Optional[Dict[str, str]] = Field(default=None, description="Environment variables for the stdio mcp server")


class SSEMCPServerConfig(MCPServerConfig):
    """Configuration for SSE MCP servers (deprecated)."""

    transport: str = "sse"
    url: str
    headers: Optional[Dict[str, str]] = Field(
        default=None, description="The HTTP headers to interact with the SSE MCP server"
    )


class StreamableHttpMCPServerConfig(MCPServerConfig):
    """Configuration for Streamable HTTP MCP servers."""

    transport: str = "streamable_http"
    url: str
    headers: Optional[Dict[str, str]] = Field(
        default=None,
        description="The HTTP headers to interact with the streamable_http MCP server",
    )


class WebSocketMCPServerConfig(MCPServerConfig):
    """Configuration for WebSocket MCP servers."""

    transport: str = "websocket"
    url: str
    headers: Optional[Dict[str, str]] = Field(
        default=None,
        description="The HTTP headers to interact with the websocket MCP server",
    )


class WorkflowConfig(BaseModel):
    """Configuration for workflow execution."""

    workflow_dirs: Optional[List[str]] = Field(
        default=None,
        description="List of directories to search for workflows. Defaults to "
        ".cliver/workflows and ~/.config/cliver/workflows",
    )
    cache_dir: Optional[str] = Field(
        default=None,
        description="Directory for caching workflow execution results. Defaults to ~/.config/cliver/workflow_cache",
    )


class AppConfig(BaseModel):
    mcpServers: Dict[str, MCPServerConfig] = {}
    default_server: Optional[str] = Field(default=None, description="The default MCP server")
    models: Dict[str, ModelConfig] = {}
    default_model: Optional[str] = Field(default=None, description="The default LLM model")
    workflow: Optional[WorkflowConfig] = Field(default=None, description="Workflow configuration")

    def model_dump(self, **kwargs):
        """Override to exclude null values."""
        data = super().model_dump(**kwargs)
        # Remove null values
        return {k: v for k, v in data.items() if v is not None}


# TODO: support the configuration from others like from a k8s ConfigMap
# TODO: shall we support yaml format as well ?


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
            logger.info(f"No configuration file found at {str(self.config_dir)}, using default configuration.")
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
                            # Handle capabilities deserialization
                            if "capabilities" in model and model["capabilities"]:
                                # Convert list of strings to set of ModelCapability enums
                                try:
                                    model["capabilities"] = {ModelCapability(cap) for cap in model["capabilities"]}
                                except ValueError as e:
                                    # we tolerate the bad capabilities configuration and just ignore it.
                                    logger.warning(f"Warning: Invalid capability in model {name}: {e}")
                                    model["capabilities"] = None

                mcp_servers_data = config_data.get("mcpServers")
                if mcp_servers_data and isinstance(mcp_servers_data, dict):
                    converted_servers = {}
                    for name, server in mcp_servers_data.items():
                        if isinstance(server, dict):
                            # Remove name from server dict to avoid
                            # duplicate keyword argument
                            server_dict = server.copy()
                            server_dict.pop("name", None)
                            transport = server_dict.get("transport")
                            server_config = {"name": name, **server_dict}
                            if transport == "stdio":
                                # Convert dict to StdioMCPServerConfig
                                converted_servers[name] = StdioMCPServerConfig(**server_config)
                            elif transport == "sse":
                                # Convert dict to SSEMCPServerConfig
                                converted_servers[name] = SSEMCPServerConfig(**server_config)
                            elif transport == "streamable_http":
                                # Convert dict to StreamableHttpMCPServerConfig
                                converted_servers[name] = StreamableHttpMCPServerConfig(**server_config)
                            elif transport == "websocket":
                                # Convert dict to WebSocketMCPServerConfig
                                converted_servers[name] = WebSocketMCPServerConfig(**server_config)
                            else:
                                raise ValueError(f"Unknown transport {transport}")
                    config_data["mcpServers"] = converted_servers

                # Handle workflow configuration
                if "workflow" in config_data and isinstance(config_data["workflow"], dict):
                    # Convert dict to WorkflowConfig
                    config_data["workflow"] = WorkflowConfig(**config_data["workflow"])

                config = AppConfig(**config_data)
                return config
        except Exception as e:
            # we don't want to tolerate this as it may lead to the whole configuration missing just because a blemish
            logger.error("Error loading configuration: %s", e, stack_info=True, exc_info=True)
            raise e

    def _save_config(self) -> None:
        """Save configuration to file."""
        try:
            if not self.config_dir.exists():
                self.config_dir.mkdir(parents=True, exist_ok=True)

            # Use the AppConfig's model_dump method which excludes null values
            config_data = self.config.model_dump()

            # Handle MCP servers serialization to preserve all fields and
            # exclude redundant name
            if "mcpServers" in config_data:
                serialized_servers = {}
                for name, server in self.config.mcpServers.items():
                    # Use the server's own model_dump to preserve all fields
                    # and exclude redundant name
                    serialized_servers[name] = server.model_dump()
                config_data["mcpServers"] = serialized_servers

            # Handle models serialization to exclude redundant name
            if "models" in config_data:
                serialized_models = {}
                for name, model in self.config.models.items():
                    # Use the model's own model_dump to preserve all fields
                    # and exclude redundant name
                    serialized_models[name] = model.model_dump()
                config_data["models"] = serialized_models

            # Handle workflow configuration serialization
            if "workflow" in config_data and self.config.workflow:
                # Use the workflow config's own model_dump
                config_data["workflow"] = self.config.workflow.model_dump()

            with open(self.config_file, "w") as f:
                json.dump(config_data, f, indent=4, sort_keys=True)

        except Exception as e:
            logger.error("Error saving configuration: %s", e)
            raise e

    def add_or_update_stdio_mcp_server(
        self,
        name: str,
        command: str,
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
        server_config = {"name": name, "command": command}
        if args is not None:
            # noinspection PyTypeChecker
            server_config["args"] = args
        if env is not None:
            # noinspection PyTypeChecker
            server_config["env"] = env
        server = StdioMCPServerConfig(**server_config)
        self.add_or_update_server(name, server)

    def add_or_update_server(self, name: str, server: Union[Dict, MCPServerConfig]) -> None:
        """Add or update a server in the configuration.

        Args:
            name: Server name
            server: Server configuration (either as dict or MCPServerConfig)
        """
        # Convert dict to appropriate server config type if needed
        if isinstance(server, dict):
            transport = server.get("transport")
            if transport == "stdio":
                server = StdioMCPServerConfig(name=name, **server)
            elif transport == "sse":
                server = SSEMCPServerConfig(name=name, **server)
            elif transport == "streamable_http":
                server = StreamableHttpMCPServerConfig(name=name, **server)
            elif transport == "websocket":
                server = WebSocketMCPServerConfig(name=name, **server)
            else:
                raise ValueError(f"Unsupported transport type: {transport}")

        # Check if server already exists
        if name in self.config.mcpServers:
            existing_server = self.config.mcpServers[name]
            if existing_server.transport != server.transport:
                raise ValueError(f"Server with name {name} already exists with a different type.")
            # Update existing server
            self.config.mcpServers[name] = server
        else:
            # Add new server
            self.config.mcpServers[name] = server
            # Set as default if first server
            if not self.config.default_server:
                self.config.default_server = name

        # Save config
        self._save_config()

    def add_or_update_sse_mcp_server(self, name: str, url: str, headers: Optional[Dict[str, str]] = None) -> None:
        """Add a SSE server to the configuration (deprecated, use streamable instead).

        Args:
            name: Server name
            url: Server URL
            headers: Headers for the server
        """
        # Create server config
        server_config = {"name": name, "url": url}
        if headers is not None:
            # noinspection PyTypeChecker
            server_config["headers"] = headers
        server = SSEMCPServerConfig(**server_config)
        self.add_or_update_server(name, server)

    def add_or_update_streamable_mcp_server(
        self, name: str, url: str, headers: Optional[Dict[str, str]] = None
    ) -> None:
        """Add a Streamable HTTP server to the configuration.

        Args:
            name: Server name
            url: Server URL
            headers: Headers for the server
        """
        # Create server config
        server_config = {"name": name, "url": url}
        if headers is not None:
            # noinspection PyTypeChecker
            server_config["headers"] = headers
        server = StreamableHttpMCPServerConfig(**server_config)
        self.add_or_update_server(name, server)

    def add_or_update_websocket_mcp_server(self, name: str, url: str, headers: Optional[Dict[str, str]] = None) -> None:
        """Add a WebSocket server to the configuration.

        Args:
            name: Server name
            url: Server URL
            headers: Headers for the server
        """
        # Create server config
        server_config = {"name": name, "url": url}
        if headers is not None:
            # noinspection PyTypeChecker
            server_config["headers"] = headers
        server = WebSocketMCPServerConfig(**server_config)
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
                self.config.default_server = None if not self.config.mcpServers else next(iter(self.config.mcpServers))

            # Save config
            self._save_config()
            return True

        return False

    def get_mcp_server(self, name: Optional[str] = None) -> Optional[MCPServerConfig]:
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

    def list_mcp_servers(self) -> Dict[str, MCPServerConfig]:
        """List all mcp servers.

        Returns:
            List of mcp server information
        """
        return self.config.mcpServers

    def list_mcp_servers_for_mcp_caller(self) -> Dict[str, Dict]:
        """List all mcp servers as dictionaries for the MCP caller.

        Returns:
            List of mcp server information as dictionaries
        """
        # Convert Pydantic models to dictionaries for compatibility with
        # MCP server caller
        return {name: server.model_dump() for name, server in self.config.mcpServers.items()}

    def list_llm_models(self) -> Dict[str, ModelConfig]:
        """List all LLM Models"""
        return self.config.models

    def add_or_update_llm_model(
        self,
        name: str,
        provider: str,
        api_key: str,
        url: str,
        options: Dict[str, Any],
        name_in_provider: str,
        capabilities: str = None,
    ) -> None:
        if not self.config.models:
            self.config.models = {}
        if name in self.config.models:
            # update as it is already in the config
            llm = self.config.models[name]
            if provider:
                llm.provider = provider
            if url:
                llm.url = url
            if name_in_provider:
                llm.name_in_provider = name_in_provider
        else:
            # create a new config for LLM
            llm = ModelConfig(name=name, provider=provider, url=url)
            self.config.models[name] = llm
            if self.config.default_model is None:
                self.config.default_model = name
            if name_in_provider:
                llm.name_in_provider = name_in_provider
            else:
                llm.name_in_provider = name

        if api_key:
            llm.api_key = api_key
        if options and len(options) > 0:
            llm.options = ModelOptions(**options)

        # Handle capabilities
        if capabilities:
            # Parse comma-separated capabilities into a set of ModelCapability enums
            try:
                capability_list = [cap.strip() for cap in capabilities.split(",") if cap.strip()]
                capability_set = set()
                for cap_str in capability_list:
                    # Convert string to ModelCapability enum
                    capability_set.add(ModelCapability(cap_str))
                llm.capabilities = capability_set
            except ValueError as e:
                # we don't tolerate this because it is saving.
                logger.error(
                    "Warning: Invalid capability specified: %s, exception: %s",
                    capabilities,
                    e,
                )
                raise e

        self._save_config()

    def remove_llm_model(self, name: str) -> bool:
        if name in self.config.models:
            # Remove model
            self.config.models.pop(name)

            # Update default model if needed
            if self.config.default_model == name:
                self.config.default_model = next(iter(self.config.models)) if self.config.models else None

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
