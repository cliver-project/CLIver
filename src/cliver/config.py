"""
Configuration module for Cliver client.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import yaml
from pydantic import BaseModel, Field

# Import model capabilities
from cliver.model_capabilities import (
    ModelCapabilities,
    ModelCapability,
    ModelCapabilityDetector,
)
from cliver.template_utils import render_template_if_needed

logger = logging.getLogger(__name__)


class RateLimitConfig(BaseModel):
    """Rate limit configuration for an LLM provider."""

    requests: int = Field(description="Number of requests allowed in the period")
    period: str = Field(description="Time period, e.g. '5h', '30m', '1d', or seconds as string")
    margin: float = Field(default=0.1, description="Safety margin (0.1 = 10% slower than limit)")


class PricingConfig(BaseModel):
    """Pricing configuration per million tokens."""

    currency: Optional[str] = Field(default=None, description="Currency code: USD, CNY, EUR, etc.")
    input: Optional[float] = Field(default=None, description="Cost per million input tokens")
    output: Optional[float] = Field(default=None, description="Cost per million output tokens")
    cached_input: Optional[float] = Field(default=None, description="Cost per million cached input tokens")


class ProviderConfig(BaseModel):
    """Configuration for an LLM provider (API endpoint + credentials + rate limit)."""

    name: str
    type: str = Field(description="Provider type: openai, ollama, anthropic")
    api_url: str = Field(description="Base URL for the provider API")
    api_key: Optional[str] = Field(default=None, description="API key (supports Jinja2 templates)")
    rate_limit: Optional[RateLimitConfig] = Field(default=None, description="Rate limit for API calls")
    image_url: Optional[str] = Field(default=None, description="Full URL for image generation endpoint")
    audio_url: Optional[str] = Field(default=None, description="Full URL for audio generation endpoint")
    pricing: Optional[PricingConfig] = Field(default=None, description="Token pricing for cost tracking")

    model_config = {"extra": "allow"}

    def get_api_key(self) -> Optional[str]:
        if self.api_key is None:
            return None
        return render_template_if_needed(self.api_key)

    def model_dump(self, **kwargs):
        data = super().model_dump(**kwargs)
        data.pop("name", None)
        return {k: v for k, v in data.items() if v is not None}


class ModelOptions(BaseModel):
    temperature: float = Field(default=0.7, description="Sampling temperature")
    top_p: float = Field(default=0.3, description="Top-p sampling cutoff")
    max_tokens: int = Field(default=16384, description="Maximum number of tokens")
    frequency_penalty: float = Field(default=0.1, description="Model’s tendency to repeat tokens")
    # special class-level variable to allow extra fields
    model_config = {"extra": "allow"}


class ModelConfig(BaseModel):
    name: str
    provider: str
    url: Optional[str] = Field(default=None, description="API URL for the model (overrides provider)")
    name_in_provider: Optional[str] = Field(default=None, description="Internal name used by provider")
    api_key: Optional[str] = Field(default=None, description="API key for the model")
    options: Optional[ModelOptions] = Field(default=None, description="Options for model")
    capabilities: Optional[Set[ModelCapability]] = Field(default=None, description="Model capabilities")
    think_mode: Optional[bool] = Field(
        default=None,
        description="Override thinking mode: true to enable, false to disable, null to auto-detect from model name.",
    )
    context_window: Optional[int] = Field(
        default=None,
        description="Context window size in tokens. Used for conversation history compression.",
    )
    pricing: Optional[PricingConfig] = Field(default=None, description="Token pricing (overrides provider)")

    model_config = {"extra": "allow"}

    # Internal: set during config loading, not serialized
    _provider_config: Optional["ProviderConfig"] = None

    def get_provider_type(self) -> str:
        """Return the provider type.

        If a ProviderConfig is linked, returns its type; otherwise falls back
        to the legacy ``provider`` field (which doubles as type).
        """
        if self._provider_config is not None:
            return self._provider_config.type
        return self.provider

    def get_resolved_url(self) -> Optional[str]:
        """Return the effective API URL.

        Priority: model-level ``url`` > linked ProviderConfig ``api_url`` > None.
        """
        if self.url is not None:
            return self.url
        if self._provider_config is not None:
            return self._provider_config.api_url
        return None

    def get_resolved_pricing(self) -> Optional[tuple]:
        """Resolve pricing: model-level fields override provider-level fields.

        Returns (input, output, cached_input, currency) per million tokens,
        or None if pricing is not configured with at least input and output.
        """
        currency = None
        input_price = None
        output_price = None
        cached_price = None

        # Provider-level base
        if self._provider_config is not None and self._provider_config.pricing is not None:
            pp = self._provider_config.pricing
            currency = pp.currency
            input_price = pp.input
            output_price = pp.output
            cached_price = pp.cached_input

        # Model-level overrides
        if self.pricing is not None:
            mp = self.pricing
            if mp.currency is not None:
                currency = mp.currency
            if mp.input is not None:
                input_price = mp.input
            if mp.output is not None:
                output_price = mp.output
            if mp.cached_input is not None:
                cached_price = mp.cached_input

        if input_price is None or output_price is None:
            return None

        if cached_price is None:
            cached_price = input_price
        if currency is None:
            currency = "USD"

        return (input_price, output_price, cached_price, currency)

    def get_capabilities(self) -> Set[ModelCapability]:
        """
        Get the model's capabilities. If not explicitly set, detect based on
        provider and model name. Applies think_mode override if configured.

        Returns:
            Set of ModelCapability enums representing the model's capabilities
        """
        if self.capabilities is not None:
            caps = set(self.capabilities)
        else:
            caps = set(self.get_model_capabilities().capabilities)

        # Apply think_mode override from config
        if self.think_mode is True:
            caps.add(ModelCapability.THINK_MODE)
        elif self.think_mode is False:
            caps.discard(ModelCapability.THINK_MODE)

        return caps

    def get_api_key(self) -> Optional[str]:
        """
        Resolve the API key, supporting Jinja2 template expressions.

        Returns the resolved API key:
        - Plain text: returned as-is
        - "{{ keyring('service', 'key') }}": resolved from system keyring
        - "{{ env.VARIABLE }}": resolved from environment variable
        - Falls back to linked ProviderConfig's api_key if model-level key is not set
        - None: if not configured or resolution fails
        """
        if self.api_key is not None:
            return render_template_if_needed(self.api_key)
        if self._provider_config is not None:
            return self._provider_config.get_api_key()
        return None

    def get_model_capabilities(self) -> ModelCapabilities:
        detector = ModelCapabilityDetector()
        capabilities = detector.detect_capabilities(self.get_provider_type(), self.name)
        return capabilities

    # we need to override this for persistence purpose to skip null values on saving
    # as we already have the name as the key, we don't want to persistent the name to the config json
    def model_dump(self, **kwargs):
        """Override to exclude name field, null values, and internal attributes."""
        data = super().model_dump(**kwargs)
        # Remove name field since it's redundant (key in models dict)
        data.pop("name", None)
        # Remove internal provider config (not serialized)
        data.pop("_provider_config", None)
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


class PlatformConfig(BaseModel):
    """Configuration for a single messaging platform adapter."""

    name: str = Field(default="", description="Platform name (set from config key)")
    type: str = Field(description="Adapter type: builtin name or module path")
    token: Optional[str] = Field(default=None, description="Bot/API token (supports Jinja2 templates)")
    app_token: Optional[str] = Field(default=None, description="App-level token (e.g., Slack Socket Mode)")
    home_channel: Optional[str] = Field(default=None, description="Default channel for cron delivery")
    allowed_users: Optional[List[str]] = Field(default=None, description="Whitelist of user IDs")

    model_config = {"extra": "allow"}

    @property
    def is_builtin(self) -> bool:
        return "." not in self.type

    def get_token(self) -> Optional[str]:
        if self.token is None:
            return None
        return render_template_if_needed(self.token)

    def get_app_token(self) -> Optional[str]:
        if self.app_token is None:
            return None
        return render_template_if_needed(self.app_token)

    def model_dump(self, **kwargs):
        data = super().model_dump(**kwargs)
        data.pop("name", None)
        return {k: v for k, v in data.items() if v is not None}


class GatewayConfig(BaseModel):
    """Configuration for the gateway daemon process."""

    host: str = Field(default="127.0.0.1", description="Host to bind the gateway HTTP server")
    port: int = Field(default=8321, description="Port for the gateway HTTP server")
    api_key: Optional[str] = Field(default=None, description="API key for authentication (optional)")
    platforms: Dict[str, PlatformConfig] = Field(default_factory=dict, description="Platform adapter configurations")
    log_file: Optional[str] = Field(
        default=None,
        description="Path to the gateway rotating log file. Default: {config_dir}/gateway.log",
    )
    log_max_bytes: int = Field(default=10 * 1024 * 1024, description="Max size per log file in bytes (default 10MB)")
    log_backup_count: int = Field(default=5, description="Number of rotated log files to keep (default 5)")


class SessionConfig(BaseModel):
    """Session storage limits — shared by CLI and gateway."""

    max_sessions: int = Field(
        default=300,
        description="Max sessions to keep; oldest deleted when exceeded",
    )
    max_turns_per_session: int = Field(default=100, description="Max turns per session")
    max_age_days: int = Field(default=365, description="Delete sessions idle for this many days")


class AppConfig(BaseModel):
    default_agent_name: str = Field(default="CLIver", description="The default agent instance name")
    providers: Dict[str, ProviderConfig] = Field(default_factory=dict)
    mcpServers: Dict[str, MCPServerConfig] = {}
    models: Dict[str, ModelConfig] = {}
    default_model: Optional[str] = Field(default=None, description="The default LLM model")
    user_agent: Optional[str] = Field(default="CLIver", description="User-Agent header for LLM provider HTTP requests")
    enabled_toolsets: Optional[List[str]] = Field(
        default=None,
        description="Override which tool groups are enabled. Default: auto-detect from environment.",
    )
    gateway: Optional[GatewayConfig] = Field(default=None, description="Gateway daemon configuration")
    session: SessionConfig = Field(default_factory=SessionConfig, description="Session storage limits")
    theme: Optional[str] = Field(default=None, description="UI theme: dark (default), light, dracula")

    def resolve_secrets(self) -> None:
        """Resolve all Jinja2 template strings in the config tree.

        Walks every string field on this model and nested models/dicts,
        replacing any value containing {{ }} with its rendered result.
        Call this once before forking a daemon — macOS Keychain segfaults
        in forked processes, so secrets must be resolved in the parent.
        """
        _resolve_obj(self)

    def model_dump(self, **kwargs):
        """Override to exclude null values."""
        data = super().model_dump(**kwargs)
        # Remove null values
        return {k: v for k, v in data.items() if v is not None}


def _resolve_obj(obj) -> None:
    """Recursively resolve template strings on a Pydantic model or dict."""
    if isinstance(obj, BaseModel):
        for field_name in obj.model_fields:
            val = getattr(obj, field_name, None)
            if val is None:
                continue
            if isinstance(val, str) and "{{" in val:
                setattr(obj, field_name, render_template_if_needed(val))
            elif isinstance(val, BaseModel):
                _resolve_obj(val)
            elif isinstance(val, dict):
                _resolve_dict(val)
            elif isinstance(val, list):
                _resolve_list(val)
        # Also resolve extra fields (model_config = {"extra": "allow"})
        if hasattr(obj, "__pydantic_extra__") and obj.__pydantic_extra__:
            _resolve_dict(obj.__pydantic_extra__)
    elif isinstance(obj, dict):
        _resolve_dict(obj)


def _resolve_dict(d: dict) -> None:
    for key, val in list(d.items()):
        if isinstance(val, str) and "{{" in val:
            d[key] = render_template_if_needed(val)
        elif isinstance(val, BaseModel):
            _resolve_obj(val)
        elif isinstance(val, dict):
            _resolve_dict(val)
        elif isinstance(val, list):
            _resolve_list(val)


def _resolve_list(lst: list) -> None:
    for i, val in enumerate(lst):
        if isinstance(val, str) and "{{" in val:
            lst[i] = render_template_if_needed(val)
        elif isinstance(val, BaseModel):
            _resolve_obj(val)
        elif isinstance(val, dict):
            _resolve_dict(val)


# TODO: support the configuration from others like from a k8s ConfigMap


class ConfigManager:
    """Configuration manager for Cliver client."""

    def __init__(self, config_dir: Path, config: Optional[AppConfig] = None):
        """Initialize the configuration manager.

        Args:
            config_dir: Configuration directory
            config: Pre-loaded config (skips reading from disk if provided)
        """
        self.config_dir = config_dir
        self.config_file = self.config_dir / "config.yaml"
        self.config = config if config is not None else self._load_config()

    def _load_config(self) -> AppConfig:
        """Load configuration from YAML file.

        Returns:
            Cliver configuration
        """
        if not self.config_file.exists():
            logger.info(f"No configuration file found at {str(self.config_dir)}, using default configuration.")
            return AppConfig()

        try:
            with open(self.config_file, "r") as f:
                config_data = yaml.safe_load(f)

            # safe_load returns None for empty files
            if not config_data:
                return AppConfig()

            # Migrate old agent_name → default_agent_name
            if "agent_name" in config_data and "default_agent_name" not in config_data:
                config_data["default_agent_name"] = config_data.pop("agent_name")

            # Ensure each ModelConfig has its name set from the key
            if "models" in config_data and isinstance(config_data["models"], dict):
                for name, model in config_data["models"].items():
                    if isinstance(model, dict):
                        model["name"] = name
                        # Handle capabilities deserialization
                        if "capabilities" in model and model["capabilities"]:
                            try:
                                model["capabilities"] = {ModelCapability(cap) for cap in model["capabilities"]}
                            except ValueError as e:
                                logger.warning(f"Warning: Invalid capability in model {name}: {e}")
                                model["capabilities"] = None

            # Parse providers section
            if "providers" in config_data and isinstance(config_data["providers"], dict):
                for pname, pdata in config_data["providers"].items():
                    if isinstance(pdata, dict):
                        pdata["name"] = pname
                        config_data["providers"][pname] = ProviderConfig(**pdata)

            mcp_servers_data = config_data.get("mcpServers")
            if mcp_servers_data and isinstance(mcp_servers_data, dict):
                converted_servers = {}
                for name, server in mcp_servers_data.items():
                    if isinstance(server, dict):
                        server_dict = server.copy()
                        server_dict.pop("name", None)
                        transport = server_dict.get("transport")
                        server_config = {"name": name, **server_dict}
                        if transport == "stdio":
                            converted_servers[name] = StdioMCPServerConfig(**server_config)
                        elif transport == "sse":
                            converted_servers[name] = SSEMCPServerConfig(**server_config)
                        elif transport == "streamable_http":
                            converted_servers[name] = StreamableHttpMCPServerConfig(**server_config)
                        elif transport == "websocket":
                            converted_servers[name] = WebSocketMCPServerConfig(**server_config)
                        else:
                            raise ValueError(f"Unknown transport {transport}")
                config_data["mcpServers"] = converted_servers

            app_config = AppConfig(**config_data)
            self._link_models_to_providers(app_config)
            return app_config
        except Exception as e:
            logger.error("Error loading configuration: %s", e, stack_info=True, exc_info=True)
            raise e

    def _link_models_to_providers(self, config: AppConfig) -> None:
        """Link models to their provider configs.

        For models whose ``provider`` field matches a key in ``config.providers``,
        sets the internal ``_provider_config`` reference.  For legacy models that
        have an inline ``url`` but no matching provider, creates synthetic
        ``ProviderConfig`` objects in memory (not persisted) so that the
        resolution helpers work uniformly.
        """
        # Phase 1: link models to explicitly-declared providers
        if config.providers:
            for model in config.models.values():
                if model.provider in config.providers:
                    model._provider_config = config.providers[model.provider]

        # Phase 2: create synthetic providers for legacy inline-url models
        synthetic: dict[str, ProviderConfig] = {}
        for model in config.models.values():
            if model._provider_config is not None:
                continue
            if not model.url:
                continue
            key = f"{model.provider}|{model.url}|{model.api_key or ''}"
            if key not in synthetic:
                synthetic[key] = ProviderConfig(
                    name=f"_auto_{len(synthetic)}",
                    type=model.provider,
                    api_url=model.url,
                    api_key=model.api_key,
                )
            model._provider_config = synthetic[key]

    def _save_config(self) -> None:
        """Save configuration to YAML file."""
        try:
            if not self.config_dir.exists():
                self.config_dir.mkdir(parents=True, exist_ok=True)

            config_data = self.config.model_dump()

            # Handle providers serialization
            if "providers" in config_data:
                serialized_providers = {}
                for name, prov in self.config.providers.items():
                    serialized_providers[name] = prov.model_dump()
                config_data["providers"] = serialized_providers

            # Handle MCP servers serialization — use each server's model_dump
            # to exclude redundant name field
            if "mcpServers" in config_data:
                serialized_servers = {}
                for name, server in self.config.mcpServers.items():
                    serialized_servers[name] = server.model_dump()
                config_data["mcpServers"] = serialized_servers

            # Handle models serialization — exclude redundant name field
            if "models" in config_data:
                serialized_models = {}
                for name, model in self.config.models.items():
                    serialized_models[name] = model.model_dump()
                config_data["models"] = serialized_models

            # Handle workflow configuration serialization
            if "workflow" in config_data and self.config.workflow:
                config_data["workflow"] = self.config.workflow.model_dump()

            with open(self.config_file, "w") as f:
                yaml.dump(config_data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

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

            # Save config
            self._save_config()
            return True

        return False

    def get_mcp_server(self, name: str) -> Optional[MCPServerConfig]:
        """Get a server configuration.

        Args:
            name: Server name

        Returns:
            Server configuration if found, None otherwise
        """
        return self.config.mcpServers.get(name)

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

    def set_default_model(self, name: str) -> bool:
        """Set the default LLM model.

        Args:
            name: Model name

        Returns:
            True if default model was set, False otherwise
        """
        if name in self.config.models:
            if self.config.default_model == name:
                return True
            self.config.default_model = name
            self._save_config()
            return True
        return False

    def set_default_agent_name(self, name: str) -> None:
        """Set the default agent name."""
        self.config.default_agent_name = name
        self._save_config()

    def set_user_agent(self, user_agent: str) -> None:
        """Set the User-Agent header string."""
        self.config.user_agent = user_agent
        self._save_config()

    def add_or_update_provider(
        self,
        name: str,
        type: str,
        api_url: str,
        api_key: Optional[str] = None,
        rate_limit: Optional[RateLimitConfig] = None,
        image_url: Optional[str] = None,
        audio_url: Optional[str] = None,
    ) -> None:
        prov = self.config.providers.get(name)
        if prov:
            prov.type = type
            prov.api_url = api_url
            if api_key is not None:
                prov.api_key = api_key
            if rate_limit is not None:
                prov.rate_limit = rate_limit
            if image_url is not None:
                prov.image_url = image_url
            if audio_url is not None:
                prov.audio_url = audio_url
        else:
            self.config.providers[name] = ProviderConfig(
                name=name,
                type=type,
                api_url=api_url,
                api_key=api_key,
                rate_limit=rate_limit,
                image_url=image_url,
                audio_url=audio_url,
            )
        # Re-link models referencing this provider
        for model in self.config.models.values():
            if model.provider == name:
                model._provider_config = self.config.providers[name]
        self._save_config()

    def remove_provider(self, name: str) -> bool:
        if name in self.config.providers:
            referencing = [m.name for m in self.config.models.values() if m.provider == name]
            if referencing:
                raise ValueError(f"Cannot remove provider '{name}': used by models: {', '.join(referencing)}")
            self.config.providers.pop(name)
            self._save_config()
            return True
        return False

    def set_provider_rate_limit(self, name: str, rate_limit: Optional[RateLimitConfig]) -> bool:
        """Set or clear the rate limit on an existing provider."""
        prov = self.config.providers.get(name)
        if not prov:
            return False
        prov.rate_limit = rate_limit
        self._save_config()
        return True

    def list_providers(self) -> Dict[str, ProviderConfig]:
        return self.config.providers

    def get_llm_model(self, name: Optional[str] = None) -> Optional[ModelConfig]:
        if not name:
            name = self.config.default_model
        if self.config.models:
            return self.config.models.get(name)
        return None
