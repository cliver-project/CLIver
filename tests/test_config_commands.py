"""Comprehensive tests for all config, mcp, and model commands after restructuring to top-level commands."""

import yaml
from click.testing import CliRunner

from cliver.config import ConfigManager


def test_mcp_server_add_stdio_with_env(load_cliver, init_config):
    """Test adding stdio MCP server with environment variables."""
    result = CliRunner().invoke(
        load_cliver,
        [
            "mcp",
            "add",
            "--name",
            "test_stdio",
            "--transport",
            "stdio",
            "--command",
            "echo",
            "--env",
            "KEY1=VALUE1",
            "--env",
            "KEY2=VALUE2",
        ],
    )
    assert result.exit_code == 0
    assert "Added MCP server: test_stdio of transport stdio" in result.output

    # Verify the server was added correctly
    result = CliRunner().invoke(load_cliver, ["mcp", "list"])
    assert result.exit_code == 0
    assert "test_stdio" in result.output
    assert "KEY1" in result.output
    assert "VALUE1" in result.output
    assert "KEY2" in result.output
    assert "VALUE2" in result.output


def test_mcp_server_add_streamable_with_headers(load_cliver, init_config):
    """Test adding streamable MCP server with headers."""
    result = CliRunner().invoke(
        load_cliver,
        [
            "mcp",
            "add",
            "--name",
            "test_streamable",
            "--transport",
            "streamable",
            "--url",
            "http://localhost:8080",
            "--header",
            "Authorization=Bearer token",
            "--header",
            "Content-Type=application/json",
        ],
    )
    assert result.exit_code == 0
    assert "Added MCP server: test_streamable of transport streamable" in result.output

    # Verify the server was added correctly
    result = CliRunner().invoke(load_cliver, ["mcp", "list"])
    assert result.exit_code == 0
    assert "test_streamable" in result.output
    assert "Authorization" in result.output
    assert "Bearer token" in result.output
    assert "Content-Type" in result.output
    assert "application/json" in result.output


def test_mcp_server_add_sse_with_headers(load_cliver, init_config):
    """Test adding SSE MCP server with headers (deprecated but still supported)."""
    result = CliRunner().invoke(
        load_cliver,
        [
            "mcp",
            "add",
            "--name",
            "test_sse",
            "--transport",
            "sse",
            "--url",
            "http://localhost:8080",
            "--header",
            "Authorization=Bearer token",
        ],
    )
    assert result.exit_code == 0
    assert "Warning: SSE transport is deprecated" in result.output
    assert "Added MCP server: test_sse of transport sse" in result.output


def test_mcp_server_set_env_and_headers(load_cliver, init_config):
    """Test updating MCP server with environment variables and headers."""
    # First add servers
    CliRunner().invoke(
        load_cliver,
        [
            "mcp",
            "add",
            "--name",
            "test_stdio",
            "--transport",
            "stdio",
            "--command",
            "echo",
        ],
    )

    CliRunner().invoke(
        load_cliver,
        [
            "mcp",
            "add",
            "--name",
            "test_streamable",
            "--transport",
            "streamable",
            "--url",
            "http://localhost:8080",
        ],
    )

    # Update stdio server with env variables
    result = CliRunner().invoke(
        load_cliver,
        [
            "mcp",
            "set",
            "--name",
            "test_stdio",
            "--env",
            "NEW_KEY=NEW_VALUE",
        ],
    )
    assert result.exit_code == 0
    assert "Updated MCP server: test_stdio" in result.output

    # Update streamable server with headers
    result = CliRunner().invoke(
        load_cliver,
        [
            "mcp",
            "set",
            "--name",
            "test_streamable",
            "--header",
            "X-Custom-Header=custom-value",
        ],
    )
    assert result.exit_code == 0
    assert "Updated MCP server: test_streamable" in result.output

    # Verify updates
    result = CliRunner().invoke(load_cliver, ["mcp", "list"])
    assert result.exit_code == 0
    assert "NEW_KEY" in result.output
    assert "NEW_VALUE" in result.output
    assert "X-Custom-Header" in result.output
    assert "custom-value" in result.output


def test_mcp_server_invalid_env_format(load_cliver, init_config):
    """Test handling of invalid environment variable format."""
    result = CliRunner().invoke(
        load_cliver,
        [
            "mcp",
            "add",
            "--name",
            "test_stdio",
            "--transport",
            "stdio",
            "--command",
            "echo",
            "--env",
            "INVALID_FORMAT",  # Missing = sign
        ],
    )
    assert result.exit_code == 0  # Should not fail, just warn
    assert "Warning: Invalid option format 'INVALID_FORMAT'" in result.output


def test_mcp_server_invalid_header_format(load_cliver, init_config):
    """Test handling of Invalid option format."""
    result = CliRunner().invoke(
        load_cliver,
        [
            "mcp",
            "add",
            "--name",
            "test_streamable",
            "--transport",
            "streamable",
            "--url",
            "http://localhost:8080",
            "--header",
            "INVALID_FORMAT",  # Missing = sign
        ],
    )
    assert result.exit_code == 0  # Should not fail, just warn
    assert "Warning: Invalid option format 'INVALID_FORMAT'" in result.output


def test_llm_model_add_with_options(load_cliver, init_config):
    """Test adding LLM model with options."""
    from cliver.model.store import ModelStore

    store = ModelStore.from_config_dir(init_config)
    provider = store.create_provider("ollama", "ollama")
    endpoint = store.create_endpoint(provider.id, "http://localhost:11434")

    result = CliRunner().invoke(
        load_cliver,
        [
            "model",
            "add",
            "--name",
            "llama3.2",
            "--provider",
            provider.id,
            "--endpoint",
            endpoint.id,
            "--option",
            "temperature=0.7",
            "--option",
            "top_p=0.9",
        ],
    )
    assert result.exit_code == 0
    assert "Added LLM Model: ollama/llama3.2" in result.output

    result = CliRunner().invoke(load_cliver, ["model", "list"])
    assert result.exit_code == 0


def test_llm_model_set_options(load_cliver, init_config):
    """Test updating LLM model options."""
    from cliver.model.store import ModelStore

    store = ModelStore.from_config_dir(init_config)
    provider = store.create_provider("ollama", "ollama")
    endpoint = store.create_endpoint(provider.id, "http://localhost:11434")
    store.create_model(provider.id, endpoint.id, "llama3.2")

    result = CliRunner().invoke(
        load_cliver,
        [
            "model",
            "set",
            "--name",
            "ollama/llama3.2",
            "--option",
            "temperature=0.8",
            "--option",
            "max_tokens=2048",
        ],
    )
    assert result.exit_code == 0
    assert "LLM Model: ollama/llama3.2 updated" in result.output

    result = CliRunner().invoke(load_cliver, ["model", "list"])
    assert result.exit_code == 0


def test_config_file_format(load_cliver, init_config):
    """Test that config file format is clean without redundant fields and null values."""
    # Add various configurations
    CliRunner().invoke(
        load_cliver,
        [
            "mcp",
            "add",
            "--name",
            "test_stdio",
            "--transport",
            "stdio",
            "--command",
            "echo",
            "--env",
            "KEY=VALUE",
        ],
    )

    CliRunner().invoke(
        load_cliver,
        [
            "mcp",
            "add",
            "--name",
            "test_streamable",
            "--transport",
            "streamable",
            "--url",
            "http://localhost:8080",
            "--header",
            "Authorization=Bearer token",
        ],
    )

    # Check the config file format
    config_file = init_config / "config.yaml"
    with open(config_file, "r") as f:
        config_data = yaml.safe_load(f)

    # Verify no redundant name fields
    assert "test_stdio" in config_data["mcpServers"]
    assert "name" not in config_data["mcpServers"]["test_stdio"]

    assert "test_streamable" in config_data["mcpServers"]
    assert "name" not in config_data["mcpServers"]["test_streamable"]

    # Verify no null values
    mcp_server = config_data["mcpServers"]["test_stdio"]
    assert None not in mcp_server.values()

    streamable_server = config_data["mcpServers"]["test_streamable"]
    assert None not in streamable_server.values()

    # Verify no top-level models section (models are nested in providers)
    assert "models" not in config_data

    # Verify secrets is not in config if None
    assert "secrets" not in config_data


def test_mcp_server_remove(load_cliver, init_config):
    """Test removing MCP servers."""
    # Add a server first
    CliRunner().invoke(
        load_cliver,
        [
            "mcp",
            "add",
            "--name",
            "test_server",
            "--transport",
            "stdio",
            "--command",
            "echo",
        ],
    )

    # Verify it exists
    result = CliRunner().invoke(load_cliver, ["mcp", "list"])
    assert result.exit_code == 0
    assert "test_server" in result.output

    # Remove it
    result = CliRunner().invoke(load_cliver, ["mcp", "remove", "--name", "test_server"])
    assert result.exit_code == 0
    assert "Removed MCP server: test_server" in result.output

    # Verify it's gone
    result = CliRunner().invoke(load_cliver, ["mcp", "list"])
    assert result.exit_code == 0
    assert "test_server" not in result.output


def test_llm_model_remove(load_cliver, init_config):
    """Test removing LLM models."""
    from cliver.model.store import ModelStore

    store = ModelStore.from_config_dir(init_config)
    provider = store.create_provider("ollama", "ollama")
    endpoint = store.create_endpoint(provider.id, "http://localhost:11434")
    store.create_model(provider.id, endpoint.id, "test_model")

    canonical = "ollama/test_model"

    result = CliRunner().invoke(load_cliver, ["model", "list"])
    assert result.exit_code == 0
    assert canonical in result.output

    result = CliRunner().invoke(load_cliver, ["model", "remove", canonical])
    assert result.exit_code == 0
    assert "Removed LLM Model: ollama/test_model" in result.output

    result = CliRunner().invoke(load_cliver, ["model", "list"])
    assert result.exit_code == 0
    assert "test_model" not in result.output


def test_config_validation(load_cliver, init_config):
    """Test configuration validation command."""
    result = CliRunner().invoke(load_cliver, ["config", "validate"])
    assert result.exit_code == 0
    assert "✓ Configuration is valid" in result.output


def test_config_show(load_cliver, init_config):
    """Test showing configuration."""
    result = CliRunner().invoke(load_cliver, ["config", "show"])
    assert result.exit_code == 0
    assert result.output
    assert "Error showing configuration" not in result.output
    assert "General Settings" in result.output
    assert "Config file:" in result.output


def test_config_path(load_cliver, init_config):
    """Test showing configuration file path."""
    result = CliRunner().invoke(load_cliver, ["config", "path"])
    assert result.exit_code == 0
    assert "Configuration file path:" in result.output


# ---------------------------------------------------------------------------
# API key masking
# ---------------------------------------------------------------------------


def test_mask_secrets_plain_text():
    """Plain text api_key should be masked."""
    from cliver.commands.config import _mask_secrets

    data = {"models": {"qwen": {"api_key": "sk-abcdef123456789xyz"}}}
    _mask_secrets(data)
    assert data["models"]["qwen"]["api_key"] == "sk-***xyz"


def test_mask_secrets_template_expression():
    """Jinja2 template expressions should NOT be masked."""
    from cliver.commands.config import _mask_secrets

    # Key names (without {{ }}) should be masked like plain text
    data = {"models": {"qwen": {"api_key": "my_api_key_name"}}}
    _mask_secrets(data)
    assert data["models"]["qwen"]["api_key"] == "my_***ame"

    # Template expressions should NOT be masked
    data2 = {"models": {"qwen": {"api_key": "{{ env.OPENAI_API_KEY }}"}}}
    _mask_secrets(data2)
    assert data2["models"]["qwen"]["api_key"] == "{{ env.OPENAI_API_KEY }}"


def test_mask_secrets_short_key():
    """Very short keys are fully masked."""
    from cliver.commands.config import _mask_secrets

    data = {"models": {"qwen": {"api_key": "short"}}}
    _mask_secrets(data)
    assert data["models"]["qwen"]["api_key"] == "***"


def test_mask_secrets_none_key():
    """None api_key should be left as-is."""
    from cliver.commands.config import _mask_secrets

    data = {"models": {"qwen": {"api_key": None}}}
    _mask_secrets(data)
    assert data["models"]["qwen"]["api_key"] is None


def test_mask_secrets_no_api_key():
    """Dicts without api_key should be untouched."""
    from cliver.commands.config import _mask_secrets

    data = {"models": {"qwen": {"url": "http://localhost"}}}
    _mask_secrets(data)
    assert data["models"]["qwen"]["url"] == "http://localhost"


def test_config_show_with_default_agent(load_cliver, init_config, config_manager):
    """config show should display the configured default_agent."""
    config_manager.config.default_agent = "coder"
    config_manager._save_config()

    result = CliRunner().invoke(load_cliver, ["config", "show"])
    assert result.exit_code == 0
    assert "Default Agent" in result.output
    assert "coder" in result.output
    assert "Error showing configuration" not in result.output


# ──────────────────────────────────────────────────────────────────────────────
# config set
# ──────────────────────────────────────────────────────────────────────────────


def test_config_set_user_agent(load_cliver, init_config):
    result = CliRunner().invoke(load_cliver, ["config", "set", "--user-agent", "TestBot/1.0"])
    assert result.exit_code == 0
    cfg = ConfigManager(init_config).config
    assert cfg.user_agent == "TestBot/1.0"


def test_config_set_user_agent_empty(load_cliver, init_config):
    result = CliRunner().invoke(load_cliver, ["config", "set", "--user-agent", ""])
    assert result.exit_code == 0
    assert "Usage: config set" in result.output


# ──────────────────────────────────────────────────────────────────────────────
# config validate
# ──────────────────────────────────────────────────────────────────────────────


def test_config_validate_valid(load_cliver, init_config):
    result = CliRunner().invoke(load_cliver, ["config", "validate"])
    assert result.exit_code == 0
    assert "Configuration is valid" in result.output


# ──────────────────────────────────────────────────────────────────────────────
# config theme
# ──────────────────────────────────────────────────────────────────────────────


def test_config_theme_show(load_cliver, init_config):
    result = CliRunner().invoke(load_cliver, ["config", "theme"])
    assert result.exit_code == 0
    assert "Current theme:" in result.output


def test_config_theme_set_invalid(load_cliver, init_config):
    result = CliRunner().invoke(load_cliver, ["config", "theme", "nonexistent"])
    assert result.exit_code == 0
    assert "Unknown theme" in result.output


def test_config_theme_set_valid(load_cliver, init_config):
    result = CliRunner().invoke(load_cliver, ["config", "theme", "light"])
    assert result.exit_code == 0
    cfg = ConfigManager(init_config).config
    assert cfg.theme == "light"


# ──────────────────────────────────────────────────────────────────────────────
# config rate-limit
# ──────────────────────────────────────────────────────────────────────────────


def test_config_rate_limit_show_no_provider(load_cliver, init_config):
    result = CliRunner().invoke(load_cliver, ["config", "rate-limit", "nonexistent"])
    assert result.exit_code == 0


def test_config_rate_limit_set_and_show(load_cliver, init_config):
    from cliver.model.store import ModelStore

    store = ModelStore.from_config_dir(init_config)
    provider = store.create_provider("testprov", "openai")
    store.create_endpoint(provider.id, "https://test.example.com")

    CliRunner().invoke(load_cliver, ["config", "rate-limit", "testprov", "100/1m"])

    store2 = ModelStore.from_config_dir(init_config)
    providers = store2.list_providers()
    prov = next((p for p in providers if p.name == "testprov"), None)
    assert prov is not None
    assert prov.rate_limit is not None
    assert prov.rate_limit["requests"] == 100
    assert prov.rate_limit["period"] == "1m"

    result = CliRunner().invoke(load_cliver, ["config", "rate-limit", "testprov"])
    assert result.exit_code == 0
    assert "100/1m" in result.output


# ──────────────────────────────────────────────────────────────────────────────
# model list
# ──────────────────────────────────────────────────────────────────────────────


def test_model_list_empty(load_cliver, init_config):
    result = CliRunner().invoke(load_cliver, ["model", "list"])
    assert result.exit_code == 0


def test_model_list_with_models(load_cliver, init_config):
    from cliver.model.store import ModelStore

    store = ModelStore.from_config_dir(init_config)
    provider = store.create_provider("ollama", "ollama")
    endpoint = store.create_endpoint(provider.id, "http://localhost:11434")
    store.create_model(provider.id, endpoint.id, "llama3.2:latest")
    store.create_model(provider.id, endpoint.id, "codellama:7b")

    result = CliRunner().invoke(load_cliver, ["model", "list"])
    assert result.exit_code == 0
    assert "ollama/llama3.2:latest" in result.output
    assert "ollama/codellama:7b" in result.output


# ──────────────────────────────────────────────────────────────────────────────
# model default
# ──────────────────────────────────────────────────────────────────────────────


def test_model_default_show_none(load_cliver, init_config):
    result = CliRunner().invoke(load_cliver, ["model", "default"])
    assert result.exit_code == 0


def test_model_default_set_and_show(load_cliver, init_config):
    from cliver.model.store import ModelStore

    store = ModelStore.from_config_dir(init_config)
    provider = store.create_provider("openai", "openai")
    endpoint = store.create_endpoint(provider.id, "https://api.openai.com")
    store.create_model(provider.id, endpoint.id, "gpt-4o")

    CliRunner().invoke(load_cliver, ["model", "default", "openai/gpt-4o"])

    default = store.get_default_model()
    assert default is not None
    assert default.name == "gpt-4o"

    result = CliRunner().invoke(load_cliver, ["model", "default"])
    assert result.exit_code == 0
    assert "openai/gpt-4o" in result.output


def test_model_default_set_nonexistent(load_cliver, init_config):
    result = CliRunner().invoke(load_cliver, ["model", "default", "nonexistent/model"])
    assert result.exit_code != 0 or "not found" in result.output.lower()


# ──────────────────────────────────────────────────────────────────────────────
# model set
# ──────────────────────────────────────────────────────────────────────────────


def test_model_set_options(load_cliver, init_config):
    from cliver.model.store import ModelStore

    store = ModelStore.from_config_dir(init_config)
    provider = store.create_provider("openai", "openai")
    endpoint = store.create_endpoint(provider.id, "https://api.openai.com")
    m = store.create_model(provider.id, endpoint.id, "gpt-4o")

    CliRunner().invoke(
        load_cliver,
        ["model", "set", "--name", "openai/gpt-4o", "--option", "temperature=0.5", "--option", "max_tokens=4096"],
    )

    updated = store.get_model(m.id)
    assert updated is not None
    assert updated.options is not None
    assert updated.options.get("temperature") == 0.5
    assert updated.options.get("max_tokens") == 4096


# ──────────────────────────────────────────────────────────────────────────────
# provider list
# ──────────────────────────────────────────────────────────────────────────────


def test_provider_list_empty(load_cliver, init_config):
    result = CliRunner().invoke(load_cliver, ["provider", "list"])
    assert result.exit_code == 0


def test_provider_list_with_providers(load_cliver, init_config, config_manager):
    config_manager.add_or_update_provider("openai", "openai", "https://api.openai.com")
    config_manager.add_or_update_provider("anthropic", "anthropic", "https://api.anthropic.com")
    result = CliRunner().invoke(load_cliver, ["provider", "list"])
    assert result.exit_code == 0
    assert "openai" in result.output
    assert "anthropic" in result.output


# ──────────────────────────────────────────────────────────────────────────────
# provider add
# ──────────────────────────────────────────────────────────────────────────────


def test_provider_add_basic(load_cliver, init_config):
    CliRunner().invoke(
        load_cliver,
        [
            "provider",
            "add",
            "--name",
            "myprov",
            "--type",
            "openai",
            "--api-url",
            "https://my.api.com",
            "--api-key",
            "sk-test",
        ],
    )
    cfg = ConfigManager(init_config).config
    prov = cfg.providers["myprov"]
    assert prov.type == "openai"
    assert prov.api_url == "https://my.api.com"


# ──────────────────────────────────────────────────────────────────────────────
# provider remove
# ──────────────────────────────────────────────────────────────────────────────


def test_provider_remove_no_models(load_cliver, init_config):
    cm = ConfigManager(init_config)
    cm.add_or_update_provider("orphan", "openai", "https://orphan.example.com")

    result = CliRunner().invoke(load_cliver, ["provider", "remove", "--name", "orphan"])
    assert result.exit_code == 0
    cfg = ConfigManager(init_config).config
    assert "orphan" not in cfg.providers
