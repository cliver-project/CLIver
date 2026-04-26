"""Comprehensive tests for all config, mcp, and model commands after restructuring to top-level commands."""

import yaml
from click.testing import CliRunner


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


def test_llm_model_add_with_options(load_cliver, init_config, config_manager):
    """Test adding LLM model with options."""
    config_manager.add_or_update_provider("ollama", "ollama", "http://localhost:11434")
    result = CliRunner().invoke(
        load_cliver,
        [
            "model",
            "add",
            "--name",
            "llama3.2",
            "--provider",
            "ollama",
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


def test_llm_model_set_options(load_cliver, init_config, config_manager):
    """Test updating LLM model options."""
    config_manager.add_or_update_provider("ollama", "ollama", "http://localhost:11434")
    CliRunner().invoke(
        load_cliver,
        ["model", "add", "--name", "llama3.2", "--provider", "ollama"],
    )

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


def test_llm_model_remove(load_cliver, init_config, config_manager):
    """Test removing LLM models."""
    config_manager.add_or_update_provider("ollama", "ollama", "http://localhost:11434")
    CliRunner().invoke(
        load_cliver,
        ["model", "add", "--name", "test_model", "--provider", "ollama"],
    )

    result = CliRunner().invoke(load_cliver, ["model", "list"])
    assert result.exit_code == 0
    assert "test_model" in result.output

    result = CliRunner().invoke(load_cliver, ["model", "remove", "ollama/test_model"])
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
    assert result.output  # Should show some configuration content


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

    data = {"models": {"qwen": {"api_key": "{{ keyring('myservice', 'mykey') }}"}}}
    _mask_secrets(data)
    assert data["models"]["qwen"]["api_key"] == "{{ keyring('myservice', 'mykey') }}"

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
