"""Comprehensive tests for all config, mcp, and llm commands after restructuring to top-level commands."""

import json
from click.testing import CliRunner
from pathlib import Path


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
    result = CliRunner().invoke(
        load_cliver,
        [
            "llm",
            "add",
            "--name",
            "test_model",
            "--provider",
            "ollama",
            "--url",
            "http://localhost:11434",
            "--name-in-provider",
            "llama3.2:latest",
            "--option",
            "temperature=0.7",
            "--option",
            "top_p=0.9",
        ],
    )
    assert result.exit_code == 0
    assert "Added LLM Model: test_model" in result.output

    # Verify the model was added correctly
    result = CliRunner().invoke(load_cliver, ["llm", "list"])
    assert result.exit_code == 0
    assert "test_model" in result.output
    assert "ollama" in result.output


def test_llm_model_set_options(load_cliver, init_config):
    """Test updating LLM model options."""
    # First add model
    CliRunner().invoke(
        load_cliver,
        [
            "llm",
            "add",
            "--name",
            "test_model",
            "--provider",
            "ollama",
            "--url",
            "http://localhost:11434",
            "--name-in-provider",
            "llama3.2:latest",
        ],
    )

    # Update model options
    result = CliRunner().invoke(
        load_cliver,
        [
            "llm",
            "set",
            "--name",
            "test_model",
            "--option",
            "temperature=0.8",
            "--option",
            "max_tokens=2048",
        ],
    )
    assert result.exit_code == 0
    assert "LLM Model: test_model updated" in result.output

    # Verify updates
    result = CliRunner().invoke(load_cliver, ["llm", "list"])
    assert result.exit_code == 0
    assert "test_model" in result.output


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

    CliRunner().invoke(
        load_cliver,
        [
            "llm",
            "add",
            "--name",
            "test_model",
            "--provider",
            "ollama",
            "--url",
            "http://localhost:11434",
            "--name-in-provider",
            "llama3.2:latest",
        ],
    )

    # Check the config file format
    config_file = init_config / "config.json"
    with open(config_file, "r") as f:
        config_data = json.load(f)

    # Verify no redundant name fields
    assert "test_stdio" in config_data["mcpServers"]
    assert "name" not in config_data["mcpServers"]["test_stdio"]

    assert "test_streamable" in config_data["mcpServers"]
    assert "name" not in config_data["mcpServers"]["test_streamable"]

    assert "test_model" in config_data["models"]
    assert "name" not in config_data["models"]["test_model"]

    # Verify no null values
    mcp_server = config_data["mcpServers"]["test_stdio"]
    assert None not in mcp_server.values()

    streamable_server = config_data["mcpServers"]["test_streamable"]
    assert None not in streamable_server.values()

    model = config_data["models"]["test_model"]
    assert None not in model.values()

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
    result = CliRunner().invoke(
        load_cliver,
        ["mcp", "remove", "--name", "test_server"]
    )
    assert result.exit_code == 0
    assert "Removed MCP server: test_server" in result.output

    # Verify it's gone
    result = CliRunner().invoke(load_cliver, ["mcp", "list"])
    assert result.exit_code == 0
    assert "test_server" not in result.output


def test_llm_model_remove(load_cliver, init_config):
    """Test removing LLM models."""
    # Add a model first
    CliRunner().invoke(
        load_cliver,
        [
            "llm",
            "add",
            "--name",
            "test_model",
            "--provider",
            "ollama",
            "--url",
            "http://localhost:11434",
        ],
    )

    # Verify it exists
    result = CliRunner().invoke(load_cliver, ["llm", "list"])
    assert result.exit_code == 0
    assert "test_model" in result.output

    # Remove it
    result = CliRunner().invoke(
        load_cliver,
        ["llm", "remove", "--name", "test_model"]
    )
    assert result.exit_code == 0
    assert "Removed LLM Model: test_model" in result.output

    # Verify it's gone
    result = CliRunner().invoke(load_cliver, ["llm", "list"])
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