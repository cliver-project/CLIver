from click.testing import CliRunner


def test_list_mcp_servers_empty(load_cliver, init_config):
    result = CliRunner().invoke(load_cliver, ["mcp", "list"])
    assert result.exit_code == 0
    assert "No MCP servers configured." in result.output


def test_list_mcp_servers_simple(load_cliver, simple_mcp_server):
    result = CliRunner().invoke(load_cliver, ["mcp", "list"])
    assert result.exit_code == 0
    assert "stdio" in result.output
    assert "ocp_mcp_server_start" in result.output
    assert "arg-a" in result.output


def test_add_stdio_mcp_server(load_cliver, init_config):
    result = CliRunner().invoke(load_cliver, ["mcp", "list"])
    assert result.exit_code == 0
    assert "No MCP servers configured." in result.output

    # add it
    result = CliRunner().invoke(
        load_cliver,
        [
            "mcp",
            "add",
            "--name",
            "blender",
            "--transport",
            "stdio",
            "--command",
            "uvx",
            "--args",
            "blender-mcp",
        ],
    )
    assert result.exit_code == 0
    assert "Added MCP server: blender of transport stdio" in result.output
    result = CliRunner().invoke(load_cliver, ["mcp", "list"])
    assert result.exit_code == 0
    assert "uvx ['blender-mcp']" in result.output
    # update it
    result = CliRunner().invoke(load_cliver, ["mcp", "set", "--name", "blender", "--command", "npx"])
    assert result.exit_code == 0
    assert "Updated MCP server: blender" in result.output
    result = CliRunner().invoke(load_cliver, ["mcp", "list"])
    assert result.exit_code == 0
    assert "npx ['blender-mcp']" in result.output

    # remove it
    result = CliRunner().invoke(load_cliver, ["mcp", "remove", "--name", "blender"])
    assert result.exit_code == 0
    assert "Removed MCP server: blender" in result.output

    # list it again
    result = CliRunner().invoke(load_cliver, ["mcp", "list"])
    assert result.exit_code == 0
    assert "No MCP servers configured." in result.output


def test_list_llm_empty(load_cliver, init_config):
    result = CliRunner().invoke(load_cliver, ["model", "list"])
    assert result.exit_code == 0
    assert "No LLM Models configured." in result.output


def test_list_llm_simple(load_cliver, simple_llm_model):
    result = CliRunner().invoke(load_cliver, ["model", "list"])
    assert result.exit_code == 0
    assert "llama3.2" in result.output
    assert "ollama" in result.output


def test_add_llm_simple(load_cliver, init_config, config_manager):
    config_manager.add_or_update_provider("ollama", "openai", "http://localhost:11434")

    result = CliRunner().invoke(
        load_cliver,
        [
            "model",
            "add",
            "--name",
            "deepseek-coder",
            "--provider",
            "ollama",
        ],
    )
    assert result.exit_code == 0
    assert "Added LLM Model: deepseek-coder (set as default)" in result.output
    result = CliRunner().invoke(load_cliver, ["model", "list"])
    assert result.exit_code == 0
    assert "deepseek-coder" in result.output
    assert "ollama" in result.output

    result = CliRunner().invoke(load_cliver, ["model", "remove", "deepseek-coder"])
    assert result.exit_code == 0
    assert "Removed LLM Model: deepseek-coder" in result.output
    result = CliRunner().invoke(load_cliver, ["model", "list"])
    assert result.exit_code == 0
    assert "No LLM Models configured." in result.output
