from click.testing import CliRunner


def test_list_mcp_servers_empty(load_cliver, init_config):
    result = CliRunner().invoke(load_cliver, ["mcp", "list"])
    assert result.exit_code == 0
    assert "No MCP servers configured." in result.output


def test_list_mcp_servers_simple(load_cliver, simple_mcp_server):
    result = CliRunner().invoke(load_cliver, ["mcp", "list"])
    assert result.exit_code == 0
    assert "stdio" in result.output
    assert "ocp_mcp_server_start ['arg-a', 'arg-b']" in result.output


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
    result = CliRunner().invoke(
        load_cliver, ["mcp", "set", "--name", "blender", "--command", "npx"]
    )
    assert result.exit_code == 0
    assert "Updated MCP server: blender" in result.output
    result = CliRunner().invoke(load_cliver, ["mcp", "list"])
    assert result.exit_code == 0
    assert "npx ['blender-mcp']" in result.output

    # remove it
    result = CliRunner().invoke(
        load_cliver, ["mcp", "remove", "--name", "blender"]
    )
    assert result.exit_code == 0
    assert "Removed MCP server: blender" in result.output

    # list it again
    result = CliRunner().invoke(load_cliver, ["mcp", "list"])
    assert result.exit_code == 0
    assert "No MCP servers configured." in result.output


def test_list_llm_empty(load_cliver, init_config):
    result = CliRunner().invoke(load_cliver, ["llm", "list"])
    assert result.exit_code == 0
    assert "No LLM Models configured." in result.output


def test_list_llm_simple(load_cliver, simple_llm_model):
    result = CliRunner().invoke(load_cliver, ["llm", "list"])
    assert result.exit_code == 0
    assert "llama3.2" in result.output
    assert "ollama" in result.output


def test_add_llm_simple(load_cliver, init_config):
    # add it
    result = CliRunner().invoke(
        load_cliver,
        [
            "llm",
            "add",
            "--name",
            "deepseek",
            "--name-in-provider",
            "deepseek-coder:6.7b",
            "--provider",
            "ollama",
            "--url",
            "https://localhost:11434",
        ],
    )
    assert result.exit_code == 0
    assert "Added LLM Model: deepseek" in result.output
    result = CliRunner().invoke(load_cliver, ["llm", "list"])
    assert result.exit_code == 0
    assert "deepseek" in result.output
    assert "ollama" in result.output

    # update it
    result = CliRunner().invoke(
        load_cliver,
        [
            "llm",
            "set",
            "--name",
            "deepseek",
            "--name-in-provider",
            "deepseek-coder:6.7b",
            "--provider",
            "vllm",
        ],
    )
    assert result.exit_code == 0
    assert "LLM Model: deepseek updated" in result.output
    result = CliRunner().invoke(load_cliver, ["llm", "list"])
    assert result.exit_code == 0
    assert "deepseek" in result.output
    assert "vllm" in result.output

    # remove it
    result = CliRunner().invoke(
        load_cliver, ["llm", "remove", "--name", "deepseek"]
    )
    assert result.exit_code == 0
    assert "Removed LLM Model: deepseek" in result.output
    result = CliRunner().invoke(load_cliver, ["llm", "list"])
    assert result.exit_code == 0
    assert "No LLM Models configured." in result.output
