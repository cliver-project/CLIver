from click.testing import CliRunner


def test_list_mcp_servers_empty(load_cliver, init_config):
    result = CliRunner().invoke(load_cliver, ["config", "mcp", "list"])
    assert result.exit_code == 0
    assert "No MCP servers configured." in result.output


def test_list_mcp_servers_simple(load_cliver, simple_mcp_server):
    result = CliRunner().invoke(load_cliver, ["config", "mcp", "list"])
    assert result.exit_code == 0
    assert "stdio" in result.output
    assert "ocp_mcp_server_start ['arg-a', 'arg-b']" in result.output


def test_add_stdio_mcp_server(load_cliver, init_config):
    result = CliRunner().invoke(load_cliver, ["config", "mcp", "list"])
    assert result.exit_code == 0
    assert "No MCP servers configured." in result.output
    result = CliRunner().invoke(load_cliver, ["config", "mcp", "add",
                                              "--name", "blender", "--transport", "stdio", "--command", "uvx", "--args", "blender-mcp"])
    assert result.exit_code == 0
    assert "Added MCP server: blender of transport stdio" in result.output
    result = CliRunner().invoke(load_cliver, ["config", "mcp", "list"])
    assert result.exit_code == 0
    assert "uvx ['blender-mcp']" in result.output


def test_list_llm_empty(load_cliver, init_config):
    result = CliRunner().invoke(load_cliver, ["config", "llm", "list"])
    assert result.exit_code == 0
    assert "No LLM Models configured." in result.output


def test_add_llm_simple(load_cliver, init_config):
    result = CliRunner().invoke(load_cliver, ["config", "llm", "add",
                                              "--name", "llama3",
                                              "--name-in-provider", "llama3.2",
                                              "--provider", "ollama",
                                              "--url", "https://localhost:11434"])
    assert result.exit_code == 0
    assert "Added LLM Model: llama3" in result.output
    result = CliRunner().invoke(load_cliver, ["config", "llm", "list"])
    assert result.exit_code == 0
    assert "llama3.2" in result.output
    assert "ollama" in result.output
