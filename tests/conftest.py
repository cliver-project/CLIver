import asyncio

import pytest

from cliver.cli import Cliver
from cliver.config import ConfigManager


@pytest.fixture(scope="session")
def event_loop():
    """Create a new event loop for each test case."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture()
def init_config(tmp_path, monkeypatch):
    config_path = tmp_path / "config"
    config_path.mkdir()
    monkeypatch.setenv("CLIVER_CONF_DIR", str(config_path))
    return config_path


@pytest.fixture()
def config_manager(init_config):
    return ConfigManager(init_config)


@pytest.fixture()
def test_cliver(init_config, config_manager):
    return Cliver()


@pytest.fixture()
def load_cliver(init_config, config_manager):
    from cliver import cli

    cli.loads_commands()
    return cli.cliver


@pytest.fixture()
def simple_mcp_server(init_config, config_manager):
    config_manager.add_or_update_stdio_mcp_server(
        "ocp",
        "ocp_mcp_server_start",
        ["arg-a", "arg-b"],
        {"KUBECONFIG": "~/.kube/config"},
    )


@pytest.fixture()
def simple_llm_model(init_config, config_manager):
    config_manager.add_or_update_llm_model("llama3.2", "ollama", "xx", "http://localhost:11434", "", "llama3.2:latest")
