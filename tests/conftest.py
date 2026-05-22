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
    return cli.cliver_cli


@pytest.fixture()
def simple_mcp_server(init_config, config_manager):
    from cliver.mcp.store import MCPServerStore
    import json

    store = MCPServerStore.from_config_dir(init_config)
    store.create_server(
        "ocp",
        transport="stdio",
        command="ocp_mcp_server_start",
        args=json.dumps(["arg-a", "arg-b"]),
        envs=json.dumps({"KUBECONFIG": "~/.kube/config"}),
    )


@pytest.fixture()
def simple_llm_model(init_config, config_manager):
    from cliver.model.store import ModelStore

    store = ModelStore.from_config_dir(init_config)
    provider = store.create_provider("ollama", "ollama")
    endpoint = store.create_endpoint(provider.id, "http://localhost:11434")
    store.create_model(provider.id, endpoint.id, "llama3.2:latest")
