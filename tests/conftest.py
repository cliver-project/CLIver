import click
import pytest
from cliver.config import ConfigManager


@pytest.fixture()
def load_cliver() -> click.Group:
    from cliver import cli
    cli.loads_commands()
    return cli.cliver


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
def simple_mcp_server(init_config, config_manager):
    config_manager.add_or_update_stdio_mcp_server(
        "ocp", "ocp_mcp_server_start", ["arg-a", "arg-b"], {"KUBECONFIG": "~/.kube/config"})
