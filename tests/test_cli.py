import pytest
from click.testing import CliRunner
from cliver.cli import cli, register_commands


@pytest.fixture(autouse=True)
def setup_cli():
    register_commands()


def test_batch_greet_and_exit():
    # let's pretend we added a "greet" leaf command for testing
    result = CliRunner().invoke(cli, ["config", "list"])
    assert result.exit_code == 0


def test_interactive_repl_exit_immediately(monkeypatch):
    monkeypatch.setattr("prompt_toolkit.prompt", lambda *args, **kwargs: "exit")
    result = CliRunner().invoke(cli, [])
    assert result.exit_code == 0
    assert "Entering interactive shell" in result.output
