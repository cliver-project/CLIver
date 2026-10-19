"""Tool-confirmation hook: host UI when installed, builtin input otherwise."""

from types import SimpleNamespace

from cliver_core import util


class _UI:
    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.calls: list[tuple[str, list[str] | None]] = []

    def ask_input(self, prompt: str, choices: list[str] | None = None) -> str:
        self.calls.append((prompt, choices))
        return self.answer


def test_confirm_uses_installed_cli_instance(monkeypatch) -> None:
    ui = _UI("yes")
    monkeypatch.setattr(util, "_cli_instance", SimpleNamespace(ui=ui))

    assert util._confirm_tool_execution("Run it? (y/n): ") is True
    assert ui.calls == [("Run it? (y/n): ", ["y", "yes", "n", "no"])]


def test_confirm_falls_back_to_builtin_input(monkeypatch) -> None:
    monkeypatch.setattr(util, "_cli_instance", None)
    monkeypatch.setattr("builtins.input", lambda _prompt: "n")

    assert util._confirm_tool_execution("Run it? (y/n): ") is False


def test_confirm_handles_eof(monkeypatch) -> None:
    monkeypatch.setattr(util, "_cli_instance", None)

    def _eof(_prompt: str) -> str:
        raise EOFError

    monkeypatch.setattr("builtins.input", _eof)
    assert util._confirm_tool_execution() is False


def test_set_cli_instance_round_trips(monkeypatch) -> None:
    monkeypatch.setattr(util, "_cli_instance", None)
    instance = object()

    util.set_cli_instance(instance)
    assert util._cli_instance is instance

    util.set_cli_instance(None)
    assert util._cli_instance is None
