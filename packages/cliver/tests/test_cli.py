"""The one-shot CLI entry point."""

from __future__ import annotations

import io
import sys
from types import SimpleNamespace

import cliver.cli as cli
import pytest
from cliver_core import ProviderError


class FakeAgent:
    def __init__(self, content: str = "answer", *, error: Exception | None = None) -> None:
        self.content = content
        self.error = error
        self.calls: list[tuple[str, str]] = []
        self.closed = False

    async def chat(self, prompt: str, **kwargs):
        self.calls.append(("chat", prompt))
        if self.error is not None:
            raise self.error
        return SimpleNamespace(message=SimpleNamespace(content=self.content))

    async def run_skill(self, name: str, prompt: str, **kwargs):
        self.calls.append(("run_skill", f"{name}:{prompt}"))
        return SimpleNamespace(message=SimpleNamespace(content=self.content))

    async def aclose(self) -> None:
        self.closed = True


class PipedStdin(io.StringIO):
    def isatty(self) -> bool:
        return False


class InteractiveStdin(io.StringIO):
    def isatty(self) -> bool:
        return True


class TtyStringIO(io.StringIO):
    def isatty(self) -> bool:
        return True


def patch_agent(monkeypatch, fake: FakeAgent) -> None:
    monkeypatch.setattr(cli, "load_environment", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "llm_binding_from_env", lambda **kwargs: object())
    # Never read the developer's ambient System One / selector settings.
    monkeypatch.setattr(cli, "system_one_binding_from_env", lambda: None)
    monkeypatch.setattr(cli, "selector_from_env", lambda binding: None)
    monkeypatch.setattr(cli, "build_agent", lambda binding, **kwargs: fake)


class FakeLocalProvider:
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


def test_main_wires_system_one_and_closes_the_local_provider(monkeypatch):
    fake = FakeAgent()
    provider = FakeLocalProvider()
    system_one = SimpleNamespace(provider=provider)
    selector = object()
    captured: dict = {}

    monkeypatch.setattr(cli, "load_environment", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "llm_binding_from_env", lambda **kwargs: object())
    monkeypatch.setattr(cli, "system_one_binding_from_env", lambda: system_one)
    monkeypatch.setattr(cli, "selector_from_env", lambda binding: selector)

    def build_agent(binding, **kwargs):
        captured.update(kwargs)
        return fake

    monkeypatch.setattr(cli, "build_agent", build_agent)

    assert cli.main(["hello", "--yes"]) == 0

    assert captured["system_one_binding"] is system_one
    assert captured["candidate_selector"] is selector
    assert provider.closed is True
    assert fake.closed is True


def test_main_prints_chat_response(monkeypatch, capsys):
    fake = FakeAgent("the answer")
    patch_agent(monkeypatch, fake)

    assert cli.main(["hello", "--yes"]) == 0

    assert fake.calls == [("chat", "hello")]
    assert fake.closed is True
    captured = capsys.readouterr()
    assert "the answer" in captured.out


def test_main_uses_run_skill_when_requested(monkeypatch, capsys):
    fake = FakeAgent("skill answer")
    patch_agent(monkeypatch, fake)

    assert cli.main(["hello", "--skill", "write-plan", "--yes"]) == 0

    assert fake.calls == [("run_skill", "write-plan:hello")]
    assert "skill answer" in capsys.readouterr().out


def test_main_reads_piped_stdin(monkeypatch, capsys):
    fake = FakeAgent()
    patch_agent(monkeypatch, fake)
    monkeypatch.setattr(sys, "stdin", PipedStdin("from stdin"))

    assert cli.main(["--yes"]) == 0

    assert fake.calls == [("chat", "from stdin")]


def test_main_starts_the_repl_when_interactive_without_a_prompt(monkeypatch):
    fake = FakeAgent()
    patch_agent(monkeypatch, fake)
    monkeypatch.setattr(sys, "stdin", InteractiveStdin(""))
    monkeypatch.setattr(sys, "stdout", TtyStringIO())
    captured: dict = {}

    async def fake_repl(agent, **kwargs):
        captured["agent"] = agent
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(cli, "run_repl", fake_repl)

    assert cli.main(["--yes"]) == 0

    assert captured["agent"] is fake
    assert captured["max_iterations"] == 50
    assert fake.closed is True


def test_main_requires_a_prompt_when_piped_stdin_is_empty(monkeypatch):
    patch_agent(monkeypatch, FakeAgent())
    monkeypatch.setattr(sys, "stdin", PipedStdin(""))

    with pytest.raises(SystemExit):
        cli.main(["--yes"])


def test_main_requires_a_prompt_for_a_skill_in_an_interactive_session(monkeypatch):
    patch_agent(monkeypatch, FakeAgent())
    monkeypatch.setattr(sys, "stdin", InteractiveStdin(""))
    monkeypatch.setattr(sys, "stdout", TtyStringIO())

    with pytest.raises(SystemExit):
        cli.main(["--skill", "write-plan", "--yes"])


def test_main_reports_a_configuration_error_without_a_traceback(monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_environment", lambda *args, **kwargs: None)

    def fail(**kwargs):
        raise ValueError("No chat model configured: pass --model or set CLIVER_MODEL.")

    monkeypatch.setattr(cli, "llm_binding_from_env", fail)

    with pytest.raises(SystemExit) as excinfo:
        cli.main(["hello", "--yes"])

    assert excinfo.value.code == 2
    captured = capsys.readouterr()
    assert "cliver: error: No chat model configured" in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""


def test_main_reports_a_missing_optional_backend_without_a_traceback(monkeypatch, capsys):
    patch_agent(monkeypatch, FakeAgent())

    def fail():
        raise ImportError("install cliver-core[local] to use the laya backend")

    monkeypatch.setattr(cli, "system_one_binding_from_env", fail)

    with pytest.raises(SystemExit) as excinfo:
        cli.main(["hello", "--yes"])

    assert excinfo.value.code == 2
    captured = capsys.readouterr()
    assert "cliver: error: install cliver-core[local]" in captured.err
    assert "Traceback" not in captured.err


def test_main_reports_a_provider_failure_without_a_traceback(monkeypatch, capsys):
    fake = FakeAgent(error=ProviderError("HTTP 401: invalid api key"))
    patch_agent(monkeypatch, fake)

    assert cli.main(["hello", "--yes"]) == 1

    captured = capsys.readouterr()
    assert "cliver: error: HTTP 401: invalid api key" in captured.err
    assert "Traceback" not in captured.err
    assert fake.closed is True


def test_main_reports_the_endpoint_with_a_provider_failure(monkeypatch, capsys):
    fake = FakeAgent(error=ProviderError("Error code: 404"))
    patch_agent(monkeypatch, fake)
    monkeypatch.setattr(
        cli,
        "llm_binding_from_env",
        lambda **kwargs: SimpleNamespace(
            base_url="https://api.deepseek.com/anthropic",
            protocol="openai",
            model="deepseek-v4-flash",
        ),
    )

    assert cli.main(["hello", "--yes"]) == 1

    err = capsys.readouterr().err
    assert "cliver: error: Error code: 404" in err
    assert "endpoint: https://api.deepseek.com/anthropic (openai protocol)" in err


def test_main_returns_130_when_interrupted(monkeypatch, capsys):
    fake = FakeAgent(error=KeyboardInterrupt())
    patch_agent(monkeypatch, fake)

    assert cli.main(["hello", "--yes"]) == 130

    captured = capsys.readouterr()
    assert "interrupted" in captured.err
    assert fake.closed is True
