"""The one-shot CLI entry point."""

from __future__ import annotations

import io
import sys
from types import SimpleNamespace

import cliver.cli as cli
import pytest


class FakeAgent:
    def __init__(self, content: str = "answer") -> None:
        self.content = content
        self.calls: list[tuple[str, str]] = []
        self.closed = False

    async def chat(self, prompt: str, **kwargs):
        self.calls.append(("chat", prompt))
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


def patch_agent(monkeypatch, fake: FakeAgent) -> None:
    monkeypatch.setattr(cli, "load_environment", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "llm_binding_from_env", lambda **kwargs: object())
    monkeypatch.setattr(cli, "build_agent", lambda binding, **kwargs: fake)


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


def test_main_requires_a_prompt_when_interactive(monkeypatch):
    patch_agent(monkeypatch, FakeAgent())
    monkeypatch.setattr(sys, "stdin", InteractiveStdin(""))

    with pytest.raises(SystemExit):
        cli.main(["--yes"])
