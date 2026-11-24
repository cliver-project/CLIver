"""The one-shot CLI entry point."""

from __future__ import annotations

import io
import sys
from types import SimpleNamespace

import cliver.cli as cli
import cliver.media as media
import pytest
from cliver.config import CLIConfig, MCPSelection
from cliver_core import CLIverMessage, ModelBinding, ProviderError
from cliver_core.media import MediaContent, MediaType
from cliver_core.provider import CLIverResponse

PNG_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="


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


class FakeBinding:
    model = "test-model"
    base_url = "https://example.com/v1"
    protocol = "openai"

    def build_provider(self):
        return None


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
    # Never read the developer's ambient config, MCP files, System One, or
    # selector settings.
    monkeypatch.setattr(cli, "load_config", lambda **kwargs: CLIConfig())
    monkeypatch.setattr(cli, "create_mcp_client", lambda config, **kwargs: MCPSelection(client=None))
    monkeypatch.setattr(cli, "llm_binding_from_env", lambda **kwargs: FakeBinding())
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
    monkeypatch.setattr(cli, "load_config", lambda **kwargs: CLIConfig())
    monkeypatch.setattr(cli, "create_mcp_client", lambda config, **kwargs: MCPSelection(client=None))
    monkeypatch.setattr(cli, "llm_binding_from_env", lambda **kwargs: FakeBinding())
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


def test_resolve_bindings_keeps_every_purpose_as_an_ordered_list(monkeypatch):
    llm_a = ModelBinding.llm(model="chat-a", api_key="k", base_url="https://a.test/v1")
    llm_b = ModelBinding.llm(model="chat-b", api_key="k", base_url="https://b.test/v1")
    image = ModelBinding.image(model="image", api_key="k", base_url="https://a.test/v1")
    system_a = ModelBinding.system_one(model="so-a", api_key="k", base_url="https://so.test")
    system_b = ModelBinding.system_one(model="so-b", api_key="k", base_url="https://so.test")
    config = CLIConfig(
        models={
            "llm": [llm_a, llm_b],
            "image": [image],
            "system_one": [system_a, system_b],
        }
    )
    args = SimpleNamespace(model=None, provider=None, base_url=None, protocol=None)

    monkeypatch.delenv("CLIVER_MODEL", raising=False)
    monkeypatch.setattr(cli, "system_one_binding_from_env", lambda: None)

    run_binding, model_bindings, primary_system_one, system_one_bindings = cli._resolve_bindings(config, args)

    assert model_bindings is not None
    assert run_binding is model_bindings
    assert model_bindings.get_all("llm") == [llm_a, llm_b]
    assert model_bindings.get_all("image") == [image]
    assert model_bindings.get_all("system_one") == [system_a, system_b]
    assert primary_system_one is system_a
    assert system_one_bindings == [system_a, system_b]


def test_resolve_bindings_keeps_the_single_binding_shape(monkeypatch):
    llm = ModelBinding.llm(model="chat", api_key="k", base_url="https://a.test/v1")
    config = CLIConfig(models={"llm": [llm]})
    args = SimpleNamespace(model=None, provider=None, base_url=None, protocol=None)

    monkeypatch.delenv("CLIVER_MODEL", raising=False)
    monkeypatch.setattr(cli, "system_one_binding_from_env", lambda: None)

    run_binding, model_bindings, system_one, system_one_bindings = cli._resolve_bindings(config, args)

    assert run_binding is llm
    assert model_bindings is None
    assert system_one is None
    assert system_one_bindings == []


def test_main_passes_the_mcp_client_to_the_agent_and_closes_it(monkeypatch, capsys):
    class FakeMCPClient:
        def __init__(self) -> None:
            self.closed = False

        async def close(self) -> None:
            self.closed = True

    fake = FakeAgent("answer")
    client = FakeMCPClient()
    captured: dict = {}

    monkeypatch.setattr(cli, "load_environment", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "load_config", lambda **kwargs: CLIConfig())
    monkeypatch.setattr(
        cli,
        "create_mcp_client",
        lambda config, **kwargs: SimpleNamespace(client=client, problems=(), summary="1 server from config"),
    )
    monkeypatch.setattr(cli, "llm_binding_from_env", lambda **kwargs: FakeBinding())
    monkeypatch.setattr(cli, "system_one_binding_from_env", lambda: None)
    monkeypatch.setattr(cli, "selector_from_env", lambda binding: None)

    def build_agent(binding, **kwargs):
        captured.update(kwargs)
        return fake

    monkeypatch.setattr(cli, "build_agent", build_agent)

    assert cli.main(["hello", "--yes"]) == 0

    assert captured["mcp_client"] is client
    assert client.closed is True
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
    assert captured["config_hint"].endswith("config.json")
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
    monkeypatch.setattr(cli, "load_config", lambda **kwargs: CLIConfig())
    monkeypatch.setattr(cli, "create_mcp_client", lambda config, **kwargs: MCPSelection(client=None))

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


def test_main_reports_provider_configuration_errors_before_the_run(monkeypatch, capsys):
    class BrokenBinding(FakeBinding):
        def build_provider(self):
            raise ValueError("the 'deepseek' provider needs an API key — pass api_key=...")

    built = []
    monkeypatch.setattr(cli, "load_environment", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "load_config", lambda **kwargs: CLIConfig())
    monkeypatch.setattr(cli, "create_mcp_client", lambda config, **kwargs: MCPSelection(client=None))
    monkeypatch.setattr(cli, "llm_binding_from_env", lambda **kwargs: BrokenBinding())
    monkeypatch.setattr(cli, "system_one_binding_from_env", lambda: None)
    monkeypatch.setattr(cli, "selector_from_env", lambda binding: None)
    monkeypatch.setattr(cli, "build_agent", lambda binding, **kwargs: built.append(binding))

    with pytest.raises(SystemExit) as excinfo:
        cli.main(["hello", "--yes"])

    assert excinfo.value.code == 2
    captured = capsys.readouterr()
    assert "cliver: error: the 'deepseek' provider needs an API key" in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""
    assert built == []


def test_help_documents_the_runtime_environment_variables(capsys):
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--help"])

    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    for name in (
        "CLIVER_MODEL",
        "CLIVER_PROVIDER",
        "CLIVER_BASE_URL",
        "CLIVER_PROTOCOL",
        "CLIVER_API_KEY",
        "CLIVER_CONF_DIR",
    ):
        assert name in out


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
            build_provider=lambda: None,
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


# ── Media subcommands ───────────────────────────────────────


def patch_media_config(monkeypatch, models: dict) -> None:
    """A media-only configuration: any chat wiring would be a bug."""

    def unexpected(*args, **kwargs):
        raise AssertionError("media commands must not need a chat model")

    monkeypatch.setattr(cli, "load_environment", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "load_config", lambda **kwargs: CLIConfig(models=models))
    monkeypatch.setattr(cli, "llm_binding_from_env", unexpected)
    monkeypatch.setattr(cli, "create_mcp_client", unexpected)
    monkeypatch.setattr(cli, "build_agent", unexpected)


def image_response(content: str = "generated") -> CLIverResponse:
    return CLIverResponse(
        message=CLIverMessage(role="assistant", content=content),
        media=[MediaContent(type=MediaType.IMAGE, data=PNG_BASE64, mime_type="image/png")],
    )


def test_main_image_generates_and_saves_into_the_current_directory(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(media, "_timestamp", lambda: "20261124-120000")
    image = ModelBinding.image(model="img", api_key="k", base_url="https://example.test/v1")
    patch_media_config(monkeypatch, {"image": [image]})
    seen: list[tuple] = []

    async def fake_run_once(binding, command, prompt, *, language=None):
        seen.append((binding, command, prompt, language))
        return image_response()

    monkeypatch.setattr(media, "run_once", fake_run_once)

    assert cli.main(["image", "a dog", "on", "the", "levee"]) == 0

    captured = capsys.readouterr()
    assert seen == [(image, "image", "a dog on the levee", None)]
    assert captured.out.strip() == "image-20261124-120000-0.png"
    assert "[media] generating image ..." in captured.err
    assert (tmp_path / "image-20261124-120000-0.png").is_file()


def test_main_saves_media_into_the_requested_directory(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(media, "_timestamp", lambda: "20261124-120000")
    image = ModelBinding.image(model="img", api_key="k", base_url="https://example.test/v1")
    patch_media_config(monkeypatch, {"image": [image]})

    async def fake_run_once(binding, command, prompt, *, language=None):
        return image_response()

    monkeypatch.setattr(media, "run_once", fake_run_once)

    assert cli.main(["image", "a dog", "--out", str(tmp_path / "out")]) == 0

    assert capsys.readouterr().out.strip() == "out/image-20261124-120000-0.png"
    assert (tmp_path / "out" / "image-20261124-120000-0.png").is_file()


def test_main_reports_a_missing_media_model(monkeypatch, tmp_path, capsys):
    config = CLIConfig(models={}, paths=(), config_dir=tmp_path)
    monkeypatch.setattr(cli, "load_environment", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "load_config", lambda **kwargs: config)

    with pytest.raises(SystemExit) as excinfo:
        cli.main(["image", "a dog"])

    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "cliver: error: no image model configured" in err
    assert 'add "image" under "models"' in err
    assert str(tmp_path / "config.json") in err
    assert "Traceback" not in err


def test_main_chat_escapes_prompts_that_begin_with_a_command_word(monkeypatch, capsys):
    fake = FakeAgent("the answer")
    patch_agent(monkeypatch, fake)

    assert cli.main(["chat", "image", "a", "dog", "--yes"]) == 0

    assert fake.calls == [("chat", "image a dog")]
    assert "the answer" in capsys.readouterr().out


def test_main_keeps_a_quoted_prompt_that_starts_with_a_command_word(monkeypatch):
    fake = FakeAgent("the answer")
    patch_agent(monkeypatch, fake)

    # The subcommand word must be its own argument: one quoted prompt stays chat.
    assert cli.main(["image classification is hard", "--yes"]) == 0

    assert fake.calls == [("chat", "image classification is hard")]


def test_main_asr_prints_the_transcript_with_the_language_hint(monkeypatch, tmp_path, capsys):
    asr = ModelBinding.asr(model="whisper", api_key="k", base_url="https://example.test/v1")
    patch_media_config(monkeypatch, {"asr": [asr]})
    seen: list[tuple] = []

    async def fake_run_once(binding, command, prompt, *, language=None):
        seen.append((binding, command, prompt, language))
        return CLIverResponse(message=CLIverMessage(role="assistant", content="the transcript"))

    monkeypatch.setattr(media, "run_once", fake_run_once)

    assert cli.main(["asr", "meeting.mp3", "--language", "zh"]) == 0

    captured = capsys.readouterr()
    assert seen == [(asr, "asr", "meeting.mp3", "zh")]
    assert captured.out.strip() == "the transcript"
    assert "[media] transcribing meeting.mp3 ..." in captured.err
    assert list(tmp_path.iterdir()) == []


def test_main_reports_a_media_failure(monkeypatch, tmp_path, capsys):
    image = ModelBinding.image(model="img", api_key="k", base_url="https://example.test/v1")
    patch_media_config(monkeypatch, {"image": [image]})

    async def fake_run_once(binding, command, prompt, *, language=None):
        return CLIverResponse(
            message=CLIverMessage(role="assistant", content="x image generation failed: boom"),
            retryable=True,
        )

    monkeypatch.setattr(media, "run_once", fake_run_once)

    assert cli.main(["image", "a dog"]) == 1

    err = capsys.readouterr().err
    assert "cliver: error: x image generation failed: boom" in err
    assert list(tmp_path.iterdir()) == []


def test_main_media_command_requires_a_prompt(monkeypatch):
    patch_agent(monkeypatch, FakeAgent())
    monkeypatch.setattr(sys, "stdin", PipedStdin(""))

    with pytest.raises(SystemExit) as excinfo:
        cli.main(["image"])

    assert excinfo.value.code == 2
