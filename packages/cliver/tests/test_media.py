"""Media commands: model selection, saving, and REPL plumbing."""

from __future__ import annotations

import io
import json

import cliver.media as media
import pytest
from cliver.config import CLIConfig
from cliver.repl import run_repl
from cliver_core import CLIverMessage, CLIverMessageChunk, ModelBinding, ModelBindings
from cliver_core.media import MediaContent, MediaType
from cliver_core.provider import CLIverResponse

PNG_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="


def _binding(purpose: str, model: str = "m") -> ModelBinding:
    return getattr(ModelBinding, purpose)(model=model, api_key="k", base_url="https://example.test/v1")


def _saved_response(*mime_types: str) -> CLIverResponse:
    return CLIverResponse(
        message=CLIverMessage(role="assistant", content="generated"),
        media=[MediaContent(type=MediaType.IMAGE, data=PNG_BASE64, mime_type=mime) for mime in mime_types],
    )


def test_binding_for_uses_the_primary_binding_of_the_purpose():
    first = _binding("image", "first")
    second = _binding("image", "second")
    config = CLIConfig(models={"image": [first, second]})

    purpose, binding = media.binding_for(config, "image")

    assert purpose == "image"
    assert binding is first


def test_binding_for_audio_prefers_tts_and_accepts_the_legacy_purpose():
    tts = _binding("tts", "tts-model")
    legacy = _binding("tts", "legacy-model")

    assert media.binding_for(CLIConfig(models={"tts": [tts], "audio": [legacy]}), "audio") == ("tts", tts)
    assert media.binding_for(CLIConfig(models={"audio": [legacy]}), "audio") == ("audio", legacy)
    assert media.binding_for(CLIConfig(models={"audio": [legacy]}), "tts") == ("audio", legacy)


def test_binding_for_reports_the_missing_purpose_with_the_config_file(tmp_path):
    config = CLIConfig(models={}, paths=(tmp_path / "config.json",), config_dir=tmp_path)

    with pytest.raises(ValueError) as excinfo:
        media.binding_for(config, "image")

    message = str(excinfo.value)
    assert 'no image model configured: add "image" under "models"' in message
    assert str(tmp_path / "config.json") in message


def test_binding_for_names_the_conventional_config_path_when_none_was_read(tmp_path):
    config = CLIConfig(models={}, paths=(), config_dir=tmp_path)

    assert media.config_hint(config) == str(tmp_path / "config.json")


def test_save_media_writes_unique_named_files(tmp_path, monkeypatch):
    monkeypatch.setattr(media, "_timestamp", lambda: "20261124-120000")

    first = media.save_media(_saved_response("image/png"), "image", tmp_path)
    second = media.save_media(_saved_response("image/png"), "image", tmp_path)

    assert [path.name for path in first] == ["image-20261124-120000-0.png"]
    assert [path.name for path in second] == ["image-20261124-120000-0-1.png"]
    assert first[0].read_bytes().startswith(b"\x89PNG")


def test_save_media_names_files_by_the_real_format(tmp_path, monkeypatch):
    monkeypatch.setattr(media, "_timestamp", lambda: "20261124-120000")

    saved = media.save_media(_saved_response("image/jpeg"), "image", tmp_path)

    assert saved[0].suffix == ".jpg"


def test_failure_message_only_reports_failed_responses():
    ok = CLIverResponse(message=CLIverMessage(role="assistant", content="Generated 1 image(s)."))
    failed = CLIverResponse(
        message=CLIverMessage(role="assistant", content="x image generation failed: boom"),
        retryable=True,
    )

    assert media.failure_message(ok) is None
    assert media.failure_message(failed) == "x image generation failed: boom"


async def test_run_once_dispatches_to_the_matching_capability(monkeypatch):
    calls: list[tuple] = []

    async def fake_image(*, prompt, binding):
        calls.append(("image", prompt, binding))
        return "image-response"

    async def fake_video(*, prompt, binding):
        calls.append(("video", prompt, binding))
        return "video-response"

    async def fake_speech(*, text, binding):
        calls.append(("tts", text, binding))
        return "tts-response"

    async def fake_transcribe(*, audio, binding, language):
        calls.append(("asr", audio, language))
        return "asr-response"

    monkeypatch.setattr(media, "generate_image", fake_image)
    monkeypatch.setattr(media, "generate_video", fake_video)
    monkeypatch.setattr(media, "text_to_speech", fake_speech)
    monkeypatch.setattr(media, "transcribe_audio", fake_transcribe)
    binding = object()

    assert await media.run_once(binding, "image", "a dog") == "image-response"
    assert await media.run_once(binding, "video", "a drone shot") == "video-response"
    assert await media.run_once(binding, "audio", "hello") == "tts-response"
    assert await media.run_once(binding, "asr", "in.mp3", language="zh") == "asr-response"
    assert calls == [
        ("image", "a dog", binding),
        ("video", "a drone shot", binding),
        ("tts", "hello", binding),
        ("asr", "in.mp3", "zh"),
    ]


async def test_run_in_agent_uses_the_bound_method_for_the_purpose():
    class FakeAgent:
        def __init__(self) -> None:
            self.calls: list[tuple] = []

        async def image(self, prompt):
            self.calls.append(("image", prompt))
            return "image"

        async def video(self, prompt):
            self.calls.append(("video", prompt))
            return "video"

        async def tts(self, prompt):
            self.calls.append(("tts", prompt))
            return "tts"

        async def audio(self, prompt):
            self.calls.append(("audio", prompt))
            return "audio"

        async def asr(self, audio, *, language=None):
            self.calls.append(("asr", audio, language))
            return "asr"

    agent = FakeAgent()

    assert await media.run_in_agent(agent, "image", "image", "a dog") == "image"
    assert await media.run_in_agent(agent, "video", "video", "a drone shot") == "video"
    assert await media.run_in_agent(agent, "audio", "tts", "hello") == "tts"
    assert await media.run_in_agent(agent, "audio", "audio", "hello") == "audio"
    assert await media.run_in_agent(agent, "asr", "asr", "in.mp3", language="zh") == "asr"
    assert agent.calls == [
        ("image", "a dog"),
        ("video", "a drone shot"),
        ("tts", "hello"),
        ("audio", "hello"),
        ("asr", "in.mp3", "zh"),
    ]


# ── REPL slash commands ─────────────────────────────────────


def scripted(*lines: str):
    iterator = iter(lines)

    def read(_prompt: str) -> str:
        try:
            return next(iterator)
        except StopIteration:
            raise EOFError from None

    return read


class FakeAgent:
    def __init__(self, *, bindings: ModelBindings | None = None, answer: str = "chat answer") -> None:
        self.bindings = bindings if bindings is not None else ModelBindings()
        self.answer = answer
        self.responses: dict[str, CLIverResponse] = {}
        self.calls: list[tuple] = []
        self.conversations: list[list] = []

    async def image(self, prompt):
        self.calls.append(("image", prompt))
        return self.responses["image"]

    async def tts(self, prompt):
        self.calls.append(("tts", prompt))
        return self.responses["tts"]

    async def audio(self, prompt):
        self.calls.append(("audio", prompt))
        return self.responses["audio"]

    async def asr(self, audio, *, language=None):
        self.calls.append(("asr", audio, language))
        return self.responses["asr"]

    async def stream(self, prompt, *, conversation=None, max_iterations=50):
        self.calls.append(("chat", prompt))
        self.conversations.append(list(conversation or []))
        yield CLIverMessageChunk(content=self.answer)
        yield CLIverMessageChunk(
            vendor_ext={"__llm_trace__": json.dumps([{"message": {"role": "assistant", "content": self.answer}}])}
        )


async def test_repl_image_generates_saves_and_prints_the_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(media, "_timestamp", lambda: "20261124-120000")
    agent = FakeAgent(bindings=ModelBindings(image=_binding("image")))
    agent.responses["image"] = _saved_response("image/png")
    out, err = io.StringIO(), io.StringIO()

    code = await run_repl(agent, input_fn=scripted("/image a dog", "/exit"), output=out, error=err)

    assert code == 0
    assert agent.calls == [("image", "a dog")]
    assert (tmp_path / "image-20261124-120000-0.png").is_file()
    assert "image-20261124-120000-0.png" in out.getvalue()
    assert "[media] generating image ..." in err.getvalue()


async def test_repl_audio_uses_the_tts_binding_and_names_the_file_audio(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(media, "_timestamp", lambda: "20261124-120000")
    agent = FakeAgent(bindings=ModelBindings(tts=_binding("tts")))
    agent.responses["tts"] = _saved_response("audio/mpeg")
    out = io.StringIO()

    await run_repl(agent, input_fn=scripted("/audio hello there", "/exit"), output=out, error=io.StringIO())

    assert agent.calls == [("tts", "hello there")]
    assert (tmp_path / "audio-20261124-120000-0.mp3").is_file()
    assert "audio-20261124-120000-0.mp3" in out.getvalue()


async def test_repl_tts_is_an_alias_for_audio(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(media, "_timestamp", lambda: "20261124-120000")
    agent = FakeAgent(bindings=ModelBindings(tts=_binding("tts")))
    agent.responses["tts"] = _saved_response("audio/mpeg")

    await run_repl(agent, input_fn=scripted("/tts hello", "/exit"), output=io.StringIO(), error=io.StringIO())

    assert agent.calls == [("tts", "hello")]
    assert (tmp_path / "tts-20261124-120000-0.mp3").is_file()


async def test_repl_asr_prints_the_transcript_with_the_language_hint(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    agent = FakeAgent(bindings=ModelBindings(asr=_binding("asr")))
    agent.responses["asr"] = CLIverResponse(message=CLIverMessage(role="assistant", content="the transcript"))
    out = io.StringIO()

    await run_repl(agent, input_fn=scripted("/asr meeting.mp3 --language zh", "/exit"), output=out, error=io.StringIO())

    assert agent.calls == [("asr", "meeting.mp3", "zh")]
    assert "the transcript" in out.getvalue()
    assert list(tmp_path.iterdir()) == []


async def test_repl_media_does_not_enter_the_conversation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    agent = FakeAgent(bindings=ModelBindings(image=_binding("image")))
    agent.responses["image"] = _saved_response("image/png")

    await run_repl(
        agent, input_fn=scripted("/image a dog", "hello", "/exit"), output=io.StringIO(), error=io.StringIO()
    )

    assert [call[0] for call in agent.calls] == ["image", "chat"]
    assert agent.conversations[0] == []


async def test_repl_reports_a_missing_media_model_and_keeps_the_session(tmp_path):
    agent = FakeAgent()
    err = io.StringIO()
    hint = str(tmp_path / "config.json")

    code = await run_repl(
        agent,
        input_fn=scripted("/image a dog", "hello", "/exit"),
        output=io.StringIO(),
        error=err,
        config_hint=hint,
    )

    assert code == 0
    assert 'no image model configured: add "image" under "models"' in err.getvalue()
    assert hint in err.getvalue()
    assert [call[0] for call in agent.calls] == ["chat"]


async def test_repl_media_failure_is_reported_and_the_session_survives(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    agent = FakeAgent(bindings=ModelBindings(image=_binding("image")))
    agent.responses["image"] = CLIverResponse(
        message=CLIverMessage(role="assistant", content="x image generation failed: boom"),
        retryable=True,
    )
    err = io.StringIO()

    code = await run_repl(agent, input_fn=scripted("/image a dog", "hello", "/exit"), output=io.StringIO(), error=err)

    assert code == 0
    assert "error: x image generation failed: boom" in err.getvalue()
    assert [call[0] for call in agent.calls] == ["image", "chat"]


async def test_repl_media_commands_need_a_prompt(tmp_path):
    agent = FakeAgent()
    err = io.StringIO()

    await run_repl(agent, input_fn=scripted("/image", "/asr", "/exit"), output=io.StringIO(), error=err)

    assert agent.calls == []
    assert "image needs a prompt" in err.getvalue()
    assert "asr needs an audio file or URL" in err.getvalue()


async def test_repl_help_lists_the_media_commands():
    out = io.StringIO()

    await run_repl(FakeAgent(), input_fn=scripted("/help", "/exit"), output=out, error=io.StringIO())

    help_text = out.getvalue()
    for command in ("/image", "/video", "/audio", "/asr"):
        assert command in help_text
