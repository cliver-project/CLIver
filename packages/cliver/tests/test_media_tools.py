"""Agent media tools: advertising, saving, failure, and policy behavior."""

from __future__ import annotations

import cliver.media as media
import pytest
from cliver.app import build_agent
from cliver.media_tools import build_media_tools
from cliver_core import CLIverMessage, ModelBinding, ModelBindings
from cliver_core.media import MediaContent, MediaType
from cliver_core.policy import ExecutionPolicy, policy_scope
from cliver_core.provider import CLIverResponse

PNG_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
AUDIO_BASE64 = "SUQzBAAAAAAAI1RTU0UAAAAPAAADTGF2ZjU4Ljc2LjEwMAAAAAAAAAAAAAAA"


def _llm() -> ModelBinding:
    return ModelBinding.llm(model="m", api_key="k", base_url="https://example.test/v1")


def _binding(purpose: str, model: str = "m") -> ModelBinding:
    return getattr(ModelBinding, purpose)(model=model, api_key="k", base_url="https://example.test/v1")


def _response(*media_items: MediaContent, content: str = "generated", retryable: bool | None = None) -> CLIverResponse:
    return CLIverResponse(
        message=CLIverMessage(role="assistant", content=content),
        media=list(media_items) or None,
        retryable=retryable,
    )


def _png() -> MediaContent:
    return MediaContent(type=MediaType.IMAGE, data=PNG_BASE64, mime_type="image/png")


def _mp3() -> MediaContent:
    return MediaContent(type=MediaType.AUDIO, data=AUDIO_BASE64, mime_type="audio/mpeg")


def _tool(tools, name: str):
    return next(tool for tool in tools if tool.name == name)


class FakeAgent:
    """Records media calls and returns canned responses."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.responses: dict[str, CLIverResponse] = {}

    async def image(self, prompt):
        self.calls.append(("image", prompt))
        return self.responses["image"]

    async def video(self, prompt):
        self.calls.append(("video", prompt))
        return self.responses["video"]

    async def tts(self, prompt):
        self.calls.append(("tts", prompt))
        return self.responses["tts"]

    async def audio(self, prompt):
        self.calls.append(("audio", prompt))
        return self.responses["audio"]

    async def asr(self, audio, *, language=None):
        self.calls.append(("asr", audio, language))
        return self.responses["asr"]


def test_advertises_only_configured_generation_purposes():
    agent = FakeAgent()
    bindings = ModelBindings(llm=_llm(), image=_binding("image"), tts=_binding("tts"))

    names = [tool.name for tool in build_media_tools(agent, bindings)]

    assert names == ["GenerateImage", "TextToSpeech"]


def test_generation_tools_ask_before_running():
    agent = FakeAgent()
    bindings = ModelBindings(llm=_llm(), image=_binding("image"), video=_binding("video"))

    tools = build_media_tools(agent, bindings)

    assert {tool.safety for tool in tools} == {"mutating"}


def test_text_to_speech_accepts_the_legacy_audio_purpose():
    agent = FakeAgent()
    bindings = ModelBindings(llm=_llm(), audio=_binding("tts"))

    assert [tool.name for tool in build_media_tools(agent, bindings)] == ["TextToSpeech"]


async def test_generate_image_saves_with_cli_naming_and_reports_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(media, "_timestamp", lambda: "20261125-120000")
    agent = FakeAgent()
    agent.responses["image"] = _response(_png())
    tool = _tool(build_media_tools(agent, ModelBindings(llm=_llm(), image=_binding("image"))), "GenerateImage")

    result = await tool.execute_async(prompt="a red panda")

    assert agent.calls == [("image", "a red panda")]
    assert (tmp_path / "image-20261125-120000-0.png").read_bytes().startswith(b"\x89PNG")
    assert result == [{"text": "Saved 1 file(s):\nimage-20261125-120000-0.png"}]


async def test_text_to_speech_calls_the_legacy_audio_method(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(media, "_timestamp", lambda: "20261125-120000")
    agent = FakeAgent()
    agent.responses["audio"] = _response(_mp3())
    bindings = ModelBindings(llm=_llm(), audio=_binding("tts"))
    tool = _tool(build_media_tools(agent, bindings), "TextToSpeech")

    result = await tool.execute_async(prompt="hello")

    assert agent.calls == [("audio", "hello")]
    assert (tmp_path / "tts-20261125-120000-0.mp3").exists()
    assert result == [{"text": "Saved 1 file(s):\ntts-20261125-120000-0.mp3"}]


async def test_generation_failure_response_saves_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    agent = FakeAgent()
    agent.responses["image"] = _response(content="x image generation failed: boom", retryable=True)
    tool = _tool(build_media_tools(agent, ModelBindings(llm=_llm(), image=_binding("image"))), "GenerateImage")

    result = await tool.execute_async(prompt="a red panda")

    assert result == [{"error": "x image generation failed: boom"}]
    assert list(tmp_path.iterdir()) == []


async def test_generation_without_media_reports_the_response_text(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    agent = FakeAgent()
    agent.responses["image"] = _response(content="the model returned text only")
    tool = _tool(build_media_tools(agent, ModelBindings(llm=_llm(), image=_binding("image"))), "GenerateImage")

    result = await tool.execute_async(prompt="a red panda")

    assert result == [{"error": "the model returned text only"}]
    assert list(tmp_path.iterdir()) == []


async def test_generation_respects_write_scopes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    agent = FakeAgent()
    agent.responses["image"] = _response(_png())
    tool = _tool(build_media_tools(agent, ModelBindings(llm=_llm(), image=_binding("image"))), "GenerateImage")

    with policy_scope(ExecutionPolicy(cwd=tmp_path, write_paths=[allowed])):
        result = await tool.execute_async(prompt="a red panda")

    assert "outside the allowed write paths" in result[0]["error"]
    assert not list(tmp_path.glob("image-*"))


async def test_generation_lets_unexpected_tool_errors_propagate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    agent = FakeAgent()

    async def boom(prompt):
        raise RuntimeError("provider exploded")

    agent.image = boom
    tool = _tool(build_media_tools(agent, ModelBindings(llm=_llm(), image=_binding("image"))), "GenerateImage")

    with pytest.raises(RuntimeError, match="provider exploded"):
        await tool.execute_async(prompt="a red panda")


def test_transcription_is_safe_and_only_advertised_when_configured():
    agent = FakeAgent()
    bindings = ModelBindings(llm=_llm(), asr=_binding("asr"))

    tools = build_media_tools(agent, bindings)

    assert [tool.name for tool in tools] == ["TranscribeAudio"]
    assert tools[0].safety == "safe"


async def test_transcribe_audio_rebases_relative_paths_against_the_pinned_cwd(tmp_path, monkeypatch):
    pinned = tmp_path / "project"
    pinned.mkdir()
    (pinned / "meeting.mp3").write_bytes(b"ID3")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    agent = FakeAgent()
    agent.responses["asr"] = _response(content="  hello world  ")
    tool = _tool(build_media_tools(agent, ModelBindings(llm=_llm(), asr=_binding("asr"))), "TranscribeAudio")

    with policy_scope(ExecutionPolicy(cwd=pinned)):
        result = await tool.execute_async(source="meeting.mp3", language="zh")

    assert agent.calls == [("asr", str(pinned / "meeting.mp3"), "zh")]
    assert result == [{"text": "hello world"}]


async def test_transcribe_audio_passes_urls_through_and_leaves_language_unset(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    agent = FakeAgent()
    agent.responses["asr"] = _response(content="remote words")
    tool = _tool(build_media_tools(agent, ModelBindings(llm=_llm(), asr=_binding("asr"))), "TranscribeAudio")

    result = await tool.execute_async(source="https://example.test/a.mp3")

    assert agent.calls == [("asr", "https://example.test/a.mp3", None)]
    assert result == [{"text": "remote words"}]


async def test_transcribe_audio_reports_empty_transcripts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    agent = FakeAgent()
    agent.responses["asr"] = _response(content="   ")
    tool = _tool(build_media_tools(agent, ModelBindings(llm=_llm(), asr=_binding("asr"))), "TranscribeAudio")

    result = await tool.execute_async(source="a.mp3")

    assert result == [{"error": "no transcript returned"}]


async def test_transcribe_audio_reports_provider_failures(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    agent = FakeAgent()
    agent.responses["asr"] = _response(content="asr failed: boom", retryable=True)
    tool = _tool(build_media_tools(agent, ModelBindings(llm=_llm(), asr=_binding("asr"))), "TranscribeAudio")

    result = await tool.execute_async(source="a.mp3")

    assert result == [{"error": "asr failed: boom"}]


async def test_transcribe_audio_respects_read_scopes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "a.mp3").write_bytes(b"ID3")
    agent = FakeAgent()
    agent.responses["asr"] = _response(content="words")
    tool = _tool(build_media_tools(agent, ModelBindings(llm=_llm(), asr=_binding("asr"))), "TranscribeAudio")

    with policy_scope(ExecutionPolicy(cwd=tmp_path, read_paths=[allowed])):
        result = await tool.execute_async(source="outside/a.mp3")

    assert "outside the allowed read paths" in result[0]["error"]
    assert agent.calls == []


def test_build_agent_registers_media_tools_for_configured_purposes(tmp_path):
    bindings = ModelBindings(llm=_llm(), image=_binding("image"), asr=_binding("asr"))

    agent = build_agent(
        bindings,
        cwd=tmp_path,
        config_dir=tmp_path / "conf",
        home=tmp_path / "home",
    )

    names = agent.tool_registry.tool_names
    assert "GenerateImage" in names
    assert "TranscribeAudio" in names
    assert "GenerateVideo" not in names
    assert "TextToSpeech" not in names
    assert {"Bash", "Read", "Write"} <= set(names)


def test_build_agent_registers_no_media_tools_without_media_bindings(tmp_path):
    agent = build_agent(
        _llm(),
        cwd=tmp_path,
        config_dir=tmp_path / "conf",
        home=tmp_path / "home",
    )

    media_names = {"GenerateImage", "GenerateVideo", "TextToSpeech", "TranscribeAudio"}
    assert media_names.isdisjoint(agent.tool_registry.tool_names)


async def test_build_agent_media_tool_calls_the_agent_and_saves(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(media, "_timestamp", lambda: "20261125-120000")
    bindings = ModelBindings(llm=_llm(), image=_binding("image"))
    agent = build_agent(bindings, cwd=tmp_path, config_dir=tmp_path / "conf", home=tmp_path / "home")
    calls: list[str] = []

    async def fake_image(prompt):
        calls.append(prompt)
        return _response(_png())

    agent.image = fake_image
    tool = agent.tool_registry.get("GenerateImage")

    result = await tool.execute_async(prompt="a red panda")

    assert calls == ["a red panda"]
    assert result == [{"text": "Saved 1 file(s):\nimage-20261125-120000-0.png"}]
    assert (tmp_path / "image-20261125-120000-0.png").exists()
