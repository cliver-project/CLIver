"""Tests for Agent data models."""

from dataclasses import asdict


def test_artifact_creation():
    from cliver.agent import Artifact

    a = Artifact(path="/tmp/image.png", media_type="image/png", size=1024, description="Generated image")
    assert a.path == "/tmp/image.png"
    assert a.media_type == "image/png"
    assert a.size == 1024
    assert a.description == "Generated image"


def test_artifact_defaults():
    from cliver.agent import Artifact

    a = Artifact(path="/tmp/file.txt", media_type="text/plain", size=None)
    assert a.description == ""
    assert a.size is None


def test_agent_result_completed():
    from cliver.agent import AgentResult, Artifact

    art = Artifact(path="/tmp/out.pdf", media_type="application/pdf", size=2048)
    result = AgentResult(
        text="Here is the analysis.",
        status="completed",
        artifacts=[art],
        duration_ms=1500,
        model="deepseek/deepseek-r1",
        token_usage={"input": 100, "output": 200},
    )
    assert result.text == "Here is the analysis."
    assert result.status == "completed"
    assert len(result.artifacts) == 1
    assert result.artifacts[0].path == "/tmp/out.pdf"
    assert result.duration_ms == 1500
    assert result.error is None
    assert result.raw is None


def test_agent_result_error():
    from cliver.agent import AgentResult

    result = AgentResult(text="", status="error", error="Connection refused", duration_ms=300)
    assert result.status == "error"
    assert result.error == "Connection refused"
    assert result.artifacts == []


def test_agent_result_timeout():
    from cliver.agent import AgentResult

    result = AgentResult(text="", status="timeout", error="Timeout after 300s", duration_ms=300000)
    assert result.status == "timeout"


def test_agent_chunk_text():
    from cliver.agent import AgentChunk

    chunk = AgentChunk(text="Hello ", chunk_type="text")
    assert chunk.text == "Hello "
    assert chunk.chunk_type == "text"
    assert chunk.artifact is None
    assert chunk.final_result is None


def test_agent_chunk_done():
    from cliver.agent import AgentChunk, AgentResult

    result = AgentResult(text="Full response", status="completed")
    chunk = AgentChunk(chunk_type="done", final_result=result)
    assert chunk.chunk_type == "done"
    assert chunk.final_result.text == "Full response"


def test_agent_chunk_artifact():
    from cliver.agent import AgentChunk, Artifact

    art = Artifact(path="/tmp/img.png", media_type="image/png", size=512)
    chunk = AgentChunk(chunk_type="artifact", artifact=art)
    assert chunk.chunk_type == "artifact"
    assert chunk.artifact.path == "/tmp/img.png"


def test_agent_result_is_dataclass():
    from cliver.agent import AgentResult

    result = AgentResult(text="test", status="completed")
    d = asdict(result)
    assert d["text"] == "test"
    assert d["status"] == "completed"
