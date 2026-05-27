"""Tests for lab cell chat functionality — stream_chat_response and SSE events."""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock


# -- Helpers -------------------------------------------------------------------


async def _async_iter(*items):
    """Produce an async iterable from items (to fake agent.stream())."""
    for item in items:
        yield item


async def _async_error(exc):
    """Produce an async iterable that immediately raises *exc*."""
    raise exc
    yield  # pragma: no cover


def _make_done_chunk(text: str, artifacts=None, status="completed"):
    """Build a mock chunk with chunk_type='done'."""
    chunk = MagicMock()
    chunk.chunk_type = "done"
    chunk.text = ""
    chunk.final_result = MagicMock()
    chunk.final_result.text = text
    chunk.final_result.artifacts = artifacts or []
    chunk.final_result.status = status
    chunk.final_result.error = None
    return chunk


def _make_chunk(chunk_type: str, text: str = ""):
    """Build a minimal mock chunk."""
    chunk = MagicMock()
    chunk.chunk_type = chunk_type
    chunk.text = text
    chunk.final_result = None
    return chunk


# -- Tests ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_chat_response_events():
    """stream_chat_response yields properly structured SSE events."""
    from cliver.lab.chat import stream_chat_response

    mock_agent = MagicMock()
    mock_agent.stream = MagicMock(return_value=_async_iter(
        _make_chunk("thinking", "Let me think..."),
        _make_chunk("text", "Hello "),
        _make_chunk("text", "from agent"),
        _make_done_chunk("Hello from agent"),
    ))

    mock_sm = MagicMock()
    mock_sm.append_turn = MagicMock()

    events = []
    async for event in stream_chat_response(
        mock_agent, "test prompt", mock_sm, "sess_01",
    ):
        events.append(event)

    # Verify event structure
    assert events[0] == {"type": "thinking", "content": "Let me think..."}
    assert events[1] == {"type": "text", "content": "Hello "}
    assert events[2] == {"type": "text", "content": "from agent"}
    assert events[3]["type"] == "done"
    assert events[3]["text"] == "Hello from agent"

    # Verify assistant turn was persisted
    mock_sm.append_turn.assert_called_once_with("sess_01", "assistant", "Hello from agent")


@pytest.mark.asyncio
async def test_stream_chat_response_error():
    """On agent error, yields error event."""
    from cliver.lab.chat import stream_chat_response

    mock_agent = MagicMock()
    mock_agent.stream = MagicMock(return_value=_async_error(RuntimeError("Agent offline")))

    mock_sm = MagicMock()

    events = []
    async for event in stream_chat_response(
        mock_agent, "prompt", mock_sm, "sess_02",
    ):
        events.append(event)

    assert events[-1]["type"] == "error"
    assert "Agent offline" in events[-1]["message"]


@pytest.mark.asyncio
async def test_stream_chat_response_with_tools():
    """Tool call events are forwarded correctly."""
    from cliver.lab.chat import stream_chat_response

    mock_agent = MagicMock()
    mock_agent.stream = MagicMock(return_value=_async_iter(
        _make_chunk("tool_use", '{"tool": "read_file", "args": {"path": "/x"}}'),
        _make_chunk("tool_result", '{"result": "content"}'),
        _make_chunk("text", "Result is ready"),
        _make_done_chunk("Done"),
    ))

    mock_sm = MagicMock()
    mock_sm.append_turn = MagicMock()

    events = []
    async for event in stream_chat_response(
        mock_agent, "test", mock_sm, "sess_03",
    ):
        events.append(event)

    assert events[0]["type"] == "tool_use"
    assert events[1]["type"] == "tool_result"
    assert events[2]["type"] == "text"
    assert events[3]["type"] == "done"


@pytest.mark.asyncio
async def test_stream_chat_response_json_output():
    """JSON output format adds data field to done event."""
    from cliver.lab.chat import stream_chat_response

    mock_agent = MagicMock()
    mock_agent.stream = MagicMock(return_value=_async_iter(
        # result_text is built from text chunks, not final_result.text
        _make_chunk("text", '{"key": "value"}'),
        _make_done_chunk('{"key": "value"}'),
    ))

    mock_sm = MagicMock()
    mock_sm.append_turn = MagicMock()

    events = []
    async for event in stream_chat_response(
        mock_agent, "prompt", mock_sm, "sess_04", output_format="json",
    ):
        events.append(event)

    assert len(events) == 2
    assert events[0]["type"] == "text"
    assert events[0]["content"] == '{"key": "value"}'
    assert events[1]["type"] == "done"
    assert events[1]["data"] == {"key": "value"}


@pytest.mark.asyncio
async def test_stream_chat_response_with_artifacts():
    """Artifacts from final_result are passed through."""
    from cliver.lab.chat import stream_chat_response

    mock_artifact = MagicMock(path="/tmp/out.csv", media_type="text/csv", size=1024)

    mock_agent = MagicMock()
    mock_agent.stream = MagicMock(return_value=_async_iter(
        _make_done_chunk("Output with file", artifacts=[mock_artifact]),
    ))

    mock_sm = MagicMock()
    mock_sm.append_turn = MagicMock()

    events = []
    async for event in stream_chat_response(
        mock_agent, "prompt", mock_sm, "sess_05",
    ):
        events.append(event)

    assert events[0]["type"] == "done"
    assert events[0]["artifacts"] == [
        {"path": "/tmp/out.csv", "media_type": "text/csv", "size": 1024},
    ]
