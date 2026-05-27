"""Tests for CliverAgent wrapping AgentCore."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from cliver.agent import AgentResult
from cliver.config import AgentConfig


def _make_mock_core(response_content="Test response"):
    core = MagicMock()
    message = MagicMock()
    message.content = response_content
    message.additional_kwargs = {}
    core.process_user_input = AsyncMock(return_value=message)
    return core


@pytest.mark.asyncio
async def test_cliver_agent_run_basic():
    from cliver.agents.cliver_agent import CliverAgent

    core = _make_mock_core("Hello from AgentCore")
    config = AgentConfig(type="cliver", model="test/model")
    agent = CliverAgent(name="default", config=config, agent_core=core)

    result = await agent._do_run("What is 2+2?")

    assert isinstance(result, AgentResult)
    assert result.text == "Hello from AgentCore"
    assert result.status == "completed"
    assert result.model == "test/model"
    assert result.duration_ms >= 0
    core.process_user_input.assert_called_once()


@pytest.mark.asyncio
async def test_cliver_agent_run_with_images():
    from cliver.agents.cliver_agent import CliverAgent

    core = _make_mock_core("Image analyzed")
    config = AgentConfig(type="cliver")
    agent = CliverAgent(name="default", config=config, agent_core=core)

    result = await agent._do_run("Describe this", images=["/tmp/img.png"])

    assert result.text == "Image analyzed"
    call_kwargs = core.process_user_input.call_args
    assert call_kwargs.kwargs.get("images") == ["/tmp/img.png"]


@pytest.mark.asyncio
async def test_cliver_agent_run_error():
    from cliver.agents.cliver_agent import CliverAgent

    core = MagicMock()
    core.process_user_input = AsyncMock(side_effect=RuntimeError("LLM unavailable"))
    config = AgentConfig(type="cliver")
    agent = CliverAgent(name="default", config=config, agent_core=core)

    result = await agent._do_run("test")

    assert result.status == "error"
    assert "LLM unavailable" in result.error


@pytest.mark.asyncio
async def test_cliver_agent_with_role():
    from cliver.agents.cliver_agent import CliverAgent

    core = _make_mock_core("OK")
    config = AgentConfig(type="cliver", role="Research assistant")
    agent = CliverAgent(name="researcher", config=config, agent_core=core)

    await agent._do_run("test")

    call_kwargs = core.process_user_input.call_args
    appender = call_kwargs.kwargs.get("system_message_appender")
    assert appender is not None
    assert "Research assistant" in appender()


@pytest.mark.asyncio
async def test_cliver_agent_without_role():
    from cliver.agents.cliver_agent import CliverAgent

    core = _make_mock_core("OK")
    config = AgentConfig(type="cliver")
    agent = CliverAgent(name="default", config=config, agent_core=core)

    await agent._do_run("test")

    call_kwargs = core.process_user_input.call_args
    appender = call_kwargs.kwargs.get("system_message_appender")
    assert appender is None


@pytest.mark.asyncio
async def test_cliver_agent_stream():
    from cliver.agents.cliver_agent import CliverAgent

    core = MagicMock()
    chunk1 = MagicMock()
    chunk1.content = "Hello "
    chunk2 = MagicMock()
    chunk2.content = "world"

    async def mock_stream(**kwargs):
        yield chunk1
        yield chunk2

    core.stream_user_input = mock_stream
    config = AgentConfig(type="cliver", model="test/model")
    agent = CliverAgent(name="default", config=config, agent_core=core)

    chunks = []
    async for c in agent.stream("test"):
        chunks.append(c)

    assert len(chunks) == 3  # 2 text + 1 done
    assert chunks[0].text == "Hello "
    assert chunks[1].text == "world"
    assert chunks[2].chunk_type == "done"
    assert chunks[2].final_result.text == "Hello world"
    assert chunks[2].final_result.status == "completed"
