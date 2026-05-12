"""Tests for CellExecutor — type-specific cell execution."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from cliver.notebook.models import Cell


def _make_runtime(variables=None):
    """Create a mock runtime for testing."""
    runtime = MagicMock()
    runtime.variables = variables or {}
    runtime.notebook = MagicMock()
    runtime.notebook.default_agent = "cliver"
    runtime.notebook.context = {"working_dir": "."}
    runtime.agent_factory = MagicMock()

    from cliver.notebook.runtime import RuntimeContext

    runtime.ctx = RuntimeContext(runtime.variables, runtime.agent_factory, runtime.notebook.context)
    return runtime


@pytest.mark.asyncio
async def test_execute_config():
    from cliver.notebook.executor import CellExecutor

    cell = Cell(
        id="setup",
        type="config",
        title="Setup",
        inputs={"schema": {"domain": {"type": "text", "required": True}}},
        outputs={"domain": "AI research"},
    )
    executor = CellExecutor()
    result = await executor.execute(cell, _make_runtime())
    assert result == {"domain": "AI research"}


@pytest.mark.asyncio
async def test_execute_config_empty_outputs():
    from cliver.notebook.executor import CellExecutor

    cell = Cell(id="setup", type="config", title="Setup", inputs={"schema": {"x": {"type": "text"}}}, outputs={})
    executor = CellExecutor()
    result = await executor.execute(cell, _make_runtime())
    assert result == {}


@pytest.mark.asyncio
async def test_execute_code_basic():
    from cliver.notebook.executor import CellExecutor

    cell = Cell(
        id="calc",
        type="code",
        title="Calculate",
        inputs={"source": "def run(ctx):\n    return {'result': 42}"},
    )
    executor = CellExecutor()
    result = await executor.execute(cell, _make_runtime())
    assert result == {"result": 42}


@pytest.mark.asyncio
async def test_execute_code_with_refs():
    from cliver.notebook.executor import CellExecutor

    variables = {"setup": {"outputs": {"items": [1, 2, 3]}}}
    cell = Cell(
        id="process",
        type="code",
        title="Process",
        inputs={
            "source": "def run(ctx):\n    items = ctx.refs('setup.outputs.items')\n    return {'total': sum(items)}"
        },
    )
    executor = CellExecutor()
    result = await executor.execute(cell, _make_runtime(variables))
    assert result == {"total": 6}


@pytest.mark.asyncio
async def test_execute_code_no_run_function():
    from cliver.notebook.executor import CellExecutor

    cell = Cell(id="bad", type="code", title="Bad", inputs={"source": "x = 1"})
    executor = CellExecutor()
    with pytest.raises(ValueError, match="must define.*run"):
        await executor.execute(cell, _make_runtime())


@pytest.mark.asyncio
async def test_execute_code_non_dict_return():
    from cliver.notebook.executor import CellExecutor

    cell = Cell(id="bad", type="code", title="Bad", inputs={"source": "def run(ctx):\n    return [1, 2, 3]"})
    executor = CellExecutor()
    with pytest.raises(TypeError, match="must return dict"):
        await executor.execute(cell, _make_runtime())


@pytest.mark.asyncio
async def test_execute_code_non_serializable():
    from cliver.notebook.executor import CellExecutor

    cell = Cell(id="bad", type="code", title="Bad", inputs={"source": "def run(ctx):\n    return {'obj': object()}"})
    executor = CellExecutor()
    with pytest.raises(TypeError, match="non-JSON-serializable"):
        await executor.execute(cell, _make_runtime())


@pytest.mark.asyncio
async def test_execute_llm():
    from cliver.agent import AgentResult
    from cliver.notebook.executor import CellExecutor

    mock_agent = AsyncMock()
    mock_agent.run = AsyncMock(
        return_value=AgentResult(
            text="Found 5 papers",
            status="completed",
            artifacts=[],
        )
    )
    mock_agent.initialize = AsyncMock()

    runtime = _make_runtime({"setup": {"outputs": {"domain": "AI"}}})
    runtime.agent_factory.create = MagicMock(return_value=mock_agent)

    cell = Cell(
        id="search",
        type="llm",
        title="Search",
        inputs={"prompt": "Find papers about ${setup.outputs.domain}", "agent": "cliver"},
    )
    executor = CellExecutor()
    result = await executor.execute(cell, runtime)
    assert result["text"] == "Found 5 papers"
    mock_agent.run.assert_called_once()
    call_args = mock_agent.run.call_args
    assert "AI" in call_args[0][0]


@pytest.mark.asyncio
async def test_execute_llm_json_output():
    from cliver.agent import AgentResult
    from cliver.notebook.executor import CellExecutor

    mock_agent = AsyncMock()
    mock_agent.run = AsyncMock(
        return_value=AgentResult(
            text='[{"title": "Paper A"}]',
            status="completed",
        )
    )
    mock_agent.initialize = AsyncMock()

    runtime = _make_runtime()
    runtime.agent_factory.create = MagicMock(return_value=mock_agent)

    cell = Cell(
        id="search",
        type="llm",
        title="Search",
        inputs={"prompt": "find papers", "output_format": "json"},
    )
    executor = CellExecutor()
    result = await executor.execute(cell, runtime)
    assert result["text"] == '[{"title": "Paper A"}]'
    assert result["data"] == [{"title": "Paper A"}]


@pytest.mark.asyncio
async def test_execute_display():
    from cliver.notebook.executor import CellExecutor

    variables = {"calc": {"outputs": {"count": 5}}}
    cell = Cell(
        id="note",
        type="display",
        title="Note",
        inputs={"content": "Found ${calc.outputs.count} items", "format": "markdown"},
    )
    executor = CellExecutor()
    result = await executor.execute(cell, _make_runtime(variables))
    assert result == {}


@pytest.mark.asyncio
async def test_execute_unknown_type():
    from cliver.notebook.executor import CellExecutor

    cell = Cell(id="x", type="unknown", title="X")
    executor = CellExecutor()
    with pytest.raises(ValueError, match="Unknown cell type"):
        await executor.execute(cell, _make_runtime())
