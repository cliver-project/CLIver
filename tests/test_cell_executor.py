"""Tests for CellExecutor — type-specific cell execution."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from cliver.lab.models import Cell


def _make_runtime(variables=None):
    """Create a mock runtime for testing."""
    runtime = MagicMock()
    runtime.variables = variables or {}
    runtime.lab = MagicMock()
    runtime.lab.default_agent = "cliver"
    runtime.lab.context = {"working_dir": "."}
    runtime.agent_factory = MagicMock()

    from cliver.lab.runtime import RuntimeContext

    runtime.ctx = RuntimeContext(runtime.variables, runtime.agent_factory, runtime.lab.context)
    return runtime


@pytest.mark.asyncio
async def test_execute_config():
    from cliver.lab.executor import CellExecutor

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
    from cliver.lab.executor import CellExecutor

    cell = Cell(id="setup", type="config", title="Setup", inputs={"schema": {"x": {"type": "text"}}}, outputs={})
    executor = CellExecutor()
    result = await executor.execute(cell, _make_runtime())
    assert result == {}


@pytest.mark.asyncio
async def test_execute_code_basic():
    from cliver.lab.executor import CellExecutor

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
    from cliver.lab.executor import CellExecutor

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
    from cliver.lab.executor import CellExecutor

    cell = Cell(id="bad", type="code", title="Bad", inputs={"source": "x = 1"})
    executor = CellExecutor()
    with pytest.raises(ValueError, match="must define.*run"):
        await executor.execute(cell, _make_runtime())


@pytest.mark.asyncio
async def test_execute_code_non_dict_return():
    from cliver.lab.executor import CellExecutor

    cell = Cell(id="bad", type="code", title="Bad", inputs={"source": "def run(ctx):\n    return [1, 2, 3]"})
    executor = CellExecutor()
    with pytest.raises(TypeError, match="must return dict"):
        await executor.execute(cell, _make_runtime())


@pytest.mark.asyncio
async def test_execute_code_non_serializable():
    from cliver.lab.executor import CellExecutor

    cell = Cell(id="bad", type="code", title="Bad", inputs={"source": "def run(ctx):\n    return {'obj': object()}"})
    executor = CellExecutor()
    with pytest.raises(TypeError, match="non-JSON-serializable"):
        await executor.execute(cell, _make_runtime())


@pytest.mark.asyncio
async def test_execute_llm():
    from cliver.agent import AgentResult
    from cliver.lab.executor import CellExecutor

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
    from cliver.lab.executor import CellExecutor

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
    from cliver.lab.executor import CellExecutor

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
    from cliver.lab.executor import CellExecutor

    cell = Cell(id="x", type="unknown", title="X")
    executor = CellExecutor()
    with pytest.raises(ValueError, match="Unknown cell type"):
        await executor.execute(cell, _make_runtime())


# --- LLM Cell Verification ---


@pytest.mark.asyncio
async def test_execute_llm_no_verification():
    """LLM cell without verification works as before."""
    from cliver.agent import AgentResult
    from cliver.lab.executor import CellExecutor

    mock_agent = AsyncMock()
    mock_agent.run = AsyncMock(return_value=AgentResult(text="Hello", status="completed"))
    mock_agent.initialize = AsyncMock()

    runtime = _make_runtime()
    runtime.agent_factory.create = MagicMock(return_value=mock_agent)

    cell = Cell(id="test", type="llm", title="Test", inputs={"prompt": "say hello"})
    executor = CellExecutor()
    result = await executor.execute(cell, runtime)
    assert result["text"] == "Hello"
    assert "_verification" not in result


@pytest.mark.asyncio
async def test_execute_llm_verification_pass_first_attempt():
    from cliver.agent import AgentResult
    from cliver.lab.executor import CellExecutor

    mock_agent = AsyncMock()
    mock_agent.run = AsyncMock(return_value=AgentResult(text="5 papers found", status="completed"))
    mock_agent.initialize = AsyncMock()

    mock_verifier = AsyncMock()
    mock_verifier.run = AsyncMock(
        return_value=AgentResult(text='{"pass": true, "reason": "Contains 5 papers"}', status="completed")
    )
    mock_verifier.initialize = AsyncMock()

    runtime = _make_runtime()
    call_count = [0]

    def create_agent(name=None):
        call_count[0] += 1
        if call_count[0] <= 1:
            return mock_agent
        return mock_verifier

    runtime.agent_factory.create = create_agent

    cell = Cell(
        id="test",
        type="llm",
        title="Test",
        inputs={"prompt": "find papers", "verification": {"expected": "At least 5 papers", "max_retries": 3}},
    )
    executor = CellExecutor()
    result = await executor.execute(cell, runtime)
    assert result["text"] == "5 papers found"
    assert result["_verification"]["passed"] is True
    assert result["_verification"]["attempt"] == 1


@pytest.mark.asyncio
async def test_execute_llm_verification_pass_on_retry():
    from cliver.agent import AgentResult
    from cliver.lab.executor import CellExecutor

    call_count = [0]

    async def mock_run(prompt, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            return AgentResult(text="only 2 papers", status="completed")
        return AgentResult(text="found 5 papers with details", status="completed")

    mock_agent = AsyncMock()
    mock_agent.run = mock_run
    mock_agent.initialize = AsyncMock()

    verify_count = [0]

    async def mock_verify(prompt, **kwargs):
        verify_count[0] += 1
        if verify_count[0] == 1:
            return AgentResult(text='{"pass": false, "reason": "Only 2 papers, need 5"}', status="completed")
        return AgentResult(text='{"pass": true, "reason": "Has 5 papers"}', status="completed")

    mock_verifier = AsyncMock()
    mock_verifier.run = mock_verify
    mock_verifier.initialize = AsyncMock()

    runtime = _make_runtime()
    agent_idx = [0]

    def create_agent(name=None):
        agent_idx[0] += 1
        if agent_idx[0] <= 1:
            return mock_agent
        return mock_verifier

    runtime.agent_factory.create = create_agent

    cell = Cell(
        id="test",
        type="llm",
        title="Test",
        inputs={"prompt": "find papers", "verification": {"expected": "At least 5 papers", "max_retries": 3}},
    )
    executor = CellExecutor()
    result = await executor.execute(cell, runtime)
    assert result["_verification"]["passed"] is True
    assert result["_verification"]["attempt"] == 2


@pytest.mark.asyncio
async def test_execute_llm_verification_fail_all_retries():
    from cliver.agent import AgentResult
    from cliver.lab.executor import CellExecutor, VerificationError

    mock_agent = AsyncMock()
    mock_agent.run = AsyncMock(return_value=AgentResult(text="bad output", status="completed"))
    mock_agent.initialize = AsyncMock()

    mock_verifier = AsyncMock()
    mock_verifier.run = AsyncMock(
        return_value=AgentResult(text='{"pass": false, "reason": "Wrong format"}', status="completed")
    )
    mock_verifier.initialize = AsyncMock()

    runtime = _make_runtime()
    idx = [0]

    def create_agent(name=None):
        idx[0] += 1
        if idx[0] <= 1:
            return mock_agent
        return mock_verifier

    runtime.agent_factory.create = create_agent

    cell = Cell(
        id="test",
        type="llm",
        title="Test",
        inputs={"prompt": "do something", "verification": {"expected": "Good output", "max_retries": 2}},
    )
    executor = CellExecutor()
    with pytest.raises(VerificationError, match="failed after 2 attempts"):
        await executor.execute(cell, runtime)


@pytest.mark.asyncio
async def test_execute_llm_verification_parse_verdict():
    from cliver.lab.executor import CellExecutor

    executor = CellExecutor()

    # Valid JSON
    v = executor._parse_verdict('{"pass": true, "reason": "good"}')
    assert v["pass"] is True

    # JSON in code block
    v = executor._parse_verdict('```json\n{"pass": false, "reason": "bad"}\n```')
    assert v["pass"] is False

    # Unparseable — keyword fallback
    v = executor._parse_verdict("The output looks correct, pass is true")
    assert v["pass"] is True

    # Unparseable — defaults to fail
    v = executor._parse_verdict("Something went wrong")
    assert v["pass"] is False
