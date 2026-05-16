"""Tests for LabRuntime and RuntimeManager."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from cliver.lab.models import Cell, Lab


def _make_lab():
    return Lab(
        id="lab_test",
        title="Test",
        default_agent="cliver",
        context={"working_dir": "."},
        cells=[
            Cell(
                id="setup",
                type="config",
                title="Setup",
                inputs={"schema": {"domain": {"type": "text"}}},
                outputs={"domain": "AI"},
                status="completed",
            ),
            Cell(id="search", type="llm", title="Search", inputs={"prompt": "Find ${setup.outputs.domain} papers"}),
            Cell(
                id="calc",
                type="code",
                title="Calc",
                inputs={
                    "source": "def run(ctx):\n    d = ctx.refs('setup.outputs.domain')\n    return {'msg': f'Domain is {d}'}"
                },
            ),
        ],
    )


def _make_agent_factory():
    factory = MagicMock()
    mock_agent = AsyncMock()
    from cliver.agent import AgentResult

    mock_agent.run = AsyncMock(return_value=AgentResult(text="Agent result", status="completed"))
    mock_agent.initialize = AsyncMock()
    factory.create = MagicMock(return_value=mock_agent)
    return factory


def test_load_from_lab():
    from cliver.lab.runtime import LabRuntime

    lab = _make_lab()
    rt = LabRuntime(lab, _make_agent_factory())
    rt.load_from_lab()
    assert "setup" in rt.variables
    assert rt.variables["setup"]["outputs"]["domain"] == "AI"
    assert "search" not in rt.variables


def test_runtime_context_refs():
    from cliver.lab.runtime import RuntimeContext

    variables = {"setup": {"outputs": {"domain": "AI", "count": 10}}}
    ctx = RuntimeContext(variables)
    assert ctx.refs("setup.outputs.domain") == "AI"
    assert ctx.refs("setup.outputs.count") == 10


def test_runtime_context_log():
    from cliver.lab.runtime import RuntimeContext

    ctx = RuntimeContext({})
    ctx.log("test message")
    assert "test message" in ctx._logs


@pytest.mark.asyncio
async def test_execute_cell_config():
    from cliver.lab.runtime import LabRuntime

    lab = _make_lab()
    rt = LabRuntime(lab, _make_agent_factory())

    result = await rt.execute_cell("setup")
    assert result["domain"] == "AI"
    assert "setup" in rt.variables
    assert lab.get_cell("setup").status == "completed"


@pytest.mark.asyncio
async def test_execute_cell_code():
    from cliver.lab.runtime import LabRuntime

    lab = _make_lab()
    rt = LabRuntime(lab, _make_agent_factory())
    rt.load_from_lab()

    result = await rt.execute_cell("calc")
    assert result["msg"] == "Domain is AI"
    assert lab.get_cell("calc").status == "completed"


@pytest.mark.asyncio
async def test_execute_all():
    from cliver.lab.runtime import LabRuntime

    lab = _make_lab()
    rt = LabRuntime(lab, _make_agent_factory())

    await rt.execute_all()
    assert lab.get_cell("setup").status == "completed"
    assert lab.get_cell("search").status == "completed"
    assert lab.get_cell("calc").status == "completed"


def test_get_available_refs():
    from cliver.lab.runtime import LabRuntime

    lab = _make_lab()
    rt = LabRuntime(lab, _make_agent_factory())
    rt.load_from_lab()

    refs = rt.get_available_refs("search")
    assert len(refs) == 1
    assert refs[0]["cell_id"] == "setup"
    assert any(f["path"] == "setup.outputs.domain" for f in refs[0]["fields"])


def test_runtime_manager_get_or_create():
    from cliver.lab.runtime import RuntimeManager

    mgr = RuntimeManager()
    lab = _make_lab()
    factory = _make_agent_factory()

    rt1 = mgr.get_or_create("lab_test", lab, factory)
    rt2 = mgr.get_or_create("lab_test", lab, factory)
    assert rt1 is rt2


@pytest.mark.asyncio
async def test_runtime_manager_cleanup():
    from cliver.lab.runtime import RuntimeManager

    mgr = RuntimeManager(timeout_s=0)
    lab = _make_lab()
    factory = _make_agent_factory()
    mgr.get_or_create("lab_test", lab, factory)
    assert len(mgr._runtimes) == 1

    await asyncio.sleep(0.1)
    await mgr.cleanup_idle()
    assert len(mgr._runtimes) == 0
