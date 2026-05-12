"""Tests for NotebookRuntime and RuntimeManager."""

import asyncio
import time

import pytest
from unittest.mock import AsyncMock, MagicMock

from cliver.notebook.models import Notebook, Cell


def _make_notebook():
    return Notebook(
        id="nb_test", title="Test",
        default_agent="cliver",
        context={"working_dir": "."},
        cells=[
            Cell(id="setup", type="config", title="Setup",
                 inputs={"schema": {"domain": {"type": "text"}}},
                 outputs={"domain": "AI"}, status="completed"),
            Cell(id="search", type="llm", title="Search",
                 inputs={"prompt": "Find ${setup.outputs.domain} papers"}),
            Cell(id="calc", type="code", title="Calc",
                 inputs={"source": "def run(ctx):\n    d = ctx.refs('setup.outputs.domain')\n    return {'msg': f'Domain is {d}'}"}),
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


def test_load_from_notebook():
    from cliver.notebook.runtime import NotebookRuntime

    nb = _make_notebook()
    rt = NotebookRuntime(nb, _make_agent_factory())
    rt.load_from_notebook()
    assert "setup" in rt.variables
    assert rt.variables["setup"]["outputs"]["domain"] == "AI"
    assert "search" not in rt.variables


def test_runtime_context_refs():
    from cliver.notebook.runtime import RuntimeContext

    variables = {"setup": {"outputs": {"domain": "AI", "count": 10}}}
    ctx = RuntimeContext(variables)
    assert ctx.refs("setup.outputs.domain") == "AI"
    assert ctx.refs("setup.outputs.count") == 10


def test_runtime_context_log():
    from cliver.notebook.runtime import RuntimeContext

    ctx = RuntimeContext({})
    ctx.log("test message")
    assert "test message" in ctx._logs


@pytest.mark.asyncio
async def test_execute_cell_config():
    from cliver.notebook.runtime import NotebookRuntime

    nb = _make_notebook()
    rt = NotebookRuntime(nb, _make_agent_factory())

    result = await rt.execute_cell("setup")
    assert result["domain"] == "AI"
    assert "setup" in rt.variables
    assert nb.get_cell("setup").status == "completed"


@pytest.mark.asyncio
async def test_execute_cell_code():
    from cliver.notebook.runtime import NotebookRuntime

    nb = _make_notebook()
    rt = NotebookRuntime(nb, _make_agent_factory())
    rt.load_from_notebook()

    result = await rt.execute_cell("calc")
    assert result["msg"] == "Domain is AI"
    assert nb.get_cell("calc").status == "completed"


@pytest.mark.asyncio
async def test_execute_all():
    from cliver.notebook.runtime import NotebookRuntime

    nb = _make_notebook()
    rt = NotebookRuntime(nb, _make_agent_factory())

    await rt.execute_all()
    assert nb.get_cell("setup").status == "completed"
    assert nb.get_cell("search").status == "completed"
    assert nb.get_cell("calc").status == "completed"


def test_get_available_refs():
    from cliver.notebook.runtime import NotebookRuntime

    nb = _make_notebook()
    rt = NotebookRuntime(nb, _make_agent_factory())
    rt.load_from_notebook()

    refs = rt.get_available_refs("search")
    assert len(refs) == 1
    assert refs[0]["cell_id"] == "setup"
    assert any(f["path"] == "setup.outputs.domain" for f in refs[0]["fields"])


def test_runtime_manager_get_or_create():
    from cliver.notebook.runtime import RuntimeManager

    mgr = RuntimeManager()
    nb = _make_notebook()
    factory = _make_agent_factory()

    rt1 = mgr.get_or_create("nb_test", nb, factory)
    rt2 = mgr.get_or_create("nb_test", nb, factory)
    assert rt1 is rt2


@pytest.mark.asyncio
async def test_runtime_manager_cleanup():
    from cliver.notebook.runtime import RuntimeManager

    mgr = RuntimeManager(timeout_s=0)
    nb = _make_notebook()
    factory = _make_agent_factory()
    mgr.get_or_create("nb_test", nb, factory)
    assert len(mgr._runtimes) == 1

    await asyncio.sleep(0.1)
    await mgr.cleanup_idle()
    assert len(mgr._runtimes) == 0
