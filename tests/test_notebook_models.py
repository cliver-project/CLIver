"""Tests for Notebook and Cell models."""

import pytest


def test_cell_defaults():
    from cliver.notebook.models import Cell

    c = Cell(id="setup", type="config", title="Setup")
    assert c.id == "setup"
    assert c.type == "config"
    assert c.inputs == {}
    assert c.outputs == {}
    assert c.status == "idle"
    assert c.error is None
    assert c.duration_ms == 0


def test_cell_with_outputs():
    from cliver.notebook.models import Cell

    c = Cell(
        id="search", type="llm", title="Search",
        inputs={"prompt": "find papers", "agent": "cliver"},
        outputs={"text": "Found 5 papers", "data": [{"title": "Paper A"}]},
        status="completed", duration_ms=1500,
    )
    assert c.outputs["text"] == "Found 5 papers"
    assert len(c.outputs["data"]) == 1
    assert c.status == "completed"


def test_cell_type_validation():
    from cliver.notebook.models import Cell

    c = Cell(id="x", type="code", title="Code")
    assert c.type == "code"
    c2 = Cell(id="y", type="display", title="Note")
    assert c2.type == "display"


def test_notebook_creation():
    from cliver.notebook.models import Notebook, Cell

    nb = Notebook(
        id="nb_abc123",
        title="Test Notebook",
        cells=[
            Cell(id="setup", type="config", title="Setup"),
            Cell(id="run", type="llm", title="Run"),
        ],
    )
    assert nb.id == "nb_abc123"
    assert nb.title == "Test Notebook"
    assert len(nb.cells) == 2
    assert nb.schema_version == "cliver-notebook-v1"
    assert nb.default_agent is None
    assert nb.context == {}


def test_notebook_with_metadata():
    from cliver.notebook.models import Notebook

    nb = Notebook(
        id="nb_xyz789",
        title="Research",
        description="Paper survey",
        scenario_id="research-ai-lab",
        default_agent="researcher",
        context={"working_dir": "./projects", "description": "Q2 research"},
    )
    assert nb.scenario_id == "research-ai-lab"
    assert nb.default_agent == "researcher"
    assert nb.context["working_dir"] == "./projects"


def test_notebook_serialization():
    from cliver.notebook.models import Notebook, Cell

    nb = Notebook(
        id="nb_test",
        title="Test",
        cells=[Cell(id="c1", type="config", title="Config", outputs={"key": "val"})],
    )
    data = nb.model_dump()
    assert data["$schema"] == "cliver-notebook-v1"
    assert data["id"] == "nb_test"
    assert len(data["cells"]) == 1
    assert data["cells"][0]["outputs"] == {"key": "val"}


def test_notebook_from_json():
    from cliver.notebook.models import Notebook

    raw = {
        "$schema": "cliver-notebook-v1",
        "id": "nb_test",
        "title": "From JSON",
        "cells": [
            {"id": "c1", "type": "llm", "title": "LLM Cell",
             "inputs": {"prompt": "hello"}, "outputs": {"text": "hi"},
             "status": "completed"},
        ],
    }
    nb = Notebook.model_validate(raw)
    assert nb.id == "nb_test"
    assert nb.cells[0].outputs["text"] == "hi"


def test_notebook_summary():
    from cliver.notebook.models import NotebookSummary

    s = NotebookSummary(
        id="nb_abc", title="Test", cell_count=3,
        status="completed", created_at="2026-01-01", updated_at="2026-01-02",
    )
    assert s.id == "nb_abc"
    assert s.cell_count == 3


def test_notebook_get_cell():
    from cliver.notebook.models import Notebook, Cell

    nb = Notebook(
        id="nb_test", title="Test",
        cells=[
            Cell(id="a", type="config", title="A"),
            Cell(id="b", type="llm", title="B"),
        ],
    )
    assert nb.get_cell("a").title == "A"
    assert nb.get_cell("b").title == "B"
    assert nb.get_cell("nonexistent") is None


def test_notebook_cells_before():
    from cliver.notebook.models import Notebook, Cell

    nb = Notebook(
        id="nb_test", title="Test",
        cells=[
            Cell(id="a", type="config", title="A"),
            Cell(id="b", type="llm", title="B"),
            Cell(id="c", type="code", title="C"),
        ],
    )
    before_c = nb.cells_before("c")
    assert [c.id for c in before_c] == ["a", "b"]
    before_a = nb.cells_before("a")
    assert before_a == []
