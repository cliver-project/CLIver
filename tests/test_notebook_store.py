"""Tests for NotebookStore CRUD."""

import tempfile
from pathlib import Path

import pytest

from cliver.notebook.models import Cell


@pytest.fixture
def store():
    from cliver.notebook.store import NotebookStore

    with tempfile.TemporaryDirectory() as tmpdir:
        yield NotebookStore(Path(tmpdir))


def test_create_notebook(store):
    nb = store.create(title="Test Notebook")
    assert nb.id.startswith("nb_")
    assert nb.title == "Test Notebook"
    assert nb.cells == []


def test_create_with_cells(store):
    cells = [
        {"id": "setup", "type": "config", "title": "Setup"},
        {"id": "run", "type": "llm", "title": "Run"},
    ]
    nb = store.create(title="With Cells", cells=cells)
    assert len(nb.cells) == 2
    assert nb.cells[0].id == "setup"


def test_create_with_metadata(store):
    nb = store.create(
        title="Research",
        description="Paper survey",
        scenario_id="research-ai-lab",
        default_agent="researcher",
        context={"working_dir": "./proj"},
    )
    assert nb.description == "Paper survey"
    assert nb.scenario_id == "research-ai-lab"
    assert nb.default_agent == "researcher"
    assert nb.context["working_dir"] == "./proj"


def test_get_notebook(store):
    created = store.create(title="Test")
    fetched = store.get(created.id)
    assert fetched is not None
    assert fetched.id == created.id
    assert fetched.title == "Test"


def test_get_nonexistent(store):
    assert store.get("nb_nonexistent") is None


def test_list_all_empty(store):
    assert store.list_all() == []


def test_list_all(store):
    store.create(title="First")
    store.create(title="Second")
    summaries = store.list_all()
    assert len(summaries) == 2
    titles = {s.title for s in summaries}
    assert titles == {"First", "Second"}


def test_update_notebook(store):
    nb = store.create(title="Original")
    nb.title = "Updated"
    nb.cells.append(Cell(id="new", type="display", title="New Cell"))
    store.update(nb)

    fetched = store.get(nb.id)
    assert fetched.title == "Updated"
    assert len(fetched.cells) == 1


def test_delete_notebook(store):
    nb = store.create(title="To Delete")
    assert store.delete(nb.id) is True
    assert store.get(nb.id) is None
    assert store.delete(nb.id) is False


def test_save_cell_output(store):
    nb = store.create(
        title="Test",
        cells=[{"id": "c1", "type": "config", "title": "Config"}],
    )
    store.save_cell_output(nb.id, "c1", {"key": "value"}, "completed", duration_ms=100)

    fetched = store.get(nb.id)
    cell = fetched.get_cell("c1")
    assert cell.outputs == {"key": "value"}
    assert cell.status == "completed"
    assert cell.duration_ms == 100


def test_save_cell_output_error(store):
    nb = store.create(
        title="Test",
        cells=[{"id": "c1", "type": "code", "title": "Code"}],
    )
    store.save_cell_output(nb.id, "c1", {}, "error", error="Something broke")

    fetched = store.get(nb.id)
    cell = fetched.get_cell("c1")
    assert cell.status == "error"
    assert cell.error == "Something broke"


def test_json_file_exists(store):
    nb = store.create(title="Test")
    json_path = store._notebooks_dir / f"{nb.id}.json"
    assert json_path.exists()


def test_list_reflects_updates(store):
    store.create(
        title="Test",
        cells=[
            {"id": "c1", "type": "config", "title": "Config"},
        ],
    )
    summaries = store.list_all()
    assert summaries[0].cell_count == 1
