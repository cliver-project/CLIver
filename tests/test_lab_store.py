"""Tests for LabStore CRUD."""

import tempfile
from pathlib import Path

import pytest

from cliver.lab.models import Cell


@pytest.fixture
def store():
    from cliver.lab.store import LabStore

    with tempfile.TemporaryDirectory() as tmpdir:
        yield LabStore(Path(tmpdir))


def test_create_lab(store):
    lab = store.create(title="Test Lab")
    assert lab.id.startswith("lab_")
    assert lab.title == "Test Lab"
    assert lab.cells == []


def test_create_with_cells(store):
    cells = [
        {"id": "setup", "type": "config", "title": "Setup"},
        {"id": "run", "type": "llm", "title": "Run"},
    ]
    lab = store.create(title="With Cells", cells=cells)
    assert len(lab.cells) == 2
    assert lab.cells[0].id == "setup"


def test_create_with_metadata(store):
    lab = store.create(
        title="Research",
        description="Paper survey",
        scenario_id="research-ai-lab",
        default_agent="researcher",
        context={"working_dir": "./proj"},
    )
    assert lab.description == "Paper survey"
    assert lab.scenario_id == "research-ai-lab"
    assert lab.default_agent == "researcher"
    assert lab.context["working_dir"] == "./proj"


def test_get_lab(store):
    created = store.create(title="Test")
    fetched = store.get(created.id)
    assert fetched is not None
    assert fetched.id == created.id
    assert fetched.title == "Test"


def test_get_nonexistent(store):
    assert store.get("lab_nonexistent") is None


def test_list_all_empty(store):
    assert store.list_all() == []


def test_list_all(store):
    store.create(title="First")
    store.create(title="Second")
    summaries = store.list_all()
    assert len(summaries) == 2
    titles = {s.title for s in summaries}
    assert titles == {"First", "Second"}


def test_update_lab(store):
    lab = store.create(title="Original")
    lab.title = "Updated"
    lab.cells.append(Cell(id="new", type="display", title="New Cell"))
    store.update(lab)

    fetched = store.get(lab.id)
    assert fetched.title == "Updated"
    assert len(fetched.cells) == 1


def test_delete_lab(store):
    lab = store.create(title="To Delete")
    assert store.delete(lab.id) is True
    assert store.get(lab.id) is None
    assert store.delete(lab.id) is False


def test_save_cell_output(store):
    lab = store.create(
        title="Test",
        cells=[{"id": "c1", "type": "config", "title": "Config"}],
    )
    store.save_cell_output(lab.id, "c1", {"key": "value"}, "completed", duration_ms=100)

    fetched = store.get(lab.id)
    cell = fetched.get_cell("c1")
    assert cell.outputs == {"key": "value"}
    assert cell.status == "completed"
    assert cell.duration_ms == 100


def test_save_cell_output_error(store):
    lab = store.create(
        title="Test",
        cells=[{"id": "c1", "type": "code", "title": "Code"}],
    )
    store.save_cell_output(lab.id, "c1", {}, "error", error="Something broke")

    fetched = store.get(lab.id)
    cell = fetched.get_cell("c1")
    assert cell.status == "error"
    assert cell.error == "Something broke"


def test_json_file_exists(store):
    lab = store.create(title="Test")
    json_path = store._labs_dir / f"{lab.id}.json"
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
