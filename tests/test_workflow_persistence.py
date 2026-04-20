"""Tests for workflow YAML persistence."""

import pytest

from cliver.workflow.persistence import WorkflowStore
from cliver.workflow.workflow_models import (
    LLMStep,
    Workflow,
)


@pytest.fixture
def store(tmp_path):
    return WorkflowStore(tmp_path / "workflows")


@pytest.fixture
def sample_workflow():
    return Workflow(
        name="test-wf",
        description="A test workflow",
        inputs={"branch": "main"},
        steps=[
            LLMStep(id="s1", name="Step 1", prompt="Hello {{ inputs.branch }}"),
        ],
    )


class TestWorkflowCRUD:
    def test_save_and_load(self, store, sample_workflow):
        store.save_workflow(sample_workflow)
        loaded = store.load_workflow("test-wf")
        assert loaded is not None
        assert loaded.name == "test-wf"
        assert len(loaded.steps) == 1

    def test_load_nonexistent(self, store):
        assert store.load_workflow("nope") is None

    def test_list_workflows(self, store, sample_workflow):
        store.save_workflow(sample_workflow)
        wf2 = Workflow(name="other", steps=[])
        store.save_workflow(wf2)
        names = store.list_workflows()
        assert "test-wf" in names
        assert "other" in names

    def test_list_empty(self, store):
        assert store.list_workflows() == []

    def test_delete_workflow(self, store, sample_workflow):
        store.save_workflow(sample_workflow)
        assert store.delete_workflow("test-wf") is True
        assert store.load_workflow("test-wf") is None

    def test_delete_nonexistent(self, store):
        assert store.delete_workflow("nope") is False
