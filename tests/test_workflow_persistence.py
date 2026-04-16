"""Tests for workflow YAML persistence."""

import pytest

from cliver.workflow.persistence import WorkflowStore
from cliver.workflow.workflow_models import (
    ExecutionContext,
    LLMStep,
    Workflow,
    WorkflowExecutionState,
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


class TestExecutionState:
    def test_save_and_load_state(self, store):
        state = WorkflowExecutionState(
            workflow_name="test-wf",
            execution_id="exec1",
            status="running",
            context=ExecutionContext(workflow_name="test-wf", inputs={"x": 1}),
        )
        store.save_state(state)
        loaded = store.load_state("test-wf")
        assert loaded is not None
        assert loaded.execution_id == "exec1"
        assert loaded.context.inputs["x"] == 1

    def test_load_state_nonexistent(self, store):
        assert store.load_state("nope") is None

    def test_state_overwritten_on_save(self, store):
        state1 = WorkflowExecutionState(
            workflow_name="wf",
            execution_id="e1",
            status="running",
            context=ExecutionContext(workflow_name="wf"),
        )
        store.save_state(state1)
        state2 = WorkflowExecutionState(
            workflow_name="wf",
            execution_id="e1",
            status="completed",
            context=ExecutionContext(workflow_name="wf"),
        )
        store.save_state(state2)
        loaded = store.load_state("wf")
        assert loaded.status == "completed"
