"""Tests for workflow YAML persistence and execution tracking."""

from datetime import datetime, timedelta, timezone

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

    def test_load_from_sub_workflows(self, store, sample_workflow):
        """load_workflow finds workflows in sub-workflows/ directory."""
        sub_dir = store.workflows_dir / "sub-workflows"
        sub_dir.mkdir(parents=True)

        import yaml

        wf_data = {
            "name": "design-phase",
            "steps": [{"id": "s1", "type": "llm", "name": "S1", "prompt": "Hi"}],
        }
        with open(sub_dir / "design-phase.yaml", "w") as f:
            yaml.dump(wf_data, f)

        loaded = store.load_workflow("design-phase")
        assert loaded is not None
        assert loaded.name == "design-phase"

    def test_top_level_takes_priority_over_sub_workflows(self, store):
        """Top-level workflow takes priority over same-named sub-workflow."""
        import yaml

        store._ensure_dir()
        sub_dir = store.workflows_dir / "sub-workflows"
        sub_dir.mkdir(parents=True)

        top_data = {"name": "shared", "description": "top-level", "steps": []}
        sub_data = {"name": "shared", "description": "sub-workflow", "steps": []}

        with open(store.workflows_dir / "shared.yaml", "w") as f:
            yaml.dump(top_data, f)
        with open(sub_dir / "shared.yaml", "w") as f:
            yaml.dump(sub_data, f)

        loaded = store.load_workflow("shared")
        assert loaded is not None
        assert loaded.description == "top-level"

    def test_list_includes_sub_workflows(self, store, sample_workflow):
        """list_workflows includes workflows from sub-workflows/ directory."""
        import yaml

        store.save_workflow(sample_workflow)
        sub_dir = store.workflows_dir / "sub-workflows"
        sub_dir.mkdir(parents=True)

        sub_data = {"name": "design-phase", "steps": []}
        with open(sub_dir / "design-phase.yaml", "w") as f:
            yaml.dump(sub_data, f)

        names = store.list_workflows()
        assert "test-wf" in names
        assert "design-phase" in names


class TestLoadFromFile:
    def test_load_workflow_from_file(self, tmp_path):
        import yaml

        wf_data = {
            "name": "from-file",
            "description": "Loaded from file",
            "steps": [{"id": "s1", "type": "llm", "name": "Step 1", "prompt": "Hello"}],
        }
        path = tmp_path / "custom.yaml"
        with open(path, "w") as f:
            yaml.dump(wf_data, f)

        loaded = WorkflowStore.load_workflow_from_file(path)
        assert loaded is not None
        assert loaded.name == "from-file"
        assert len(loaded.steps) == 1

    def test_load_workflow_from_file_not_found(self, tmp_path):
        result = WorkflowStore.load_workflow_from_file(tmp_path / "nonexistent.yaml")
        assert result is None

    def test_load_workflow_from_file_with_subdirectory(self, tmp_path):
        import yaml

        sub_dir = tmp_path / "sub-workflows"
        sub_dir.mkdir()
        wf_data = {
            "name": "sub-wf",
            "steps": [{"id": "s1", "type": "llm", "name": "S1", "prompt": "Hi"}],
        }
        path = sub_dir / "sub.yaml"
        with open(path, "w") as f:
            yaml.dump(wf_data, f)

        loaded = WorkflowStore.load_workflow_from_file(path)
        assert loaded is not None
        assert loaded.name == "sub-wf"

    def test_load_workflow_from_project_local(self, tmp_path, monkeypatch):
        """Relative path falls back to .cliver/workflows/ in the project."""
        import yaml

        monkeypatch.chdir(tmp_path)
        local_dir = tmp_path / ".cliver" / "workflows"
        local_dir.mkdir(parents=True)
        wf_data = {
            "name": "local-wf",
            "steps": [{"id": "s1", "type": "llm", "name": "S1", "prompt": "Hi"}],
        }
        with open(local_dir / "my-workflow.yaml", "w") as f:
            yaml.dump(wf_data, f)

        loaded = WorkflowStore.load_workflow_from_file("my-workflow.yaml")
        assert loaded is not None
        assert loaded.name == "local-wf"


class TestExecutionTracking:
    def test_record_and_list_execution(self, tmp_path):
        db_path = tmp_path / "checkpoints.db"
        WorkflowStore.record_execution_start(db_path, "wf_abc", "my-wf", "abc", inputs={"x": 1})

        execs = WorkflowStore.list_executions(db_path, "my-wf")
        assert len(execs) == 1
        assert execs[0]["thread_id"] == "wf_abc"
        assert execs[0]["workflow_name"] == "my-wf"
        assert execs[0]["status"] == "running"

    def test_record_execution_end(self, tmp_path):
        db_path = tmp_path / "checkpoints.db"
        WorkflowStore.record_execution_start(db_path, "wf_abc", "my-wf", "abc")
        WorkflowStore.record_execution_end(db_path, "wf_abc", status="completed")

        execs = WorkflowStore.list_executions(db_path, "my-wf")
        assert execs[0]["status"] == "completed"
        assert execs[0]["finished_at"] is not None

    def test_record_execution_failure(self, tmp_path):
        db_path = tmp_path / "checkpoints.db"
        WorkflowStore.record_execution_start(db_path, "wf_abc", "my-wf", "abc")
        WorkflowStore.record_execution_end(db_path, "wf_abc", status="failed", error="boom")

        execs = WorkflowStore.list_executions(db_path, "my-wf")
        assert execs[0]["status"] == "failed"
        assert execs[0]["error"] == "boom"

    def test_list_executions_filtered(self, tmp_path):
        db_path = tmp_path / "checkpoints.db"
        WorkflowStore.record_execution_start(db_path, "wf1_a", "wf1", "a")
        WorkflowStore.record_execution_start(db_path, "wf2_b", "wf2", "b")

        execs = WorkflowStore.list_executions(db_path, "wf1")
        assert len(execs) == 1
        assert execs[0]["workflow_name"] == "wf1"

    def test_list_executions_all(self, tmp_path):
        db_path = tmp_path / "checkpoints.db"
        WorkflowStore.record_execution_start(db_path, "wf1_a", "wf1", "a")
        WorkflowStore.record_execution_start(db_path, "wf2_b", "wf2", "b")

        execs = WorkflowStore.list_executions(db_path)
        assert len(execs) == 2

    def test_list_empty(self, tmp_path):
        db_path = tmp_path / "checkpoints.db"
        assert WorkflowStore.list_executions(db_path, "nope") == []

    def test_list_nonexistent_db(self, tmp_path):
        assert WorkflowStore.list_executions(tmp_path / "nope.db") == []


class TestDeleteExecution:
    def test_delete_execution_record(self, tmp_path):
        db_path = tmp_path / "checkpoints.db"
        WorkflowStore.record_execution_start(db_path, "wf_abc", "my-wf", "abc")

        deleted = WorkflowStore.delete_execution("wf_abc", db_path)
        assert deleted is True

        execs = WorkflowStore.list_executions(db_path, "my-wf")
        assert len(execs) == 0

    def test_delete_nonexistent_execution(self, tmp_path):
        db_path = tmp_path / "checkpoints.db"
        WorkflowStore._ensure_executions_table(db_path)
        assert WorkflowStore.delete_execution("nope", db_path) is False

    def test_delete_nonexistent_db(self, tmp_path):
        assert WorkflowStore.delete_execution("nope", tmp_path / "nope.db") is False


class TestPruneExecutions:
    def test_prune_old_executions(self, tmp_path):
        db_path = tmp_path / "checkpoints.db"
        WorkflowStore._ensure_executions_table(db_path)

        from cliver.db import get_store

        store = get_store(db_path)
        old_ts = (datetime.now(timezone.utc) - timedelta(days=400)).isoformat()
        recent_ts = datetime.now(timezone.utc).isoformat()

        with store.write() as conn:
            conn.execute(
                "INSERT INTO workflow_executions VALUES (?, ?, ?, ?, ?, NULL, NULL, NULL)",
                ("old-wf_a", "old-wf", "a", "completed", old_ts),
            )
            conn.execute(
                "INSERT INTO workflow_executions VALUES (?, ?, ?, ?, ?, NULL, NULL, NULL)",
                ("new-wf_b", "new-wf", "b", "completed", recent_ts),
            )

        pruned = WorkflowStore.prune_executions(db_path, retention_days=300)
        assert pruned == 1

        execs = WorkflowStore.list_executions(db_path)
        assert len(execs) == 1
        assert execs[0]["thread_id"] == "new-wf_b"

    def test_prune_skips_running(self, tmp_path):
        """Running executions should not be pruned even if old."""
        db_path = tmp_path / "checkpoints.db"
        WorkflowStore._ensure_executions_table(db_path)

        from cliver.db import get_store

        store = get_store(db_path)
        old_ts = (datetime.now(timezone.utc) - timedelta(days=400)).isoformat()

        with store.write() as conn:
            conn.execute(
                "INSERT INTO workflow_executions VALUES (?, ?, ?, ?, ?, NULL, NULL, NULL)",
                ("wf_a", "wf", "a", "running", old_ts),
            )

        pruned = WorkflowStore.prune_executions(db_path, retention_days=300)
        assert pruned == 0

    def test_prune_nothing_when_all_recent(self, tmp_path):
        db_path = tmp_path / "checkpoints.db"
        WorkflowStore.record_execution_start(db_path, "wf_a", "wf", "a")
        WorkflowStore.record_execution_end(db_path, "wf_a", status="completed")

        pruned = WorkflowStore.prune_executions(db_path, retention_days=300)
        assert pruned == 0

    def test_prune_nonexistent_db(self, tmp_path):
        assert WorkflowStore.prune_executions(tmp_path / "nope.db") == 0


class TestDeleteWorkflowWithExecutions:
    def test_delete_removes_execution_records(self, store, sample_workflow, tmp_path):
        db_path = tmp_path / "checkpoints.db"
        store.save_workflow(sample_workflow)

        WorkflowStore.record_execution_start(db_path, "test-wf_a", "test-wf", "a")
        WorkflowStore.record_execution_start(db_path, "test-wf_b", "test-wf", "b")

        result = store.delete_workflow("test-wf", db_path=db_path)
        assert result is True

        execs = WorkflowStore.list_executions(db_path, "test-wf")
        assert len(execs) == 0
