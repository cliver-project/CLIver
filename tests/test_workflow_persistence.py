# tests/test_workflow_persistence.py
from cliver.workflow.persistence import WorkflowStore
from cliver.workflow.workflow_models import Workflow


class TestWorkflowStore:
    def test_save_and_load(self, tmp_path):
        store = WorkflowStore(tmp_path / "workflows")
        wf = Workflow(
            name="test-wf",
            description="A test",
            steps=[{"id": "s1", "type": "llm", "prompt": "Hello"}],
        )
        store.save_workflow(wf)
        loaded = store.load_workflow("test-wf")
        assert loaded is not None
        assert loaded.name == "test-wf"
        assert len(loaded.steps) == 1

    def test_load_nonexistent(self, tmp_path):
        store = WorkflowStore(tmp_path / "workflows")
        assert store.load_workflow("nope") is None

    def test_list_workflows(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        store = WorkflowStore(tmp_path / "workflows")
        wf1 = Workflow(name="a", steps=[{"id": "s1", "type": "llm", "prompt": "Hi"}])
        wf2 = Workflow(name="b", steps=[{"id": "s1", "type": "llm", "prompt": "Hi"}])
        store.save_workflow(wf1)
        store.save_workflow(wf2)
        names = store.list_workflows()
        assert sorted(names) == ["a", "b"]

    def test_delete_workflow(self, tmp_path):
        store = WorkflowStore(tmp_path / "workflows")
        wf = Workflow(name="del-me", steps=[{"id": "s1", "type": "llm", "prompt": "Hi"}])
        store.save_workflow(wf)
        assert store.delete_workflow("del-me") is True
        assert store.load_workflow("del-me") is None

    def test_load_from_file(self, tmp_path):
        import yaml

        path = tmp_path / "custom.yaml"
        data = {"name": "custom", "steps": [{"id": "s1", "type": "llm", "prompt": "Hello"}]}
        path.write_text(yaml.dump(data))
        wf = WorkflowStore.load_workflow_from_file(path)
        assert wf is not None
        assert wf.name == "custom"


class TestExecutionTracking:
    def test_record_and_list(self, tmp_path):
        db_path = tmp_path / "test.db"
        WorkflowStore.record_execution_start(db_path, "t1", "wf1", "exec1", {"k": "v"})
        execs = WorkflowStore.list_executions(db_path, "wf1")
        assert len(execs) == 1
        assert execs[0]["thread_id"] == "t1"
        assert execs[0]["status"] == "running"

    def test_record_end(self, tmp_path):
        db_path = tmp_path / "test.db"
        WorkflowStore.record_execution_start(db_path, "t1", "wf1", "exec1")
        WorkflowStore.record_execution_end(db_path, "t1", "completed")
        execs = WorkflowStore.list_executions(db_path, "wf1")
        assert execs[0]["status"] == "completed"

    def test_delete_execution(self, tmp_path):
        db_path = tmp_path / "test.db"
        WorkflowStore.record_execution_start(db_path, "t1", "wf1", "exec1")
        assert WorkflowStore.delete_execution("t1", db_path) is True
        assert WorkflowStore.list_executions(db_path, "wf1") == []
