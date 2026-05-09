# tests/test_workflow_executor.py
from pathlib import Path
from unittest.mock import MagicMock

from cliver.workflow.executor import WorkflowExecutor


class TestWorkflowExecutor:
    def test_init(self, tmp_path):
        config = MagicMock()
        config.workflow_runs_dir = str(tmp_path / "runs")
        executor = WorkflowExecutor(app_config=config)
        assert executor._runs_dir == Path(tmp_path / "runs")

    def test_outputs_dir_structure(self, tmp_path):
        config = MagicMock()
        config.workflow_runs_dir = str(tmp_path / "runs")
        executor = WorkflowExecutor(app_config=config)
        path = executor._make_outputs_dir("my-workflow", "exec-001")
        assert "my-workflow" in str(path)
        assert "exec-001" in str(path)

    def test_checkpoints_dir_structure(self, tmp_path):
        config = MagicMock()
        config.workflow_runs_dir = str(tmp_path / "runs")
        executor = WorkflowExecutor(app_config=config)
        path = executor._checkpoints_db_path("my-workflow")
        assert "my-workflow" in str(path)
        assert "checkpoints" in str(path)
        assert str(path).endswith("checkpoints.db")

    def test_init_default_runs_dir(self):
        config = MagicMock(spec=[])
        executor = WorkflowExecutor(app_config=config)
        assert "workflow-runs" in str(executor._runs_dir)
