"""
Tests for nested step outputs in the CLIver Workflow Engine.
"""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest
import yaml

from cliver.llm import TaskExecutor
from cliver.workflow.workflow_executor import WorkflowExecutor
from cliver.workflow.workflow_manager_local import LocalDirectoryWorkflowManager
from cliver.workflow.workflow_models import ExecutionResult


@pytest.fixture
def mock_task_executor():
    return Mock(spec=TaskExecutor)


@pytest.fixture
def workflow_manager(mock_task_executor, workflow_path):
    return LocalDirectoryWorkflowManager(workflow_dirs=[workflow_path])


@pytest.fixture
def workflow_executor(workflow_manager, mock_task_executor, workflow_cache_path):
    from cliver.workflow.persistence.local_cache import LocalCacheProvider

    return WorkflowExecutor(
        task_executor=mock_task_executor,
        workflow_manager=workflow_manager,
        persistence_provider=LocalCacheProvider(cache_dir=workflow_cache_path),
    )


class TestNestedStepOutputs:
    @pytest.mark.asyncio
    async def test_nested_outputs_are_stored_correctly(self, workflow_manager, workflow_executor):
        workflow_data = {
            "name": "test_nested_outputs",
            "description": "Test workflow for nested outputs",
            "steps": [
                {
                    "id": "step1",
                    "name": "First Step",
                    "type": "function",
                    "function": "cliver.workflow.examples.compute_something",
                    "inputs": {"greeting": "Hello World"},
                    "outputs": ["result"],
                },
                {
                    "id": "step2",
                    "name": "Second Step",
                    "type": "function",
                    "function": "cliver.workflow.examples.process_results",
                    "inputs": {
                        "greeting": "{{ step1.outputs.result }}",
                        "analysis": "Test analysis",
                    },
                    "outputs": ["result"],
                },
            ],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            workflow_manager.workflow_dirs = [Path(temp_dir)]
            workflow_file = Path(temp_dir) / "test_nested_outputs.yaml"
            with open(workflow_file, "w") as f:
                yaml.dump(workflow_data, f)
            workflow_manager.refresh_workflows()

            with patch.object(WorkflowExecutor, "_create_step_executor") as mock_create:
                mock_step1 = AsyncMock()
                mock_step1.execute = AsyncMock(
                    return_value=ExecutionResult(
                        step_id="step1",
                        outputs={"result": "Computed result from step 1"},
                        success=True,
                        execution_time=0.1,
                    )
                )

                mock_step2 = AsyncMock()
                mock_step2.execute = AsyncMock(
                    return_value=ExecutionResult(
                        step_id="step2",
                        outputs={"result": "Processed result from step 2"},
                        success=True,
                        execution_time=0.1,
                    )
                )

                mock_create.side_effect = lambda step: mock_step1 if step.id == "step1" else mock_step2

                result = await workflow_executor.execute_workflow("test_nested_outputs")

                assert result.status == "completed"
                assert "step1" in result.context.steps
                assert "step2" in result.context.steps
                assert result.context.steps["step1"]["outputs"]["result"] == "Computed result from step 1"
                assert result.context.steps["step2"]["outputs"]["result"] == "Processed result from step 2"

    @pytest.mark.asyncio
    async def test_nested_outputs_persistence_on_resume(self, workflow_manager, workflow_executor):
        workflow_data = {
            "name": "test_resume",
            "description": "Test resume",
            "steps": [
                {
                    "id": "step1",
                    "name": "First Step",
                    "type": "function",
                    "function": "cliver.workflow.examples.compute_something",
                    "inputs": {"greeting": "Hello"},
                    "outputs": ["result"],
                },
                {
                    "id": "step2",
                    "name": "Second Step",
                    "type": "function",
                    "function": "cliver.workflow.examples.process_results",
                    "inputs": {"greeting": "{{ step1.outputs.result }}"},
                    "outputs": ["result"],
                },
            ],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            workflow_manager.workflow_dirs = [Path(temp_dir)]
            workflow_file = Path(temp_dir) / "test_resume.yaml"
            with open(workflow_file, "w") as f:
                yaml.dump(workflow_data, f)
            workflow_manager.refresh_workflows()

            with patch.object(WorkflowExecutor, "_create_step_executor") as mock_create:
                mock_step1 = AsyncMock()
                mock_step1.execute = AsyncMock(
                    return_value=ExecutionResult(
                        step_id="step1",
                        outputs={"result": "Step 1 result"},
                        success=True,
                        execution_time=0.1,
                    )
                )
                mock_step2 = AsyncMock()
                mock_step2.execute = AsyncMock(
                    return_value=ExecutionResult(
                        step_id="step2",
                        outputs={"result": "Step 2 result"},
                        success=True,
                        execution_time=0.1,
                    )
                )
                mock_create.side_effect = lambda step: mock_step1 if step.id == "step1" else mock_step2

                execution_id = "test-resume-id"
                result = await workflow_executor.execute_workflow("test_resume", execution_id=execution_id)
                assert result.status == "completed"

                # Pause completed workflow (won't actually pause since it's completed)
                await workflow_executor.pause_execution("test_resume", execution_id)

                # Resume — since status is now completed (not paused), it won't resume
                resumed = await workflow_executor.resume_execution("test_resume", execution_id)
                # Completed workflows don't resume
                assert resumed is None
