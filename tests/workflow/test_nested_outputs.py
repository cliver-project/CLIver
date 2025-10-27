"""
Tests for nested step outputs in the Cliver Workflow Engine.
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, Mock, patch
from pathlib import Path
import tempfile
import yaml

from cliver.workflow.workflow_models import (
    Workflow, ExecutionContext, ExecutionResult,
    FunctionStep, LLMStep, WorkflowStep, StepType
)
from cliver.workflow.workflow_manager_local import LocalDirectoryWorkflowManager
from cliver.workflow.workflow_executor import WorkflowExecutor
from cliver.workflow.steps.function_step import FunctionStepExecutor
from cliver.workflow.steps.llm_step import LLMStepExecutor
from cliver.workflow.steps.workflow_step import WorkflowStepExecutor
from cliver.llm import TaskExecutor


@pytest.fixture
def mock_task_executor():
    """Create a mock TaskExecutor."""
    return Mock(spec=TaskExecutor)


@pytest.fixture
def workflow_manager(mock_task_executor, workflow_path):
    """Create a LocalDirectoryWorkflowManager with mock dependencies."""
    return LocalDirectoryWorkflowManager(workflow_dirs=[workflow_path])


@pytest.fixture
def workflow_executor(workflow_manager, mock_task_executor, workflow_cache_path):
    """Create a WorkflowExecutor with mock dependencies."""
    from cliver.workflow.persistence.local_cache import LocalCacheProvider
    return WorkflowExecutor(
        task_executor=mock_task_executor,
        workflow_manager=workflow_manager,
        persistence_provider=LocalCacheProvider(cache_dir=workflow_cache_path)
    )


class TestNestedStepOutputs:
    """Tests for nested step outputs functionality."""

    @pytest.mark.asyncio
    async def test_nested_outputs_are_stored_correctly(self, workflow_manager, workflow_executor):
        """Test that step outputs are stored in both flat and nested structures."""
        # Create a simple workflow YAML structure
        workflow_data = {
            "name": "test_nested_outputs",
            "description": "Test workflow for nested outputs",
            "steps": [
                {
                    "id": "step1",
                    "name": "First Step",
                    "type": "function",
                    "function": "cliver.workflow.examples.compute_something",
                    "inputs": {
                        "greeting": "Hello World"
                    },
                    "outputs": ["result"]
                },
                {
                    "id": "step2",
                    "name": "Second Step",
                    "type": "function",
                    "function": "cliver.workflow.examples.process_results",
                    "inputs": {
                        "greeting": "{{ step1.result }}",
                        "analysis": "Test analysis"
                    },
                    "outputs": ["result"]
                }
            ]
        }

        # Create a temporary directory for workflow files
        temp_dir = tempfile.TemporaryDirectory()
        try:
            # Set the workflow manager to use this temporary directory
            workflow_manager.workflow_dirs = [Path(temp_dir.name)]

            # Create a workflow file in the temporary directory
            workflow_file = Path(temp_dir.name) / "test_nested_outputs.yaml"
            with open(workflow_file, 'w') as f:
                yaml.dump(workflow_data, f)

            # Refresh the workflow manager cache
            workflow_manager.refresh_workflows()

            # Mock the step executor creation to return mocked executors
            with patch.object(WorkflowExecutor, '_create_step_executor') as mock_create:
                # Create mock executors with predefined results
                mock_step1_executor = AsyncMock()
                mock_step1_executor.execute_with_retry = AsyncMock(return_value=ExecutionResult(
                    step_id="step1",
                    outputs={"result": "Computed result from step 1"},
                    success=True,
                    error=None,
                    execution_time=0.1
                ))
                mock_step1_executor.evaluate_condition = Mock(return_value=True)

                mock_step2_executor = AsyncMock()
                mock_step2_executor.execute_with_retry = AsyncMock(return_value=ExecutionResult(
                    step_id="step2",
                    outputs={"result": "Processed result from step 2"},
                    success=True,
                    error=None,
                    execution_time=0.1
                ))
                mock_step2_executor.evaluate_condition = Mock(return_value=True)

                # Configure the mock to return different executors for different steps
                def side_effect(step):
                    if step.id == "step1":
                        return mock_step1_executor
                    elif step.id == "step2":
                        return mock_step2_executor
                    else:
                        raise ValueError(f"Unknown step: {step.id}")

                mock_create.side_effect = side_effect

                # Execute the workflow
                result = await workflow_executor.execute_workflow("test_nested_outputs")

                # Verify the final result
                assert result.status == "completed"

                # Check that step execution info is stored correctly
                assert "step1" in result.context.steps
                assert "step2" in result.context.steps
                assert result.context.steps["step1"].outputs["result"] == "Computed result from step 1"
                assert result.context.steps["step2"].outputs["result"] == "Processed result from step 2"

        finally:
            # Clean up temporary directory
            temp_dir.cleanup()

    @pytest.mark.asyncio
    async def test_nested_outputs_persistence_on_resume(self, workflow_manager, workflow_executor):
        """Test that nested outputs are properly loaded when resuming a workflow."""
        # Create a simple workflow YAML structure
        workflow_data = {
            "name": "test_resume_nested_outputs",
            "description": "Test workflow for nested outputs on resume",
            "steps": [
                {
                    "id": "step1",
                    "name": "First Step",
                    "type": "function",
                    "function": "cliver.workflow.examples.compute_something",
                    "inputs": {
                        "greeting": "Hello World"
                    },
                    "outputs": ["result"]
                },
                {
                    "id": "step2",
                    "name": "Second Step",
                    "type": "function",
                    "function": "cliver.workflow.examples.process_results",
                    "inputs": {
                        "greeting": "{{ step1.result }}",
                        "analysis": "Test analysis"
                    },
                    "outputs": ["result"]
                }
            ]
        }

        # Create a temporary directory for workflow files
        temp_dir = tempfile.TemporaryDirectory()
        try:
            # Set the workflow manager to use this temporary directory
            workflow_manager.workflow_dirs = [Path(temp_dir.name)]

            # Create a workflow file in the temporary directory
            workflow_file = Path(temp_dir.name) / "test_resume_nested_outputs.yaml"
            with open(workflow_file, 'w') as f:
                yaml.dump(workflow_data, f)

            # Refresh the workflow manager cache
            workflow_manager.refresh_workflows()

            # Mock the step executor creation to return mocked executors
            with patch.object(WorkflowExecutor, '_create_step_executor') as mock_create:
                # Create mock executors with predefined results
                mock_step1_executor = AsyncMock()
                mock_step1_executor.execute_with_retry = AsyncMock(return_value=ExecutionResult(
                    step_id="step1",
                    outputs={"result": "Computed result from step 1"},
                    success=True,
                    error=None,
                    execution_time=0.1
                ))
                mock_step1_executor.evaluate_condition = Mock(return_value=True)

                mock_step2_executor = AsyncMock()
                mock_step2_executor.execute_with_retry = AsyncMock(return_value=ExecutionResult(
                    step_id="step2",
                    outputs={"result": "Processed result from step 2"},
                    success=True,
                    error=None,
                    execution_time=0.1
                ))
                mock_step2_executor.evaluate_condition = Mock(return_value=True)

                # Configure the mock to return different executors for different steps
                def side_effect(step):
                    if step.id == "step1":
                        return mock_step1_executor
                    elif step.id == "step2":
                        return mock_step2_executor
                    else:
                        raise ValueError(f"Unknown step: {step.id}")

                mock_create.side_effect = side_effect

                # Execute the workflow with a specific execution ID
                execution_id = "test-execution-id"
                result = await workflow_executor.execute_workflow("test_resume_nested_outputs", execution_id=execution_id)

                # Verify the initial execution
                assert result.status == "completed"

                # Now pause the workflow (this will create a paused state)
                await workflow_executor.pause_execution("test_resume_nested_outputs", execution_id)

                # Resume the workflow execution
                resumed_result = await workflow_executor.resume_execution("test_resume_nested_outputs", execution_id)

                # Verify that nested outputs are properly maintained when resuming
                assert resumed_result.status == "completed"

                # Check that step execution info is stored correctly after resume
                assert "step1" in resumed_result.context.steps
                assert "step2" in resumed_result.context.steps
                assert resumed_result.context.steps["step1"].outputs["result"] == "Computed result from step 1"
                assert resumed_result.context.steps["step2"].outputs["result"] == "Processed result from step 2"

        finally:
            # Clean up temporary directory
            temp_dir.cleanup()


if __name__ == "__main__":
    pytest.main([__file__])