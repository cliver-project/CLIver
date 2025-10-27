"""
Tests for the Cliver Workflow Engine with mocked results.
"""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest
import yaml

from cliver.llm import TaskExecutor
from cliver.workflow.steps.function_step import FunctionStepExecutor
from cliver.workflow.steps.llm_step import LLMStepExecutor
from cliver.workflow.steps.workflow_step import WorkflowStepExecutor
from cliver.workflow.workflow_executor import WorkflowExecutor
from cliver.workflow.workflow_manager_local import LocalDirectoryWorkflowManager
from cliver.workflow.workflow_models import (
    ExecutionContext,
    ExecutionResult,
    FunctionStep,
    LLMStep,
    StepExecutionInfo,
    StepType,
    WorkflowStep,
)


@pytest.fixture
def mock_task_executor():
    """Create a mock TaskExecutor."""
    return Mock(spec=TaskExecutor)


@pytest.fixture
def workflow_manager(mock_task_executor):
    """Create a LocalDirectoryWorkflowManager with mock dependencies."""
    return LocalDirectoryWorkflowManager()


@pytest.fixture
def workflow_executor(workflow_manager, mock_task_executor):
    """Create a WorkflowExecutor with mock dependencies."""
    from cliver.workflow.persistence.local_cache import LocalCacheProvider

    return WorkflowExecutor(
        task_executor=mock_task_executor,
        workflow_manager=workflow_manager,
        persistence_provider=LocalCacheProvider(),
    )


class TestFunctionStepExecutor:
    """Tests for FunctionStepExecutor with mocked results."""

    def test_function_step_executor_creation(self):
        """Test creating a FunctionStepExecutor."""
        function_step = FunctionStep(
            id="test_func",
            name="Test Function",
            function="cliver.workflow.examples.process_results",
            inputs={"greeting": "Hello", "analysis": "Test analysis"},
            outputs=["result"],
        )
        executor = FunctionStepExecutor(function_step)
        assert executor.step == function_step

    @pytest.mark.asyncio
    async def test_execute_function_step_success(self):
        """Test successful execution of a function step with mocked function."""
        function_step = FunctionStep(
            id="test_func",
            name="Test Function",
            function="cliver.workflow.examples.process_results",
            inputs={"greeting": "Hello", "analysis": "Test analysis"},
            outputs=["result"],
        )
        executor = FunctionStepExecutor(function_step)

        # Create context with inputs
        context = ExecutionContext(
            workflow_name="test",
            inputs={"user_name": "John"},
            variables={"user_name": "John"},
            outputs={"analysis": "Test analysis"},
        )

        # Mock the function execution by patching the module import and function
        with patch("cliver.workflow.steps.function_step.importlib.import_module") as mock_import:
            # Create a mock module
            mock_module = Mock()
            mock_import.return_value = mock_module

            # Set the function attribute on the mock module to return our expected result
            mock_module.process_results = Mock(return_value="Processed result")

            # Execute the step
            result = await executor.execute(context)

            # Verify the result
            assert result.success is True
            assert result.step_id == "test_func"
            assert "result" in result.outputs
            assert result.outputs["result"] == "Processed result"

    @pytest.mark.asyncio
    async def test_execute_function_step_with_async_function(self):
        """Test execution of an async function step with mocked function."""
        # Create a function step that uses an async function
        function_step = FunctionStep(
            id="test_func_async",
            name="Test Async Function",
            function="cliver.workflow.examples.async_compute_something",
            inputs={"greeting": "Hello"},
            outputs=["result"],
        )
        executor = FunctionStepExecutor(function_step)

        # Create context
        context = ExecutionContext(
            workflow_name="test",
            inputs={"greeting": "Hello"},
            variables={"greeting": "Hello"},
        )

        # Mock the async function execution
        async def mock_async_function(**kwargs):
            return "Async result"

        with patch("cliver.workflow.steps.function_step.importlib.import_module") as mock_import:
            # Create a mock module
            mock_module = Mock()
            mock_import.return_value = mock_module

            # Set the function attribute on the mock module to return our async function
            mock_module.async_compute_something = mock_async_function

            # Execute the step
            result = await executor.execute(context)

            # Verify the result
            assert result.success is True
            assert result.step_id == "test_func_async"
            assert "result" in result.outputs
            assert result.outputs["result"] == "Async result"


class TestLLMStepExecutor:
    """Tests for LLMStepExecutor with mocked results."""

    def test_llm_step_executor_creation(self, mock_task_executor):
        """Test creating an LLMStepExecutor."""
        llm_step = LLMStep(
            id="test_llm",
            name="Test LLM",
            prompt="Analyze {{ inputs.topic }}",
            model="gpt-3.5-turbo",
            outputs=["analysis"],
        )
        executor = LLMStepExecutor(llm_step, mock_task_executor)
        assert executor.step == llm_step
        assert executor.task_executor == mock_task_executor

    @pytest.mark.asyncio
    async def test_execute_llm_step_success(self, mock_task_executor):
        """Test successful execution of an LLM step with mocked response."""
        llm_step = LLMStep(
            id="test_llm",
            name="Test LLM",
            prompt="Analyze {{ inputs.topic }}",
            model="gpt-3.5-turbo",
            outputs=["analysis"],
        )
        executor = LLMStepExecutor(llm_step, mock_task_executor)

        # Create context with inputs
        context = ExecutionContext(
            workflow_name="test",
            inputs={"topic": "AI technologies"},
            variables={"topic": "AI technologies"},
        )

        # Mock the resolve_variable method for different parameters
        def mock_resolve_variable(value, context):
            if value == "Analyze {{ inputs.topic }}":
                return "Analyze AI technologies"
            elif value == "gpt-3.5-turbo":
                return "gpt-3.5-turbo"
            return value

        with patch.object(executor, "resolve_variable", side_effect=mock_resolve_variable):
            # Mock the task executor's process_user_input method
            mock_response = Mock()
            mock_response.content = "Analysis of AI technologies: Machine learning, deep learning, and neural networks."
            mock_task_executor.process_user_input = AsyncMock(return_value=mock_response)

            # Execute the step
            result = await executor.execute(context)

            # Verify the result
            assert result.success is True
            assert result.step_id == "test_llm"
            assert "analysis" in result.outputs
            assert (
                result.outputs["analysis"]
                == "Analysis of AI technologies: Machine learning, deep learning, and neural networks."
            )

            # Verify the task executor was called with correct parameters
            mock_task_executor.process_user_input.assert_called_once()
            call_args = mock_task_executor.process_user_input.call_args.kwargs
            assert call_args["user_input"] == "Analyze AI technologies"
            assert call_args["model"] == "gpt-3.5-turbo"

    @pytest.mark.asyncio
    async def test_execute_llm_step_with_streaming(self, mock_task_executor):
        """Test execution of an LLM step with streaming response."""
        # Create an LLM step with streaming enabled
        llm_step = LLMStep(
            id="test_llm_stream",
            name="Test LLM Stream",
            prompt="Generate creative ideas",
            stream=True,
            outputs=["ideas"],
        )
        executor = LLMStepExecutor(llm_step, mock_task_executor)

        # Create context
        context = ExecutionContext(workflow_name="test", inputs={}, variables={})

        # Mock the resolve_variable method for different parameters
        def mock_resolve_variable(value, context):
            if value == "Generate creative ideas":
                return "Generate creative ideas"
            elif value is True:  # for stream parameter
                return True
            return value

        with patch.object(executor, "resolve_variable", side_effect=mock_resolve_variable):
            # Mock the task executor's stream_user_input method
            async def mock_stream():
                chunks = [
                    "Creative ",
                    "ideas ",
                    "for ",
                    "AI: ",
                    "1. ",
                    "Chatbots",
                    " 2. ",
                    "Image ",
                    "generation",
                ]
                for chunk in chunks:
                    mock_chunk = Mock()
                    mock_chunk.content = chunk
                    yield mock_chunk

            mock_task_executor.stream_user_input = Mock(return_value=mock_stream())

            # Execute the step
            result = await executor.execute(context)

            # Verify the result
            assert result.success is True
            assert result.step_id == "test_llm_stream"
            assert "ideas" in result.outputs
            assert result.outputs["ideas"] == "Creative ideas for AI: 1. Chatbots 2. Image generation"


class TestWorkflowStepExecutor:
    """Tests for WorkflowStepExecutor with mocked results."""

    def test_workflow_step_executor_creation(self):
        """Test creating a WorkflowStepExecutor."""
        workflow_step = WorkflowStep(
            id="test_workflow",
            name="Test Workflow",
            workflow="sub_workflow.yaml",
            workflow_inputs={"input_data": "{{ func_step.result }}"},
            outputs=["sub_result"],
        )
        # Create a mock workflow executor
        mock_workflow_executor = Mock()
        executor = WorkflowStepExecutor(workflow_step, mock_workflow_executor)
        assert executor.step == workflow_step
        assert executor.workflow_executor == mock_workflow_executor

    @pytest.mark.asyncio
    async def test_execute_workflow_step_success(self):
        """Test successful execution of a workflow step with mocked sub-workflow."""
        # Create a workflow step
        workflow_step = WorkflowStep(
            id="test_workflow",
            name="Test Workflow",
            workflow="sub_workflow.yaml",
            workflow_inputs={"input_data": "processed result"},
            outputs=["sub_result"],
        )

        # Create a mock workflow executor
        mock_workflow_executor = Mock()

        executor = WorkflowStepExecutor(workflow_step, mock_workflow_executor)

        # Create context with inputs
        context = ExecutionContext(
            workflow_name="test",
            inputs={"data": "input data"},
            steps={
                "func_step": StepExecutionInfo(
                    id="func_step",
                    name="Function Step",
                    type=StepType.FUNCTION,
                    outputs={"result": "processed result"},
                )
            },
        )

        # Mock the workflow executor's execute_workflow method
        mock_step_info = Mock()
        mock_step_info.outputs = {"sub_result": "Sub-workflow result"}

        mock_context = Mock()
        mock_context.steps = {"sub_step": mock_step_info}

        mock_result = Mock()
        mock_result.context = mock_context
        mock_result.success = True
        mock_result.error = None
        mock_result.execution_time = 0.1
        mock_workflow_executor.execute_workflow = AsyncMock(return_value=mock_result)

        # Execute the step
        result = await executor.execute(context)

        # Verify the result
        assert result.success is True
        assert result.step_id == "test_workflow"
        assert "sub_result" in result.outputs
        assert result.outputs["sub_result"] == "Sub-workflow result"

        # Verify the workflow executor was called with correct parameters
        mock_workflow_executor.execute_workflow.assert_called_once()
        call_args = mock_workflow_executor.execute_workflow.call_args
        assert call_args.kwargs["workflow_name"] == "sub_workflow.yaml"
        assert "input_data" in call_args.kwargs["inputs"]
        assert call_args.kwargs["inputs"]["input_data"] == "processed result"


class TestLocalDirectoryWorkflowManager:
    """Tests for WorkflowManager with mocked step executors."""

    @pytest.mark.asyncio
    async def test_execute_workflow_with_mocked_steps(self, workflow_manager, workflow_executor):
        """Test workflow execution with mocked step executors."""
        # Create a simple workflow YAML structure that can be loaded
        workflow_data = {
            "name": "test_workflow",
            "description": "Test workflow",
            "steps": [
                {
                    "id": "func_step",
                    "name": "Function Step",
                    "type": "function",
                    "function": "cliver.workflow.examples.process_results",
                    "inputs": {"greeting": "Hello", "analysis": "Test"},
                    "outputs": ["result"],
                },
                {
                    "id": "llm_step",
                    "name": "LLM Step",
                    "type": "llm",
                    "prompt": "Analyze topic",
                    "model": "gpt-3.5-turbo",
                    "outputs": ["analysis"],
                },
            ],
        }

        # Create a temporary directory for workflow files
        temp_dir = tempfile.TemporaryDirectory()
        try:
            # Set the workflow manager to use this temporary directory
            workflow_manager.workflow_dirs = [Path(temp_dir.name)]

            # Create a workflow file in the temporary directory
            workflow_file = Path(temp_dir.name) / "test_workflow.yaml"
            with open(workflow_file, "w") as f:
                yaml.dump(workflow_data, f)

            # Refresh the workflow manager cache
            workflow_manager.refresh_workflows()

            # Mock the step executor creation to return mocked executors
            with patch.object(WorkflowExecutor, "_create_step_executor") as mock_create:
                # Create mock executors with predefined results
                mock_func_executor = AsyncMock()
                mock_func_executor.execute_with_retry = AsyncMock(
                    return_value=ExecutionResult(
                        step_id="func_step",
                        outputs={"result": "Function result"},
                        success=True,
                        error=None,
                        execution_time=0.1,
                    )
                )
                mock_func_executor.evaluate_condition = Mock(return_value=True)

                mock_llm_executor = AsyncMock()
                mock_llm_executor.execute_with_retry = AsyncMock(
                    return_value=ExecutionResult(
                        step_id="llm_step",
                        outputs={"analysis": "LLM analysis"},
                        success=True,
                        error=None,
                        execution_time=0.2,
                    )
                )
                mock_llm_executor.evaluate_condition = Mock(return_value=True)

                # Configure the mock to return different executors for different step types
                def side_effect(step):
                    if step.id == "func_step":
                        return mock_func_executor
                    elif step.id == "llm_step":
                        return mock_llm_executor
                    else:
                        raise ValueError(f"Unknown step: {step.id}")

                mock_create.side_effect = side_effect

                # Execute the workflow using workflow_executor
                inputs = {"user_name": "John", "topic": "AI"}
                result = await workflow_executor.execute_workflow("test_workflow", inputs)

                # Verify the final result - now returns WorkflowExecutionState
                assert result.status == "completed"
                # Check that step execution info is stored correctly
                assert "func_step" in result.context.steps
                assert "llm_step" in result.context.steps
                assert result.context.steps["func_step"].outputs["result"] == "Function result"
                assert result.context.steps["llm_step"].outputs["analysis"] == "LLM analysis"

                # Verify each executor was called
                mock_func_executor.execute_with_retry.assert_called_once()
                mock_llm_executor.execute_with_retry.assert_called_once()
        finally:
            # Clean up temporary directory
            temp_dir.cleanup()


if __name__ == "__main__":
    pytest.main([__file__])
