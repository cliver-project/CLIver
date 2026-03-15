"""
Tests for the CLIver Workflow Engine.
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
    WorkflowStep,
)


@pytest.fixture
def mock_task_executor():
    return Mock(spec=TaskExecutor)


@pytest.fixture
def workflow_manager():
    return LocalDirectoryWorkflowManager()


@pytest.fixture
def workflow_executor(workflow_manager, mock_task_executor):
    from cliver.workflow.persistence.local_cache import LocalCacheProvider

    return WorkflowExecutor(
        task_executor=mock_task_executor,
        workflow_manager=workflow_manager,
        persistence_provider=LocalCacheProvider(),
    )


class TestFunctionStepExecutor:
    def test_function_step_executor_creation(self):
        step = FunctionStep(
            id="test_func",
            name="Test Function",
            function="cliver.workflow.examples.process_results",
            inputs={"greeting": "Hello", "analysis": "Test analysis"},
            outputs=["result"],
        )
        executor = FunctionStepExecutor(step)
        assert executor.step == step

    @pytest.mark.asyncio
    async def test_execute_function_step_success(self):
        step = FunctionStep(
            id="test_func",
            name="Test Function",
            function="cliver.workflow.examples.process_results",
            inputs={"greeting": "Hello", "analysis": "Test analysis"},
            outputs=["result"],
        )
        executor = FunctionStepExecutor(step)

        context = ExecutionContext(
            workflow_name="test",
            inputs={"user_name": "John"},
        )

        with patch("cliver.workflow.steps.function_step.importlib.import_module") as mock_import:
            mock_module = Mock()
            mock_import.return_value = mock_module
            mock_module.process_results = Mock(return_value="Processed result")

            result = await executor.execute(context)

            assert result.success is True
            assert result.step_id == "test_func"
            assert result.outputs["result"] == "Processed result"

    @pytest.mark.asyncio
    async def test_execute_function_step_with_async_function(self):
        step = FunctionStep(
            id="test_func_async",
            name="Test Async Function",
            function="cliver.workflow.examples.async_compute_something",
            inputs={"greeting": "Hello"},
            outputs=["result"],
        )
        executor = FunctionStepExecutor(step)

        context = ExecutionContext(
            workflow_name="test",
            inputs={"greeting": "Hello"},
        )

        async def mock_async_function(**kwargs):
            return "Async result"

        with patch("cliver.workflow.steps.function_step.importlib.import_module") as mock_import:
            mock_module = Mock()
            mock_import.return_value = mock_module
            mock_module.async_compute_something = mock_async_function

            result = await executor.execute(context)

            assert result.success is True
            assert result.step_id == "test_func_async"
            assert result.outputs["result"] == "Async result"


class TestLLMStepExecutor:
    def test_llm_step_executor_creation(self, mock_task_executor):
        step = LLMStep(
            id="test_llm",
            name="Test LLM",
            prompt="Analyze {{ inputs.topic }}",
            model="gpt-3.5-turbo",
            outputs=["analysis"],
        )
        executor = LLMStepExecutor(step, mock_task_executor)
        assert executor.step == step

    @pytest.mark.asyncio
    async def test_execute_llm_step_success(self, mock_task_executor):
        step = LLMStep(
            id="test_llm",
            name="Test LLM",
            prompt="Analyze {{ inputs.topic }}",
            model="gpt-3.5-turbo",
            outputs=["analysis"],
        )
        executor = LLMStepExecutor(step, mock_task_executor)

        context = ExecutionContext(
            workflow_name="test",
            inputs={"topic": "AI technologies"},
        )

        def mock_resolve(value, ctx):
            if value == "Analyze {{ inputs.topic }}":
                return "Analyze AI technologies"
            if value == "gpt-3.5-turbo":
                return "gpt-3.5-turbo"
            return value

        with patch.object(executor, "resolve_variable", side_effect=mock_resolve):
            mock_response = Mock()
            mock_response.content = "AI analysis result"
            mock_task_executor.process_user_input = AsyncMock(return_value=mock_response)

            result = await executor.execute(context)

            assert result.success is True
            assert result.outputs["analysis"] == "AI analysis result"
            mock_task_executor.process_user_input.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_llm_step_with_streaming(self, mock_task_executor):
        step = LLMStep(
            id="test_llm_stream",
            name="Test LLM Stream",
            prompt="Generate creative ideas",
            stream=True,
            outputs=["ideas"],
        )
        executor = LLMStepExecutor(step, mock_task_executor)
        context = ExecutionContext(workflow_name="test", inputs={})

        def mock_resolve(value, ctx):
            return value

        with patch.object(executor, "resolve_variable", side_effect=mock_resolve):

            async def mock_stream():
                for chunk_text in ["Creative ", "ideas ", "for AI"]:
                    mock_chunk = Mock()
                    mock_chunk.content = chunk_text
                    yield mock_chunk

            mock_task_executor.stream_user_input = Mock(return_value=mock_stream())

            result = await executor.execute(context)

            assert result.success is True
            assert result.outputs["ideas"] == "Creative ideas for AI"


class TestWorkflowStepExecutor:
    def test_workflow_step_executor_creation(self):
        step = WorkflowStep(
            id="test_workflow",
            name="Test Workflow",
            workflow="sub_workflow.yaml",
            workflow_inputs={"input_data": "{{ func_step.outputs.result }}"},
            outputs=["sub_result"],
        )
        mock_executor = Mock()
        executor = WorkflowStepExecutor(step, mock_executor)
        assert executor.step == step

    @pytest.mark.asyncio
    async def test_execute_workflow_step_success(self):
        step = WorkflowStep(
            id="test_workflow",
            name="Test Workflow",
            workflow="sub_workflow.yaml",
            workflow_inputs={"input_data": "processed result"},
            outputs=["sub_result"],
        )

        mock_executor = Mock()
        executor = WorkflowStepExecutor(step, mock_executor)

        context = ExecutionContext(
            workflow_name="test",
            inputs={"data": "input data"},
            steps={
                "func_step": {"outputs": {"result": "processed result"}},
            },
        )

        # Mock the sub-workflow result
        mock_sub_context = ExecutionContext(
            workflow_name="sub_workflow",
            steps={"sub_step": {"outputs": {"sub_result": "Sub-workflow result"}}},
        )
        mock_result = Mock()
        mock_result.context = mock_sub_context
        mock_result.status = "completed"
        mock_result.error = None
        mock_result.execution_time = 0.1
        mock_executor.execute_workflow = AsyncMock(return_value=mock_result)

        result = await executor.execute(context)

        assert result.success is True
        assert result.outputs["sub_result"] == "Sub-workflow result"
        mock_executor.execute_workflow.assert_called_once()


class TestLocalDirectoryWorkflowManager:
    @pytest.mark.asyncio
    async def test_execute_workflow_with_mocked_steps(self, workflow_manager, workflow_executor):
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

        with tempfile.TemporaryDirectory() as temp_dir:
            workflow_manager.workflow_dirs = [Path(temp_dir)]
            workflow_file = Path(temp_dir) / "test_workflow.yaml"
            with open(workflow_file, "w") as f:
                yaml.dump(workflow_data, f)
            workflow_manager.refresh_workflows()

            with patch.object(WorkflowExecutor, "_create_step_executor") as mock_create:
                mock_func_executor = AsyncMock()
                mock_func_executor.execute = AsyncMock(
                    return_value=ExecutionResult(
                        step_id="func_step",
                        outputs={"result": "Function result"},
                        success=True,
                        execution_time=0.1,
                    )
                )

                mock_llm_executor = AsyncMock()
                mock_llm_executor.execute = AsyncMock(
                    return_value=ExecutionResult(
                        step_id="llm_step",
                        outputs={"analysis": "LLM analysis"},
                        success=True,
                        execution_time=0.2,
                    )
                )

                def side_effect(step):
                    if step.id == "func_step":
                        return mock_func_executor
                    elif step.id == "llm_step":
                        return mock_llm_executor
                    raise ValueError(f"Unknown step: {step.id}")

                mock_create.side_effect = side_effect

                result = await workflow_executor.execute_workflow("test_workflow", {"user_name": "John"})

                assert result.status == "completed"
                assert "func_step" in result.context.steps
                assert "llm_step" in result.context.steps
                assert result.context.steps["func_step"]["outputs"]["result"] == "Function result"
                assert result.context.steps["llm_step"]["outputs"]["analysis"] == "LLM analysis"

                mock_func_executor.execute.assert_called_once()
                mock_llm_executor.execute.assert_called_once()
