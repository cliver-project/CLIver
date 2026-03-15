"""
Thorough tests for result propagation between workflow steps.

This is the most critical behavior of the workflow engine:
step outputs must be correctly available to subsequent steps
via Jinja2 templating ({{ step_id.outputs.key }}).
"""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest
import yaml

from cliver.llm import TaskExecutor
from cliver.workflow.persistence.local_cache import LocalCacheProvider
from cliver.workflow.steps.function_step import FunctionStepExecutor
from cliver.workflow.workflow_executor import WorkflowExecutor
from cliver.workflow.workflow_manager_local import LocalDirectoryWorkflowManager
from cliver.workflow.workflow_models import (
    ExecutionContext,
    ExecutionResult,
    FunctionStep,
)

# ---------------------------------------------------------------------------
# Variable resolution (the foundation of result propagation)
# ---------------------------------------------------------------------------


class TestVariableResolution:
    """Test that resolve_variable correctly accesses context data."""

    def _make_executor(self):
        step = FunctionStep(id="s", name="s", function="m.f")
        return FunctionStepExecutor(step)

    def test_resolve_workflow_input(self):
        executor = self._make_executor()
        context = ExecutionContext(workflow_name="w", inputs={"name": "Alice"})
        assert executor.resolve_variable("{{ inputs.name }}", context) == "Alice"

    def test_resolve_input_shorthand(self):
        """Inputs are promoted to root level for shorthand access."""
        executor = self._make_executor()
        context = ExecutionContext(workflow_name="w", inputs={"name": "Bob"})
        assert executor.resolve_variable("{{ name }}", context) == "Bob"

    def test_resolve_step_output(self):
        executor = self._make_executor()
        context = ExecutionContext(
            workflow_name="w",
            steps={"step1": {"outputs": {"result": "hello"}}},
        )
        assert executor.resolve_variable("{{ step1.outputs.result }}", context) == "hello"

    def test_resolve_nested_step_output(self):
        executor = self._make_executor()
        context = ExecutionContext(
            workflow_name="w",
            steps={"fetch": {"outputs": {"data": {"count": 42, "items": ["a", "b"]}}}},
        )
        result = executor.resolve_variable("{{ fetch.outputs.data.count }}", context)
        assert result == "42"

    def test_resolve_multiple_references(self):
        executor = self._make_executor()
        context = ExecutionContext(
            workflow_name="w",
            inputs={"user": "Alice"},
            steps={"greet": {"outputs": {"msg": "Hello"}}},
        )
        result = executor.resolve_variable("{{ greet.outputs.msg }}, {{ inputs.user }}!", context)
        assert result == "Hello, Alice!"

    def test_resolve_non_string_passthrough(self):
        executor = self._make_executor()
        context = ExecutionContext(workflow_name="w")
        assert executor.resolve_variable(42, context) == 42
        assert executor.resolve_variable(None, context) is None
        assert executor.resolve_variable(True, context) is True

    def test_resolve_dict_recursively(self):
        executor = self._make_executor()
        context = ExecutionContext(workflow_name="w", inputs={"x": "val"})
        result = executor.resolve_variable({"a": "{{ inputs.x }}", "b": "static"}, context)
        assert result == {"a": "val", "b": "static"}

    def test_resolve_list_recursively(self):
        executor = self._make_executor()
        context = ExecutionContext(workflow_name="w", inputs={"x": "val"})
        result = executor.resolve_variable(["{{ inputs.x }}", "static"], context)
        assert result == ["val", "static"]

    def test_no_template_markers_skips_rendering(self):
        """Strings without {{ should be returned as-is (performance)."""
        executor = self._make_executor()
        context = ExecutionContext(workflow_name="w")
        assert executor.resolve_variable("plain text", context) == "plain text"

    def test_unresolved_variable_returns_empty(self):
        """Unknown variables render as empty string (Jinja2 default)."""
        executor = self._make_executor()
        context = ExecutionContext(workflow_name="w")
        result = executor.resolve_variable("{{ nonexistent }}", context)
        assert result == ""

    def test_step_output_does_not_leak_to_wrong_step(self):
        executor = self._make_executor()
        context = ExecutionContext(
            workflow_name="w",
            steps={
                "step_a": {"outputs": {"x": "from_a"}},
                "step_b": {"outputs": {"y": "from_b"}},
            },
        )
        assert executor.resolve_variable("{{ step_a.outputs.x }}", context) == "from_a"
        assert executor.resolve_variable("{{ step_b.outputs.y }}", context) == "from_b"
        # step_a doesn't have y
        assert executor.resolve_variable("{{ step_a.outputs.y }}", context) == ""


# ---------------------------------------------------------------------------
# Output extraction
# ---------------------------------------------------------------------------


class TestOutputExtraction:
    def _make_executor(self, outputs=None):
        step = FunctionStep(id="s", name="s", function="m.f", outputs=outputs)
        return FunctionStepExecutor(step)

    @pytest.mark.asyncio
    async def test_single_output(self):
        executor = self._make_executor(outputs=["result"])
        outputs = await executor.extract_outputs("hello")
        assert outputs == {"result": "hello"}

    @pytest.mark.asyncio
    async def test_multiple_outputs_from_dict(self):
        executor = self._make_executor(outputs=["a", "b"])
        outputs = await executor.extract_outputs({"a": 1, "b": 2, "c": 3})
        assert outputs == {"a": 1, "b": 2}

    @pytest.mark.asyncio
    async def test_multiple_outputs_from_list(self):
        executor = self._make_executor(outputs=["first", "second"])
        outputs = await executor.extract_outputs([10, 20])
        assert outputs == {"first": 10, "second": 20}

    @pytest.mark.asyncio
    async def test_no_outputs_defined(self):
        executor = self._make_executor(outputs=None)
        outputs = await executor.extract_outputs("anything")
        assert outputs == {}


# ---------------------------------------------------------------------------
# End-to-end: result propagation through a multi-step workflow
# ---------------------------------------------------------------------------


class TestEndToEndPropagation:
    """Test that outputs from step N are correctly accessible in step N+1."""

    @pytest.mark.asyncio
    async def test_three_step_chain(self):
        """Step 1 → Step 2 → Step 3: outputs propagate correctly."""
        workflow_data = {
            "name": "chain_test",
            "steps": [
                {
                    "id": "step1",
                    "name": "Step 1",
                    "type": "function",
                    "function": "m.f",
                    "outputs": ["data"],
                },
                {
                    "id": "step2",
                    "name": "Step 2",
                    "type": "function",
                    "function": "m.f",
                    "outputs": ["processed"],
                },
                {
                    "id": "step3",
                    "name": "Step 3",
                    "type": "function",
                    "function": "m.f",
                    "outputs": ["final"],
                },
            ],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            manager = LocalDirectoryWorkflowManager(workflow_dirs=[temp_dir])
            (Path(temp_dir) / "chain_test.yaml").write_text(yaml.dump(workflow_data))
            manager.refresh_workflows()

            executor = WorkflowExecutor(
                task_executor=Mock(spec=TaskExecutor),
                workflow_manager=manager,
                persistence_provider=LocalCacheProvider(cache_dir=temp_dir + "/cache"),
            )

            # Track what context each step receives
            received_contexts = []

            with patch.object(WorkflowExecutor, "_create_step_executor") as mock_create:
                call_count = [0]

                def make_mock_step(step):
                    mock = AsyncMock()
                    call_count[0] += 1

                    async def execute(ctx):
                        received_contexts.append((step.id, ctx.model_copy(deep=True)))
                        return ExecutionResult(
                            step_id=step.id,
                            outputs={step.outputs[0]: f"output_from_{step.id}"},
                            success=True,
                        )

                    mock.execute = execute
                    return mock

                mock_create.side_effect = make_mock_step

                result = await executor.execute_workflow("chain_test")

            assert result.status == "completed"

            # Step 1: no previous step outputs
            _, ctx1 = received_contexts[0]
            assert len(ctx1.steps) == 0

            # Step 2: should see step1's outputs
            _, ctx2 = received_contexts[1]
            assert "step1" in ctx2.steps
            assert ctx2.steps["step1"]["outputs"]["data"] == "output_from_step1"

            # Step 3: should see both step1 and step2 outputs
            _, ctx3 = received_contexts[2]
            assert "step1" in ctx3.steps
            assert "step2" in ctx3.steps
            assert ctx3.steps["step1"]["outputs"]["data"] == "output_from_step1"
            assert ctx3.steps["step2"]["outputs"]["processed"] == "output_from_step2"

            # Final state has all outputs
            assert result.context.steps["step1"]["outputs"]["data"] == "output_from_step1"
            assert result.context.steps["step2"]["outputs"]["processed"] == "output_from_step2"
            assert result.context.steps["step3"]["outputs"]["final"] == "output_from_step3"

    @pytest.mark.asyncio
    async def test_failed_step_stops_propagation(self):
        """When a step fails, the workflow stops and no further steps run."""
        workflow_data = {
            "name": "fail_test",
            "steps": [
                {"id": "ok", "name": "OK", "type": "function", "function": "m.f", "outputs": ["x"]},
                {"id": "fail", "name": "Fail", "type": "function", "function": "m.f", "outputs": ["y"]},
                {"id": "never", "name": "Never", "type": "function", "function": "m.f", "outputs": ["z"]},
            ],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            manager = LocalDirectoryWorkflowManager(workflow_dirs=[temp_dir])
            (Path(temp_dir) / "fail_test.yaml").write_text(yaml.dump(workflow_data))
            manager.refresh_workflows()

            executor = WorkflowExecutor(
                task_executor=Mock(spec=TaskExecutor),
                workflow_manager=manager,
                persistence_provider=LocalCacheProvider(cache_dir=temp_dir + "/cache"),
            )

            with patch.object(WorkflowExecutor, "_create_step_executor") as mock_create:

                def make_step(step):
                    mock = AsyncMock()
                    if step.id == "fail":
                        mock.execute = AsyncMock(
                            return_value=ExecutionResult(step_id="fail", success=False, error="boom")
                        )
                    else:
                        mock.execute = AsyncMock(
                            return_value=ExecutionResult(step_id=step.id, outputs={"x": "ok"}, success=True)
                        )
                    return mock

                mock_create.side_effect = make_step
                result = await executor.execute_workflow("fail_test")

            assert result.status == "failed"
            assert result.error == "boom"
            assert "ok" in result.context.steps  # completed before failure
            assert "never" not in result.context.steps  # never ran

    @pytest.mark.asyncio
    async def test_skipped_step_not_in_context(self):
        """Skipped steps don't produce outputs and don't appear in context."""
        workflow_data = {
            "name": "skip_test",
            "steps": [
                {"id": "a", "name": "A", "type": "function", "function": "m.f", "outputs": ["x"]},
                {"id": "b", "name": "B", "type": "function", "function": "m.f", "outputs": ["y"], "skipped": True},
                {"id": "c", "name": "C", "type": "function", "function": "m.f", "outputs": ["z"]},
            ],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            manager = LocalDirectoryWorkflowManager(workflow_dirs=[temp_dir])
            (Path(temp_dir) / "skip_test.yaml").write_text(yaml.dump(workflow_data))
            manager.refresh_workflows()

            executor = WorkflowExecutor(
                task_executor=Mock(spec=TaskExecutor),
                workflow_manager=manager,
                persistence_provider=LocalCacheProvider(cache_dir=temp_dir + "/cache"),
            )

            with patch.object(WorkflowExecutor, "_create_step_executor") as mock_create:

                def make_step(step):
                    mock = AsyncMock()
                    mock.execute = AsyncMock(
                        return_value=ExecutionResult(step_id=step.id, outputs={step.outputs[0]: f"from_{step.id}"}, success=True)
                    )
                    return mock

                mock_create.side_effect = make_step
                result = await executor.execute_workflow("skip_test")

            assert result.status == "completed"
            assert "a" in result.context.steps
            assert "b" not in result.context.steps  # skipped
            assert "c" in result.context.steps

    @pytest.mark.asyncio
    async def test_inputs_available_to_all_steps(self):
        """Workflow inputs are accessible from every step."""
        workflow_data = {
            "name": "inputs_test",
            "inputs": {"greeting": "Hello"},
            "steps": [
                {"id": "s1", "name": "S1", "type": "function", "function": "m.f", "outputs": ["x"]},
                {"id": "s2", "name": "S2", "type": "function", "function": "m.f", "outputs": ["y"]},
            ],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            manager = LocalDirectoryWorkflowManager(workflow_dirs=[temp_dir])
            (Path(temp_dir) / "inputs_test.yaml").write_text(yaml.dump(workflow_data))
            manager.refresh_workflows()

            executor = WorkflowExecutor(
                task_executor=Mock(spec=TaskExecutor),
                workflow_manager=manager,
                persistence_provider=LocalCacheProvider(cache_dir=temp_dir + "/cache"),
            )

            received_inputs = []

            with patch.object(WorkflowExecutor, "_create_step_executor") as mock_create:

                def make_step(step):
                    mock = AsyncMock()

                    async def execute(ctx):
                        received_inputs.append(dict(ctx.inputs))
                        return ExecutionResult(step_id=step.id, outputs={"x": "ok"}, success=True)

                    mock.execute = execute
                    return mock

                mock_create.side_effect = make_step
                await executor.execute_workflow("inputs_test", inputs={"greeting": "Hi"})

            # Both steps should see the same inputs
            assert received_inputs[0]["greeting"] == "Hi"
            assert received_inputs[1]["greeting"] == "Hi"

    @pytest.mark.asyncio
    async def test_multiple_outputs_per_step(self):
        """Steps can produce multiple named outputs."""
        workflow_data = {
            "name": "multi_output",
            "steps": [
                {"id": "s1", "name": "S1", "type": "function", "function": "m.f", "outputs": ["a", "b"]},
                {"id": "s2", "name": "S2", "type": "function", "function": "m.f", "outputs": ["c"]},
            ],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            manager = LocalDirectoryWorkflowManager(workflow_dirs=[temp_dir])
            (Path(temp_dir) / "multi_output.yaml").write_text(yaml.dump(workflow_data))
            manager.refresh_workflows()

            executor = WorkflowExecutor(
                task_executor=Mock(spec=TaskExecutor),
                workflow_manager=manager,
                persistence_provider=LocalCacheProvider(cache_dir=temp_dir + "/cache"),
            )

            with patch.object(WorkflowExecutor, "_create_step_executor") as mock_create:

                def make_step(step):
                    mock = AsyncMock()
                    if step.id == "s1":
                        mock.execute = AsyncMock(
                            return_value=ExecutionResult(step_id="s1", outputs={"a": "val_a", "b": "val_b"}, success=True)
                        )
                    else:
                        mock.execute = AsyncMock(
                            return_value=ExecutionResult(step_id="s2", outputs={"c": "val_c"}, success=True)
                        )
                    return mock

                mock_create.side_effect = make_step
                result = await executor.execute_workflow("multi_output")

            assert result.context.steps["s1"]["outputs"]["a"] == "val_a"
            assert result.context.steps["s1"]["outputs"]["b"] == "val_b"
            assert result.context.steps["s2"]["outputs"]["c"] == "val_c"


# ---------------------------------------------------------------------------
# Persistence: results survive serialization (resume support)
# ---------------------------------------------------------------------------


class TestResultPersistence:
    @pytest.mark.asyncio
    async def test_outputs_survive_serialization(self):
        """Step outputs stored in WorkflowExecutionState survive JSON round-trip."""
        from cliver.workflow.workflow_models import WorkflowExecutionState

        state = WorkflowExecutionState(
            workflow_name="test",
            execution_id="exec-1",
            status="completed",
            completed_steps=["s1", "s2"],
            context=ExecutionContext(
                workflow_name="test",
                execution_id="exec-1",
                inputs={"topic": "AI"},
                steps={
                    "s1": {"outputs": {"data": "result_1"}},
                    "s2": {"outputs": {"analysis": "result_2", "nested": {"deep": True}}},
                },
            ),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            cache = LocalCacheProvider(cache_dir=temp_dir)
            cache.save_execution_state(state)
            loaded = cache.load_execution_state("test", "exec-1")

            assert loaded.context.steps["s1"]["outputs"]["data"] == "result_1"
            assert loaded.context.steps["s2"]["outputs"]["analysis"] == "result_2"
            assert loaded.context.steps["s2"]["outputs"]["nested"]["deep"] is True
            assert loaded.context.inputs["topic"] == "AI"
