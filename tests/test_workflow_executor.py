"""Tests for workflow DAG executor."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from cliver.workflow.workflow_executor import WorkflowExecutor
from cliver.workflow.workflow_models import (
    Branch,
    DecisionStep,
    ExecutionContext,
    LLMStep,
    Workflow,
    WorkflowExecutionState,
)
from cliver.workflow.persistence import WorkflowStore


@pytest.fixture
def mock_task_executor():
    te = MagicMock()
    te.permission_manager = None
    return te


@pytest.fixture
def store(tmp_path):
    return WorkflowStore(tmp_path / "workflows")


class TestDAGExecution:
    def test_linear_workflow(self, mock_task_executor, store):
        """Steps without depends_on execute in order."""
        wf = Workflow(
            name="linear",
            steps=[
                LLMStep(id="s1", name="Step 1", prompt="first"),
                LLMStep(id="s2", name="Step 2", prompt="second", depends_on=["s1"]),
            ],
        )
        store.save_workflow(wf)

        call_order = []

        async def mock_process(user_input, **kwargs):
            call_order.append(user_input)
            msg = MagicMock()
            msg.content = f"result of {user_input}"
            return msg

        mock_task_executor.process_user_input = AsyncMock(side_effect=mock_process)

        executor = WorkflowExecutor(mock_task_executor, store)
        state = asyncio.run(executor.execute_workflow("linear"))

        assert state.status == "completed"
        assert len(state.completed_steps) == 2
        assert call_order == ["first", "second"]

    def test_parallel_steps(self, mock_task_executor, store):
        """Steps with same dependency run concurrently."""
        wf = Workflow(
            name="parallel",
            steps=[
                LLMStep(id="root", name="Root", prompt="start"),
                LLMStep(id="a", name="A", prompt="branch a", depends_on=["root"]),
                LLMStep(id="b", name="B", prompt="branch b", depends_on=["root"]),
                LLMStep(id="join", name="Join", prompt="done", depends_on=["a", "b"]),
            ],
        )
        store.save_workflow(wf)

        async def mock_process(user_input, **kwargs):
            msg = MagicMock()
            msg.content = f"done: {user_input}"
            return msg

        mock_task_executor.process_user_input = AsyncMock(side_effect=mock_process)

        executor = WorkflowExecutor(mock_task_executor, store)
        state = asyncio.run(executor.execute_workflow("parallel"))

        assert state.status == "completed"
        assert set(state.completed_steps) == {"root", "a", "b", "join"}


class TestDecisionBranching:
    def test_decision_selects_branch(self, mock_task_executor, store):
        """DecisionStep should route to the correct branch."""
        wf = Workflow(
            name="branching",
            steps=[
                LLMStep(id="check", name="Check", prompt="check status", outputs=["status"]),
                DecisionStep(
                    id="decide",
                    name="Decide",
                    depends_on=["check"],
                    branches=[
                        Branch(condition="'PASS' in check.outputs.status", next_step="deploy"),
                        Branch(condition="'FAIL' in check.outputs.status", next_step="fix"),
                    ],
                    default="fix",
                ),
                LLMStep(id="deploy", name="Deploy", prompt="deploy now", depends_on=["decide"]),
                LLMStep(id="fix", name="Fix", prompt="fix issues", depends_on=["decide"]),
            ],
        )
        store.save_workflow(wf)

        call_order = []

        async def mock_process(user_input, **kwargs):
            call_order.append(user_input)
            msg = MagicMock()
            if "check" in user_input:
                msg.content = "status: PASS"
            else:
                msg.content = f"done: {user_input}"
            return msg

        mock_task_executor.process_user_input = AsyncMock(side_effect=mock_process)

        executor = WorkflowExecutor(mock_task_executor, store)
        state = asyncio.run(executor.execute_workflow("branching"))

        assert state.status == "completed"
        assert "deploy" in state.completed_steps
        assert "fix" in state.skipped_steps

    def test_decision_default_branch(self, mock_task_executor, store):
        """If no branch matches, use default."""
        wf = Workflow(
            name="default-branch",
            steps=[
                LLMStep(id="check", name="Check", prompt="check", outputs=["status"]),
                DecisionStep(
                    id="decide",
                    name="Decide",
                    depends_on=["check"],
                    branches=[
                        Branch(condition="'SPECIAL' in check.outputs.status", next_step="special"),
                    ],
                    default="normal",
                ),
                LLMStep(id="special", name="Special", prompt="special path", depends_on=["decide"]),
                LLMStep(id="normal", name="Normal", prompt="normal path", depends_on=["decide"]),
            ],
        )
        store.save_workflow(wf)

        async def mock_process(user_input, **kwargs):
            msg = MagicMock()
            msg.content = "status: REGULAR"
            return msg

        mock_task_executor.process_user_input = AsyncMock(side_effect=mock_process)

        executor = WorkflowExecutor(mock_task_executor, store)
        state = asyncio.run(executor.execute_workflow("default-branch"))

        # Should fall to default
        assert "normal" in state.completed_steps
        assert "special" in state.skipped_steps
        assert state.status == "completed"


class TestConditionGating:
    def test_condition_skips_step(self, mock_task_executor, store):
        """Step with false condition should be skipped."""
        wf = Workflow(
            name="conditional",
            steps=[
                LLMStep(id="s1", name="Step 1", prompt="first", outputs=["result"]),
                LLMStep(
                    id="s2",
                    name="Step 2",
                    prompt="conditional",
                    depends_on=["s1"],
                    condition="'yes' in s1.outputs.result",
                ),
            ],
        )
        store.save_workflow(wf)

        async def mock_process(user_input, **kwargs):
            msg = MagicMock()
            msg.content = "no match here"
            return msg

        mock_task_executor.process_user_input = AsyncMock(side_effect=mock_process)

        executor = WorkflowExecutor(mock_task_executor, store)
        state = asyncio.run(executor.execute_workflow("conditional"))

        assert state.status == "completed"
        assert "s2" in state.skipped_steps


class TestRetry:
    def test_step_retries_on_failure(self, mock_task_executor, store):
        """Step with retry should retry on failure."""
        wf = Workflow(
            name="retry-wf",
            steps=[
                LLMStep(id="s1", name="Flaky", prompt="flaky", retry=2),
            ],
        )
        store.save_workflow(wf)

        call_count = 0

        async def mock_process(user_input, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("temporary failure")
            msg = MagicMock()
            msg.content = "success"
            return msg

        mock_task_executor.process_user_input = AsyncMock(side_effect=mock_process)

        executor = WorkflowExecutor(mock_task_executor, store)
        state = asyncio.run(executor.execute_workflow("retry-wf"))

        assert state.status == "completed"
        assert call_count == 3


class TestPauseResume:
    def test_resume_skips_completed(self, mock_task_executor, store):
        """Resuming should skip already-completed steps."""
        wf = Workflow(
            name="resume-wf",
            steps=[
                LLMStep(id="s1", name="Step 1", prompt="first"),
                LLMStep(id="s2", name="Step 2", prompt="second", depends_on=["s1"]),
            ],
        )
        store.save_workflow(wf)

        # Pre-save a paused state with s1 completed
        state = WorkflowExecutionState(
            workflow_name="resume-wf",
            execution_id="exec1",
            completed_steps=["s1"],
            status="paused",
            context=ExecutionContext(
                workflow_name="resume-wf",
                execution_id="exec1",
                steps={"s1": {"outputs": {"result": "done"}, "status": "completed"}},
            ),
        )
        store.save_state(state)

        call_order = []

        async def mock_process(user_input, **kwargs):
            call_order.append(user_input)
            msg = MagicMock()
            msg.content = f"done: {user_input}"
            return msg

        mock_task_executor.process_user_input = AsyncMock(side_effect=mock_process)

        executor = WorkflowExecutor(mock_task_executor, store)
        result = asyncio.run(executor.resume_workflow("resume-wf"))

        assert result.status == "completed"
        # Only s2 should have been executed (s1 was already done)
        assert call_order == ["second"]
