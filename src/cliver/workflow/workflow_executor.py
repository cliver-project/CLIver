"""
Workflow Executor for CLIver Workflow Engine.

Simplified pure-async executor — no threading locks.
"""

import logging
import time
import uuid
from typing import Any, Dict, Optional

from cliver.llm import TaskExecutor
from cliver.workflow.callback_handler import WorkflowCallbackHandler
from cliver.workflow.default_callback_handler import DefaultCallbackHandler
from cliver.workflow.persistence import LocalCacheProvider, PersistenceProvider
from cliver.workflow.steps.base import StepExecutor
from cliver.workflow.steps.function_step import FunctionStepExecutor
from cliver.workflow.steps.human_step import HumanStepExecutor
from cliver.workflow.steps.llm_step import LLMStepExecutor
from cliver.workflow.steps.workflow_step import WorkflowStepExecutor
from cliver.workflow.workflow_manager_base import WorkflowManager
from cliver.workflow.workflow_models import (
    ExecutionContext,
    StepType,
    WorkflowExecutionState,
)

logger = logging.getLogger(__name__)


class WorkflowExecutor:
    """Execute workflows — the main entry point for workflow execution."""

    def __init__(
        self,
        task_executor: TaskExecutor,
        workflow_manager: WorkflowManager,
        persistence_provider: Optional[PersistenceProvider] = None,
        callback_handler: Optional[WorkflowCallbackHandler] = None,
    ):
        self.task_executor = task_executor
        self.workflow_manager = workflow_manager
        self.persistence_provider = persistence_provider or LocalCacheProvider()
        self.callback_handler = callback_handler or DefaultCallbackHandler()

    async def execute_workflow(
        self,
        workflow_name: str,
        inputs: Optional[Dict[str, Any]] = None,
        execution_id: Optional[str] = None,
    ) -> Optional[WorkflowExecutionState]:
        """Execute a workflow, resuming from a paused state if applicable.

        Args:
            workflow_name: Name of the workflow to execute
            inputs: Input variables for the workflow
            execution_id: Unique execution ID (generated if not provided)

        Returns:
            Final WorkflowExecutionState, or None if workflow not found
        """
        if not execution_id:
            execution_id = str(uuid.uuid4())

        workflow = self.workflow_manager.load_workflow(workflow_name)
        if not workflow:
            return None

        # Check for existing execution state (resume support)
        start_index = 0
        existing_state = self.persistence_provider.load_execution_state(workflow_name, execution_id)

        if existing_state:
            if existing_state.status == "running":
                return existing_state  # already running
            if existing_state.status == "paused":
                # Resume: use saved context (includes all completed step outputs)
                context = existing_state.context
                start_index = existing_state.current_step_index
                logger.info(f"Resuming workflow '{workflow_name}' from step index {start_index}")
            else:
                # Cancelled/completed/failed — start fresh
                context = self._create_context(workflow, inputs, execution_id)
        else:
            context = self._create_context(workflow, inputs, execution_id)

        # Initialize execution state
        state = WorkflowExecutionState(
            execution_id=execution_id,
            workflow_name=workflow.name,
            current_step_index=start_index,
            completed_steps=(
                existing_state.completed_steps if existing_state and existing_state.status == "paused" else []
            ),
            status="running",
            context=context,
        )
        self.persistence_provider.save_execution_state(state)

        # Execute steps
        start_time = time.time()
        await self.callback_handler.on_workflow_start(workflow.name, execution_id)

        try:
            for i in range(start_index, len(workflow.steps)):
                step = workflow.steps[i]
                if step.skipped:
                    logger.info(f"Skipping step '{step.name}'")
                    continue

                # Update and persist current progress
                state.current_step_index = i
                state.context = context
                self.persistence_provider.save_execution_state(state)

                # Create and execute step
                step_executor = self._create_step_executor(step)
                await self.callback_handler.on_step_start(step.id, step.name, step.type.value)

                step_result = await step_executor.execute(context)

                if step_result.success:
                    # Store outputs in context for subsequent steps
                    context.steps[step.id] = {"outputs": step_result.outputs}
                    state.completed_steps.append(step.id)
                    state.context = context
                    self.persistence_provider.save_execution_state(state)
                    await self.callback_handler.on_step_complete(step.id, step.name, step_result)
                    logger.info(f"Step '{step.id}' completed")
                else:
                    # Step failed — stop workflow
                    state.status = "failed"
                    state.error = step_result.error
                    state.execution_time = time.time() - start_time
                    self.persistence_provider.save_execution_state(state)
                    await self.callback_handler.on_step_complete(step.id, step.name, step_result)
                    await self.callback_handler.on_workflow_complete(
                        workflow.name, execution_id, "failed", step_result.error
                    )
                    return state

            # All steps completed
            state.status = "completed"
            state.execution_time = time.time() - start_time
            self.persistence_provider.save_execution_state(state)
            await self.callback_handler.on_workflow_complete(workflow.name, execution_id, "completed")
            return state

        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            state.status = "failed"
            state.error = str(e)
            state.execution_time = time.time() - start_time
            self.persistence_provider.save_execution_state(state)
            await self.callback_handler.on_workflow_complete(workflow.name, execution_id, "failed", str(e))
            return state

    def _create_context(self, workflow, inputs, execution_id) -> ExecutionContext:
        """Create a fresh execution context."""
        return ExecutionContext(
            workflow_name=workflow.name,
            inputs=workflow.get_initial_inputs(inputs),
            execution_id=execution_id,
        )

    def _create_step_executor(self, step) -> StepExecutor:
        """Create the appropriate step executor for a step."""
        if step.type == StepType.FUNCTION:
            return FunctionStepExecutor(step)
        elif step.type == StepType.LLM:
            cache_dir = None
            if hasattr(self.persistence_provider, "get_execution_cache_dir"):
                cache_dir = self.persistence_provider.get_execution_cache_dir(
                    step.workflow_name or "", step.execution_id or ""
                )
            return LLMStepExecutor(step, self.task_executor, cache_dir)
        elif step.type == StepType.WORKFLOW:
            return WorkflowStepExecutor(step, self)
        elif step.type == StepType.HUMAN:
            return HumanStepExecutor(step)
        else:
            raise ValueError(f"Unknown step type: {step.type}")

    async def pause_execution(self, workflow_name: str, execution_id: str) -> bool:
        """Pause a running workflow execution."""
        state = self.persistence_provider.load_execution_state(workflow_name, execution_id)
        if state and state.status == "running":
            state.status = "paused"
            self.persistence_provider.save_execution_state(state)
            await self.callback_handler.on_workflow_complete(workflow_name, execution_id, "paused")
            return True
        return False

    async def resume_execution(self, workflow_name: str, execution_id: str) -> Optional[WorkflowExecutionState]:
        """Resume a paused workflow execution."""
        state = self.persistence_provider.load_execution_state(workflow_name, execution_id)
        if not state or state.status != "paused":
            return None
        return await self.execute_workflow(
            workflow_name=workflow_name,
            inputs=state.context.inputs,
            execution_id=execution_id,
        )

    def get_execution_state(self, workflow_name: str, execution_id: str) -> Optional[WorkflowExecutionState]:
        """Get the current state of a workflow execution."""
        return self.persistence_provider.load_execution_state(workflow_name, execution_id)

    def cancel_execution(self, workflow_name: str, execution_id: str) -> bool:
        """Cancel a workflow execution."""
        state = self.persistence_provider.load_execution_state(workflow_name, execution_id)
        if state:
            state.status = "cancelled"
            return self.persistence_provider.save_execution_state(state)
        return False

    def list_executions(self, workflow_name: str) -> Dict[str, Dict[str, Any]]:
        """List all cached workflow executions."""
        return self.persistence_provider.list_executions(workflow_name)

    def clear_all_executions(self, workflow_name: str) -> int:
        """Clear all cached workflow executions."""
        return self.persistence_provider.clear_all_executions(workflow_name)

    def remove_workflow_execution(self, workflow_name: str, execution_id: str) -> bool:
        """Remove a workflow execution state."""
        return self.persistence_provider.remove_execution_state(workflow_name, execution_id)
