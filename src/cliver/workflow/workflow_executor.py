"""
Workflow Executor for Cliver Workflow Engine.

This module provides a high-level interface for workflow execution operations.
"""

import logging
import threading
import time
import uuid
from typing import Dict, Any, Optional

from cliver.workflow.persistence import LocalCacheProvider, PersistenceProvider
from cliver.workflow.workflow_manager_base import WorkflowManager
from cliver.workflow.workflow_models import (
    ExecutionContext, ExecutionResult, WorkflowExecutionState, StepType, OnErrorAction
)
from cliver.workflow.steps.base import StepExecutor
from cliver.workflow.steps.function_step import FunctionStepExecutor
from cliver.workflow.steps.llm_step import LLMStepExecutor
from cliver.workflow.steps.workflow_step import WorkflowStepExecutor
from cliver.workflow.steps.human_step import HumanStepExecutor
from cliver.llm import TaskExecutor

logger = logging.getLogger(__name__)

class WorkflowExecutor:
    """
    Thread-safe WorkflowExecutor is the main entry point for workflow execution locally.
    """

    def __init__(
        self,
        task_executor: TaskExecutor,
        workflow_manager: WorkflowManager,
        persistence_provider: Optional[PersistenceProvider] = None
    ):
        """
        Initialize the thread-safe WorkflowExecutor.

        Args:
            task_executor: TaskExecutor for LLM interactions for LLM steps
            workflow_manager: WorkflowManager for loading and executing workflows
            persistence_provider: Provider for caching execution results
        """
        self.task_executor = task_executor
        self.workflow_manager = workflow_manager
        self.persistence_provider = persistence_provider or LocalCacheProvider()
        # Add lock for operations that modify shared state
        self._execution_lock = threading.RLock()

    async def execute_workflow(
        self,
        workflow_name: str,
        inputs: Optional[Dict[str, Any]] = None,
        execution_id: Optional[str] = None
    ) -> Optional[WorkflowExecutionState]:
        """
        Thread-safe execute a workflow.

        If the execution_id is found and in a resumable state (paused), the execution state will be resumed.
        If the execution_id is found but cancelled, a new execution will be started.
        In the running loop, if the step is marked as skipped, the execution will continue without executing the step.

        Args:
            workflow_name: Workflow name
            inputs: Input variables for the workflow
            execution_id: Unique execution identifier (generated if not provided)

        Returns:
            Workflow execution state with final outputs and execution status
        """
        # Generate execution ID if not provided (thread-safe)
        if not execution_id:
            execution_id = str(uuid.uuid4())

        # Load workflow (thread-safe through workflow manager)
        workflow = self.workflow_manager.load_workflow(workflow_name)
        if not workflow:
            # no workflow found
            return None

        # Use lock only for execution state management to prevent concurrent execution of same execution_id
        with self._execution_lock:
            # Create execution context
            context = ExecutionContext(
                workflow_name=workflow.name,
                inputs=inputs or {},
                variables=inputs or {},
                execution_id=execution_id,
                current_step=None,
                outputs={},
            )

            # Try to load existing execution state
            execution_state = self.persistence_provider.load_execution_state(workflow.name, execution_id)
            if execution_state:
                # Check if already running (prevent concurrent execution of same execution_id)
                if execution_state.status == "running":
                    return execution_state
                # Only resume if the execution was paused, not if it was cancelled
                if execution_state.status == "paused":
                    # Resume from existing state including its inputs and outputs
                    context = execution_state.context
                    start_index = execution_state.current_step_index
                    logger.info(f"Resuming workflow {workflow.name} from step index {start_index}")

                    # Load step results for completed steps to ensure context is properly populated
                    for completed_step_id in execution_state.completed_steps:
                        step_result = self.persistence_provider.load_step_result(
                            workflow.name,
                            execution_id,
                            completed_step_id
                        )
                        if step_result and step_result.outputs:
                            context.outputs.update(step_result.outputs)
                            context.step_outputs[completed_step_id] = step_result.outputs
                            context.variables.update(step_result.outputs)
                            # Also add step outputs under the step ID for nested access
                            context.variables[completed_step_id] = {"outputs": step_result.outputs}

                    # Update status to running
                    execution_state.status = "running"
                    self.persistence_provider.save_execution_state(execution_state)
                else:
                    # For cancelled or completed executions, start fresh
                    start_index = 0
                    execution_state = WorkflowExecutionState(
                        execution_id=execution_id,
                        workflow_name=workflow.name,
                        current_step_index=0,
                        context=context,
                        completed_steps=[],
                        status="running",
                        error=None
                    )
                    # Save initial state
                    self.persistence_provider.save_execution_state(execution_state)
            else:
                # Start fresh execution
                start_index = 0
                execution_state = WorkflowExecutionState(
                    execution_id=execution_id,
                    workflow_name=workflow.name,
                    current_step_index=0,
                    context=context,
                    completed_steps=[],
                    status="running",
                    error=None
                )
                # Save initial state
                self.persistence_provider.save_execution_state(execution_state)

        # Release lock before starting the actual execution to allow concurrent workflows
        start_time = time.time()
        # Execute steps (outside of lock for better concurrency)
        try:
            for i in range(start_index, len(workflow.steps)):
                step = workflow.steps[i]
                if step.skipped:
                    logger.info(f"Skipping step {step.name}")
                    continue

                # Update current step in context (thread-safe as each execution has its own context)
                context.current_step = step.id
                execution_state.current_step_index = i
                execution_state.context = context

                # Save state before executing step
                with self._execution_lock:
                    self.persistence_provider.save_execution_state(execution_state)

                # Set workflow_name and execution_id on the step for context
                step.workflow_name = workflow.name
                step.execution_id = execution_id

                # Create step executor
                step_executor = self._create_step_executor(step)

                # Evaluate condition if the condition is defined.
                if not step_executor.evaluate_condition(context):
                    logger.info(f"Skipping step {step.id} due to condition doesn't satisfy.")
                    continue

                # Execute step with retry logic
                step_result = await step_executor.execute_with_retry(context)

                # Handle step result
                if step_result.success:
                    # Update context with step outputs
                    context.outputs.update(step_result.outputs)
                    context.step_outputs[step.id] = step_result.outputs
                    context.variables.update(step_result.outputs)
                    # Also add step outputs under the step ID for nested access
                    context.variables[step.id] = {"outputs": step_result.outputs}
                    execution_state.completed_steps.append(step.id)
                    logger.info(f"Step {step.id} completed successfully")

                    # Save step result to cache
                    # TODO: check it later that do we need it anymore as the outputs have been stored in the context!?
                    self.persistence_provider.save_step_result(
                        workflow.name,
                        execution_id,
                        step.id,
                        step_result
                    )

                    # Update execution state
                    execution_state.context = context
                    with self._execution_lock:
                        self.persistence_provider.save_execution_state(execution_state)
                else:
                    # Handle step failure based on on_error policy
                    with self._execution_lock:  # Re-acquire lock for state modification
                        execution_state.status = "failed"
                        execution_state.error = step_result.error
                        self.persistence_provider.save_execution_state(execution_state)
                    on_error_action = step.on_error or OnErrorAction.FAIL
                    if on_error_action == OnErrorAction.FAIL:
                        return execution_state
                    elif on_error_action == OnErrorAction.CONTINUE:
                        logger.warning(f"Step {step.id} failed but continuing: {step_result.error}")
                        continue

            # all steps finished, the workflow completed successfully
            with self._execution_lock:  # Re-acquire lock for final state update
                execution_state.status = "completed"
                execution_time = time.time() - start_time
                execution_state.execution_time = execution_time
                self.persistence_provider.save_execution_state(execution_state)
            # Returns the result of the whole workflow execution
            return execution_state

        except Exception as e:
            with self._execution_lock:  # Re-acquire lock for error state update
                logger.error(f"Workflow execution failed: {str(e)}")
                execution_time = time.time() - start_time
                execution_state.execution_time = execution_time
                execution_state.status = "failed"
                execution_state.error = str(e)
                self.persistence_provider.save_execution_state(execution_state)
            return execution_state

    def _create_step_executor(self, step) -> StepExecutor:
        """Create an appropriate step executor for the given step.

        Args:
            step: Step to create executor for

        Returns:
            StepExecutor instance
        """
        if step.type == StepType.FUNCTION:
            return FunctionStepExecutor(step)
        elif step.type == StepType.LLM:
            # Get cache directory for this step
            cache_dir = None
            # TODO: maybe a better way to support other caching the step results like to some store services.
            if (self.persistence_provider and
                hasattr(self.persistence_provider, 'get_execution_cache_dir') and
                hasattr(step, 'workflow_name') and hasattr(step, 'execution_id') and
                step.workflow_name and step.execution_id):
                cache_dir = self.persistence_provider.get_execution_cache_dir(
                    step.workflow_name, step.execution_id
                )
            return LLMStepExecutor(step, self.task_executor, cache_dir)
        elif step.type == StepType.WORKFLOW:
            return WorkflowStepExecutor(step, self)
        elif step.type == StepType.HUMAN:
            return HumanStepExecutor(step)
        else:
            raise ValueError(f"Unknown step type: {step.type}")

    async def pause_execution(self, workflow_name: str, execution_id: str) -> bool:
        """
        Thread-safe pause a workflow execution.

        Args:
            workflow_name: Name of the workflow
            execution_id: Execution identifier

        Returns:
            True if paused successfully, False otherwise
        """
        if not workflow_name or len(workflow_name.strip()) == 0:
            return False
        if not execution_id or len(execution_id.strip()) == 0:
            return False
        with self._execution_lock:
            state = self.persistence_provider.load_execution_state(workflow_name, execution_id)
            if state:
                state.status = "paused"
                return self.persistence_provider.save_execution_state(state)
            return False

    async def resume_execution(self, workflow_name: str, execution_id: str) -> Optional[WorkflowExecutionState]:
        """
        Thread-safe resume a paused workflow execution.

        This method will only resume executions that are in the "paused" state.
        If the execution is in any other state (running, cancelled, completed, failed),
        it will return None and not attempt to resume.

        Args:
            workflow_name: Name of the workflow
            execution_id: Execution identifier

        Returns:
            Execution result if resumed successfully, None otherwise
        """
        if not workflow_name or len(workflow_name.strip()) == 0:
            return None
        if not execution_id or len(execution_id.strip()) == 0:
            return None
        # Load the state first without lock to check if it exists
        # Use lock only for state update
        with self._execution_lock:
            state = self.persistence_provider.load_execution_state(workflow_name, execution_id)
            if not state or state.status != "paused":
                # we only care the paused status
                return None

        # Resume execution using the stored workflow name (outside of lock for better concurrency)
        # it will update the state status
        return await self.execute_workflow(
            workflow_name=workflow_name,
            inputs=state.context.inputs,
            execution_id=execution_id
        )

    def get_execution_state(self, workflow_name: str, execution_id: str) -> Optional[WorkflowExecutionState]:
        """
        Thread-safe get the current state of a workflow execution.

        Args:
            workflow_name: Name of the workflow
            execution_id: Execution identifier

        Returns:
            Current execution state or None if not found
        """
        if not workflow_name or len(workflow_name.strip()) == 0:
            return None
        if not execution_id or len(execution_id.strip()) == 0:
            return None
        return self.persistence_provider.load_execution_state(workflow_name, execution_id)

    def cancel_execution(self, workflow_name: str, execution_id: str) -> bool:
        """
        Thread-safe cancel a workflow execution.

        Args:
            workflow_name: Name of the workflow
            execution_id: Execution identifier

        Returns:
            True if cancelled successfully, False otherwise
        """
        if not workflow_name or len(workflow_name.strip()) == 0:
            return False
        if not execution_id or len(execution_id.strip()) == 0:
            return False
        with self._execution_lock:
            state = self.persistence_provider.load_execution_state(workflow_name, execution_id)
            if state:
                state.status = "cancelled"
                return self.persistence_provider.save_execution_state(state)
            return False

    def list_executions(self, workflow_name: str) -> Dict[str, Dict[str, Any]]:
        """
        List all cached workflow executions.

        Args:
            workflow_name: Name of the workflow

        Returns:
            Dictionary of execution metadata
        """
        return self.persistence_provider.list_executions(workflow_name)

    def clear_all_executions(self, workflow_name: str) -> int:
        """Thread-safe clear all cached workflow executions.

        Args:
            workflow_name: Name of the workflow

        Returns:
            Number of executions cleared
        """
        with self._execution_lock:
            return self.persistence_provider.clear_all_executions(workflow_name)

    def remove_workflow_execution(self, workflow_name: str, execution_id: str) -> bool:
        """Thread-safe remove a workflow execution state.

        Args:
            workflow_name: Name of the workflow
            execution_id: ID of execution to remove

        Returns:
            True if removed, False if not found
        """
        if not workflow_name or len(workflow_name.strip()) == 0:
            return False
        if not execution_id or len(execution_id.strip()) == 0:
            return False
        with self._execution_lock:
            return self.persistence_provider.remove_execution_state(workflow_name, execution_id)