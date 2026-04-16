"""
Workflow DAG Executor for CLIver.

Replaces the old linear executor with a DAG scheduler that supports:
- Parallel execution of independent steps
- Decision branching (DecisionStep)
- Conditional step gating
- Retry on failure
- Pause/resume with persistent state
- Step-to-step result propagation via ExecutionContext
"""

import asyncio
import logging
import time
import uuid
from typing import Any, Dict, Optional

from cliver.llm import AgentCore
from cliver.workflow.context_renderer import evaluate_condition, render_template
from cliver.workflow.persistence import WorkflowStore
from cliver.workflow.workflow_models import (
    BaseStep,
    DecisionStep,
    ExecutionContext,
    ExecutionResult,
    FunctionStep,
    HumanStep,
    LLMStep,
    Workflow,
    WorkflowExecutionState,
    WorkflowStep,
)

logger = logging.getLogger(__name__)


class WorkflowExecutor:
    """DAG-based workflow executor."""

    def __init__(self, task_executor: AgentCore, store: WorkflowStore):
        self.task_executor = task_executor
        self.store = store

    async def execute_workflow(
        self,
        workflow_name: str,
        inputs: Optional[Dict[str, Any]] = None,
        execution_id: Optional[str] = None,
    ) -> Optional[WorkflowExecutionState]:
        """Execute a workflow by name."""
        workflow = self.store.load_workflow(workflow_name)
        if not workflow:
            logger.error(f"Workflow '{workflow_name}' not found")
            return None

        if not execution_id:
            execution_id = str(uuid.uuid4())[:8]

        context = ExecutionContext(
            workflow_name=workflow_name,
            execution_id=execution_id,
            inputs=workflow.get_initial_inputs(inputs),
        )

        state = WorkflowExecutionState(
            workflow_name=workflow_name,
            execution_id=execution_id,
            context=context,
        )

        return await self._run_dag(workflow, state)

    async def resume_workflow(self, workflow_name: str) -> Optional[WorkflowExecutionState]:
        """Resume a paused workflow from its last saved state."""
        state = self.store.load_state(workflow_name)
        if not state or state.status != "paused":
            logger.error(f"No paused state for workflow '{workflow_name}'")
            return None

        workflow = self.store.load_workflow(workflow_name)
        if not workflow:
            return None

        state.status = "running"
        return await self._run_dag(workflow, state)

    async def _run_dag(self, workflow: Workflow, state: WorkflowExecutionState) -> WorkflowExecutionState:
        """Execute the workflow DAG until all steps are done or an error occurs."""
        start_time = time.time()
        steps_by_id = {step.id: step for step in workflow.steps}

        try:
            while True:
                ready = self._find_ready_steps(steps_by_id, state)
                if not ready:
                    break

                # Execute all ready steps concurrently
                results = await asyncio.gather(
                    *[self._execute_step(step, state) for step in ready],
                    return_exceptions=True,
                )

                for step, result in zip(ready, results, strict=True):
                    if isinstance(result, Exception):
                        state.status = "failed"
                        state.error = f"Step '{step.id}' raised: {result}"
                        state.execution_time = time.time() - start_time
                        self.store.save_state(state)
                        return state

                # Persist after each batch
                self.store.save_state(state)

            # Check if all non-skipped steps completed
            all_step_ids = set(steps_by_id.keys())
            done_ids = set(state.completed_steps) | set(state.skipped_steps)
            if all_step_ids <= done_ids:
                state.status = "completed"
            else:
                # Some steps couldn't run (failed dependencies)
                unreachable = all_step_ids - done_ids
                if state.status != "failed":
                    state.status = "completed"
                    logger.info(f"Steps not reached: {unreachable}")

        except Exception as e:
            state.status = "failed"
            state.error = str(e)

        state.execution_time = time.time() - start_time
        self.store.save_state(state)
        return state

    def _find_ready_steps(self, steps_by_id: Dict[str, BaseStep], state: WorkflowExecutionState) -> list[BaseStep]:
        """Find steps whose dependencies are all satisfied."""
        ready = []
        done = set(state.completed_steps) | set(state.skipped_steps)

        for step_id, step in steps_by_id.items():
            if step_id in done:
                continue
            if step.skipped:
                state.skipped_steps.append(step_id)
                continue
            # Check all dependencies are done
            if all(dep in done for dep in step.depends_on):
                # Check if any dependency failed (not just skipped)
                dep_failed = any(state.context.steps.get(dep, {}).get("status") == "failed" for dep in step.depends_on)
                if dep_failed:
                    state.skipped_steps.append(step_id)
                    continue
                ready.append(step)

        return ready

    async def _execute_step(self, step: BaseStep, state: WorkflowExecutionState) -> None:
        """Execute a single step, handling its type-specific logic."""
        context = state.context
        logger.info(f"Executing step '{step.id}' ({step.type.value})")

        # Check condition gate
        if step.condition and not evaluate_condition(step.condition, context):
            logger.info(f"Step '{step.id}' skipped (condition false)")
            state.skipped_steps.append(step.id)
            return

        # Handle decision steps
        if isinstance(step, DecisionStep):
            await self._execute_decision(step, state)
            return

        # Execute with retry
        last_error = None
        for attempt in range(step.retry + 1):
            try:
                result = await self._execute_step_by_type(step, context)
                if result.success:
                    context.steps[step.id] = {
                        "outputs": result.outputs,
                        "status": "completed",
                        "execution_time": result.execution_time,
                    }
                    state.completed_steps.append(step.id)
                    return
                else:
                    last_error = result.error
                    if attempt < step.retry:
                        logger.info(f"Step '{step.id}' failed (attempt {attempt + 1}/{step.retry + 1}), retrying")
            except Exception as e:
                last_error = str(e)
                if attempt < step.retry:
                    logger.info(f"Step '{step.id}' error (attempt {attempt + 1}/{step.retry + 1}): {e}")

        # All retries exhausted
        context.steps[step.id] = {
            "outputs": {"error": last_error},
            "status": "failed",
        }
        state.status = "failed"
        state.error = f"Step '{step.id}' failed: {last_error}"

    async def _execute_decision(self, step: DecisionStep, state: WorkflowExecutionState) -> None:
        """Evaluate a decision step and route to the selected branch."""
        context = state.context
        chosen_step = None

        for branch in step.branches:
            if evaluate_condition(branch.condition, context):
                chosen_step = branch.next_step
                break

        if chosen_step is None:
            chosen_step = step.default

        if chosen_step is None:
            state.status = "failed"
            state.error = f"Decision step '{step.id}': no branch matched and no default"
            return

        logger.info(f"Decision '{step.id}' chose: {chosen_step}")

        # Mark unchosen branch targets as skipped
        chosen_targets = {chosen_step}
        all_targets = {b.next_step for b in step.branches}
        if step.default:
            all_targets.add(step.default)
        skipped_targets = all_targets - chosen_targets

        for target in skipped_targets:
            if target not in state.completed_steps and target not in state.skipped_steps:
                state.skipped_steps.append(target)

        context.steps[step.id] = {
            "outputs": {"chosen": chosen_step},
            "status": "completed",
        }
        state.completed_steps.append(step.id)

    async def _execute_step_by_type(self, step: BaseStep, context: ExecutionContext) -> ExecutionResult:
        """Execute a step based on its type."""
        start_time = time.time()

        if isinstance(step, LLMStep):
            return await self._execute_llm_step(step, context, start_time)
        elif isinstance(step, FunctionStep):
            return await self._execute_function_step(step, context, start_time)
        elif isinstance(step, HumanStep):
            return await self._execute_human_step(step, context, start_time)
        elif isinstance(step, WorkflowStep):
            return await self._execute_workflow_step(step, context, start_time)
        else:
            return ExecutionResult(
                step_id=step.id,
                success=False,
                error=f"Unknown step type: {step.type}",
                execution_time=time.time() - start_time,
            )

    async def _execute_llm_step(self, step: LLMStep, context: ExecutionContext, start_time: float) -> ExecutionResult:
        """Execute an LLM step via AgentCore."""
        try:
            rendered_prompt = render_template(step.prompt, context)
            rendered_images = render_template(step.images, context) if step.images else None
            rendered_audio = render_template(step.audio_files, context) if step.audio_files else None
            rendered_video = render_template(step.video_files, context) if step.video_files else None
            rendered_files = render_template(step.files, context) if step.files else None

            response = await self.task_executor.process_user_input(
                user_input=rendered_prompt,
                model=step.model,
                images=rendered_images,
                audio_files=rendered_audio,
                video_files=rendered_video,
                files=rendered_files,
            )

            result_text = str(response.content) if response and response.content else ""

            outputs = {"result": result_text}
            if step.outputs:
                for name in step.outputs:
                    outputs[name] = result_text

            return ExecutionResult(
                step_id=step.id,
                outputs=outputs,
                success=True,
                execution_time=time.time() - start_time,
            )
        except Exception as e:
            return ExecutionResult(
                step_id=step.id,
                success=False,
                error=str(e),
                execution_time=time.time() - start_time,
            )

    async def _execute_function_step(
        self, step: FunctionStep, context: ExecutionContext, start_time: float
    ) -> ExecutionResult:
        """Execute a Python function step."""
        try:
            import importlib

            module_path, func_name = step.function.rsplit(".", 1)
            module = importlib.import_module(module_path)
            func = getattr(module, func_name)

            rendered_inputs = render_template(step.inputs, context) if step.inputs else {}
            result = func(**rendered_inputs) if rendered_inputs else func()

            outputs = {"result": result}
            if step.outputs and isinstance(result, dict):
                for name in step.outputs:
                    if name in result:
                        outputs[name] = result[name]

            return ExecutionResult(
                step_id=step.id,
                outputs=outputs,
                success=True,
                execution_time=time.time() - start_time,
            )
        except Exception as e:
            return ExecutionResult(
                step_id=step.id,
                success=False,
                error=str(e),
                execution_time=time.time() - start_time,
            )

    async def _execute_human_step(
        self, step: HumanStep, context: ExecutionContext, start_time: float
    ) -> ExecutionResult:
        """Execute a human input step."""
        rendered_prompt = render_template(step.prompt, context)

        if step.auto_confirm:
            return ExecutionResult(
                step_id=step.id,
                outputs={"result": "auto-confirmed"},
                success=True,
                execution_time=time.time() - start_time,
            )

        from cliver.agent_profile import get_input_fn

        input_fn = get_input_fn()
        user_input = input_fn(f"[Workflow] {rendered_prompt}: ")

        return ExecutionResult(
            step_id=step.id,
            outputs={"result": user_input},
            success=True,
            execution_time=time.time() - start_time,
        )

    async def _execute_workflow_step(
        self, step: WorkflowStep, context: ExecutionContext, start_time: float
    ) -> ExecutionResult:
        """Execute a nested workflow step."""
        try:
            rendered_inputs = render_template(step.workflow_inputs, context) if step.workflow_inputs else None
            sub_state = await self.execute_workflow(
                step.workflow,
                inputs=rendered_inputs,
            )
            if sub_state and sub_state.status == "completed":
                return ExecutionResult(
                    step_id=step.id,
                    outputs={"result": sub_state.context.steps},
                    success=True,
                    execution_time=time.time() - start_time,
                )
            error = sub_state.error if sub_state else "Sub-workflow not found"
            return ExecutionResult(
                step_id=step.id,
                success=False,
                error=error,
                execution_time=time.time() - start_time,
            )
        except Exception as e:
            return ExecutionResult(
                step_id=step.id,
                success=False,
                error=str(e),
                execution_time=time.time() - start_time,
            )
