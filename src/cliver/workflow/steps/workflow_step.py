"""
Workflow step implementation — executes a nested sub-workflow.
"""

import logging
import time
from typing import TYPE_CHECKING

from cliver.workflow.steps.base import StepExecutor
from cliver.workflow.workflow_models import ExecutionContext, ExecutionResult, WorkflowStep

if TYPE_CHECKING:
    from cliver.workflow.workflow_executor import WorkflowExecutor

logger = logging.getLogger(__name__)


class WorkflowStepExecutor(StepExecutor):
    """Executor for workflow steps."""

    def __init__(self, step: WorkflowStep, workflow_executor: "WorkflowExecutor"):
        super().__init__(step)
        self.step = step
        self.workflow_executor = workflow_executor

    async def execute(self, context: ExecutionContext) -> ExecutionResult:
        start_time = time.time()
        try:
            # Prepare inputs for the sub-workflow
            if self.step.workflow_inputs:
                workflow_inputs = {
                    k: self.resolve_variable(v, context) for k, v in self.step.workflow_inputs.items()
                }
            else:
                # Pass all context inputs + completed step outputs
                workflow_inputs = dict(context.inputs)
                for step_data in context.steps.values():
                    outputs = step_data.get("outputs", {})
                    if outputs:
                        workflow_inputs.update(outputs)

            # Execute sub-workflow
            result = await self.workflow_executor.execute_workflow(
                workflow_name=self.step.workflow, inputs=workflow_inputs
            )

            # Collect outputs from sub-workflow steps
            outputs = {}
            if result and result.context:
                for step_data in result.context.steps.values():
                    step_outputs = step_data.get("outputs", {})
                    if step_outputs:
                        outputs.update(step_outputs)

            # Extract specific outputs if defined
            if self.step.outputs:
                final_outputs = {}
                for name in self.step.outputs:
                    if name in outputs:
                        final_outputs[name] = outputs[name]
                    elif len(self.step.outputs) == 1:
                        # Single output — assign the whole dict
                        final_outputs[name] = outputs
            else:
                final_outputs = outputs

            return ExecutionResult(
                step_id=self.step.id,
                outputs=final_outputs,
                success=result.status == "completed" if result else False,
                error=result.error if result else "No execution result",
                execution_time=time.time() - start_time,
            )

        except Exception as e:
            logger.error(f"Error executing workflow step {self.step.id}: {e}")
            return ExecutionResult(
                step_id=self.step.id,
                success=False,
                error=str(e),
                execution_time=time.time() - start_time,
            )
