"""
Workflow step implementation.
"""
import logging
import time
from typing import Any, TYPE_CHECKING
from cliver.workflow.steps.base import StepExecutor
from cliver.workflow.workflow_models import WorkflowStep, ExecutionContext, ExecutionResult

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
        """Execute a sub-workflow.

        Args:
            context: Execution context containing inputs

        Returns:
            ExecutionResult with outputs
        """
        start_time = time.time()
        try:
            # Prepare inputs for the sub-workflow
            workflow_inputs = {}

            # If specific workflow inputs are defined, use them
            if self.step.workflow_inputs:
                for key, value in self.step.workflow_inputs.items():
                    # Resolve variable references in inputs
                    resolved_value = self.resolve_variable(value, context)
                    workflow_inputs[key] = resolved_value
            else:
                # If no specific inputs defined, pass all context inputs and step outputs
                workflow_inputs.update(context.inputs)
                # Add outputs from all completed steps
                for step_info in context.steps.values():
                    if step_info.outputs:
                        workflow_inputs.update(step_info.outputs)

            # Execute the sub-workflow
            execution_result = await self.workflow_executor.execute_workflow(
                workflow_name=self.step.workflow,
                inputs=workflow_inputs
            )

            # Prepare outputs from sub-workflow results
            outputs = {}
            if hasattr(execution_result, 'context') and execution_result.context:
                # Add outputs from all completed steps in the sub-workflow
                for step_info in execution_result.context.steps.values():
                    if step_info.outputs:
                        outputs.update(step_info.outputs)

            # Extract specific outputs if defined in the step
            if self.step.outputs:
                final_outputs = {}
                if len(self.step.outputs) == 1:
                    # Single output - if the outputs dict has the same key, use that value directly
                    # Otherwise, assign the entire outputs dict to that output name
                    output_name = self.step.outputs[0]
                    if output_name in outputs:
                        final_outputs[output_name] = outputs[output_name]
                    else:
                        final_outputs[output_name] = outputs
                else:
                    # Multiple outputs - extract specific keys from the outputs dict
                    for output_name in self.step.outputs:
                        if output_name in outputs:
                            final_outputs[output_name] = outputs[output_name]
            else:
                # No specific outputs defined - pass through all outputs
                final_outputs = outputs

            return ExecutionResult(
                step_id=self.step.id,
                outputs=final_outputs,
                success=execution_result.success if execution_result else False,
                error=execution_result.error if execution_result else "No execution result",
                execution_time=execution_result.execution_time if execution_result else 0.0,
            )

        except Exception as e:
            logger.error(f"Error executing workflow step {self.step.id}: {str(e)}")
            execution_time = time.time() - start_time
            return ExecutionResult(
                step_id=self.step.id,
                success=False,
                error=str(e),
                execution_time=execution_time
            )
