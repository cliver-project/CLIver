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
                # If no specific inputs defined, pass all context variables
                workflow_inputs.update(context.inputs)
                workflow_inputs.update(context.outputs)
                workflow_inputs.update(context.variables)

            # Execute the sub-workflow
            execution_result = await self.workflow_executor.execute_workflow(
                workflow_name=self.step.workflow,
                inputs=workflow_inputs
            )

            # Prepare outputs from sub-workflow results
            outputs = {}
            if self.step.outputs:
                # Map specific outputs if defined
                for output_name in self.step.outputs:
                    if output_name in execution_result.outputs:
                        outputs[output_name] = execution_result.outputs[output_name]
                    else:
                        logger.warning(
                            f"Output {output_name} not found in sub-workflow result for step {self.step.id}"
                        )
            else:
                # Pass through all sub-workflow outputs
                outputs.update(execution_result.outputs)

            return ExecutionResult(
                step_id=self.step.id,
                outputs=outputs,
                success=execution_result.success,
                error=execution_result.error,
                execution_time=execution_result.execution_time,
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
