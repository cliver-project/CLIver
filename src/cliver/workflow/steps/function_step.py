"""
Function step implementation.
"""
import asyncio
import importlib
import logging
import time

from cliver.workflow.steps.base import StepExecutor
from cliver.workflow.workflow_models import FunctionStep, ExecutionContext, ExecutionResult

logger = logging.getLogger(__name__)


def _parse_function(function: str) -> tuple:
    """
    Parse a function path into module path and function name.

    Args:
        function: String like "module.submodule.function_name"

    Returns:
        Tuple of (module_path, function_name)
    """
    if '.' not in function:
        raise ValueError(f"Invalid function path: {function}")

    parts = function.split('.')
    function_name = parts[-1]
    module_path = '.'.join(parts[:-1])

    return module_path, function_name


class FunctionStepExecutor(StepExecutor):
    """Executor for function steps."""

    def __init__(self, step: FunctionStep):
        super().__init__(step)
        self.step = step

    async def execute(self, context: ExecutionContext) -> ExecutionResult:
        """Execute a Python function.

        Args:
            context: Execution context containing inputs

        Returns:
            ExecutionResult with outputs
        """
        start_time = time.time()
        try:
            # Parse the function path
            module_path, function_name = _parse_function(self.step.function)
            # Import the module
            module = importlib.import_module(module_path)

            # Get the function, the getattr will raise exception if the function_name does not exist.
            func = getattr(module, function_name)

            # Prepare arguments from context
            args = {}
            if self.step.inputs:
                for arg_name, arg_value in self.step.inputs.items():
                    # Resolve variable references in inputs
                    resolved_value = self.resolve_variable(arg_value, context)
                    args[arg_name] = resolved_value

            # Execute the function
            if asyncio.iscoroutinefunction(func):
                result = await func(**args)
            else:
                result = func(**args)

            # Prepare outputs
            outputs = await self.extract_outputs(result)

            execution_time = time.time() - start_time
            return ExecutionResult(
                step_id=self.step.id,
                outputs=outputs,
                success=True,
                error=None,
                execution_time=execution_time
            )

        except Exception as e:
            logger.error(f"Error executing function step {self.step.id}: {str(e)}")
            execution_time = time.time() - start_time
            return ExecutionResult(
                step_id=self.step.id,
                success=False,
                error=str(e),
                execution_time=execution_time
            )
