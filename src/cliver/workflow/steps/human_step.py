"""
Human step implementation.
"""
import logging
import time
from cliver.workflow.steps.base import StepExecutor
from cliver.workflow.workflow_models import HumanStep, ExecutionContext, ExecutionResult

logger = logging.getLogger(__name__)


def _get_user_confirmation() -> bool:
    """Get confirmation from user.

    Returns:
        True if confirmed, False otherwise
    """
    while True:
        try:
            response = input("Continue? (y/n): ").strip().lower()
            if response in ['y', 'yes']:
                return True
            elif response in ['n', 'no']:
                return False
            else:
                print("Please enter 'y' for yes or 'n' for no.")
        except (KeyboardInterrupt, EOFError):
            print("\nOperation cancelled.")
            return False


class HumanStepExecutor(StepExecutor):
    """Executor for human steps."""

    def __init__(self, step: HumanStep):
        super().__init__(step)
        self.step = step

    async def execute(self, context: ExecutionContext) -> ExecutionResult:
        """Wait for human confirmation.

        Args:
            context: Execution context

        Returns:
            ExecutionResult with outputs
        """
        start_time = time.time()
        try:
            if self.step.auto_confirm:
                # Automatically confirm without user input
                logger.info(f"Auto-confirming step {self.step.id}")
                return ExecutionResult(
                    step_id=self.step.id,
                    outputs={"confirmed": True},
                    success=True,
                    error=None,
                    execution_time=0.0
                )

            # Show prompt to user
            resolved_prompt = self.resolve_variable(self.step.prompt, context)
            print(f"\n{resolved_prompt}")

            # Get user confirmation
            confirmed = _get_user_confirmation()

            # Prepare outputs
            outputs = await self.extract_outputs(confirmed)

            execution_time = time.time() - start_time
            return ExecutionResult(
                step_id=self.step.id,
                outputs=outputs,
                success=True,
                error = None,
                execution_time = execution_time
            )

        except Exception as e:
            logger.error(f"Error executing human step {self.step.id}: {str(e)}")
            execution_time = time.time() - start_time
            return ExecutionResult(
                step_id=self.step.id,
                success=False,
                error=str(e),
                execution_time=execution_time
            )
