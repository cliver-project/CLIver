"""
Callback handler interface for workflow execution notifications.
"""

from abc import ABC, abstractmethod
from typing import Optional

from cliver.workflow.workflow_models import ExecutionResult


class WorkflowCallbackHandler(ABC):
    """Abstract base class for workflow callback handlers."""

    @abstractmethod
    async def on_step_start(self, step_id: str, step_name: str, step_type: str) -> None:
        """Called when a step is about to start execution.

        Args:
            step_id: Unique identifier of the step
            step_name: Descriptive name of the step
            step_type: Type of the step (function, llm, workflow, human)
        """
        pass

    @abstractmethod
    async def on_step_complete(self, step_id: str, step_name: str, result: ExecutionResult) -> None:
        """Called when a step completes execution (success or failure).

        Args:
            step_id: Unique identifier of the step
            step_name: Descriptive name of the step
            result: Execution result containing outputs, success status, and error if any
        """
        pass

    @abstractmethod
    async def on_workflow_start(self, workflow_name: str, execution_id: str) -> None:
        """Called when a workflow starts execution.

        Args:
            workflow_name: Name of the workflow
            execution_id: Unique execution identifier
        """
        pass

    @abstractmethod
    async def on_workflow_complete(
        self,
        workflow_name: str,
        execution_id: str,
        status: str,
        error: Optional[str] = None,
    ) -> None:
        """Called when a workflow completes execution.

        Args:
            workflow_name: Name of the workflow
            execution_id: Unique execution identifier
            status: Final status of the workflow (completed, failed, paused, cancelled)
            error: Error message if workflow failed
        """
        pass
