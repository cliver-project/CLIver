"""
Default callback handler implementation for workflow execution notifications.
"""

from typing import Optional

from cliver.workflow.callback_handler import WorkflowCallbackHandler
from cliver.workflow.workflow_models import ExecutionResult


class DefaultCallbackHandler(WorkflowCallbackHandler):
    """Default no-op implementation of the workflow callback handler."""

    async def on_step_start(self, step_id: str, step_name: str, step_type: str) -> None:
        """Called when a step is about to start execution."""
        pass

    async def on_step_complete(self, step_id: str, step_name: str, result: ExecutionResult) -> None:
        """Called when a step completes execution."""
        pass

    async def on_workflow_start(self, workflow_name: str, execution_id: str) -> None:
        """Called when a workflow starts execution."""
        pass

    async def on_workflow_complete(
        self,
        workflow_name: str,
        execution_id: str,
        status: str,
        error: Optional[str] = None,
    ) -> None:
        """Called when a workflow completes execution."""
        pass
