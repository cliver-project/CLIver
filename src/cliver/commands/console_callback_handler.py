"""
Console callback handler implementation for workflow execution notifications.
"""

import logging
from typing import Optional

import click

from cliver.workflow.callback_handler import WorkflowCallbackHandler
from cliver.workflow.workflow_models import ExecutionResult

logger = logging.getLogger(__name__)


class ConsoleCallbackHandler(WorkflowCallbackHandler):
    """Console implementation of the workflow callback handler that prints status
    updates."""

    async def on_step_start(self, step_id: str, step_name: str, step_type: str) -> None:
        """Called when a step is about to start execution."""
        click.echo(f"[WORKFLOW] Starting step: {step_name} ({step_id}) [{step_type}]")

    async def on_step_complete(self, step_id: str, step_name: str, result: ExecutionResult) -> None:
        """Called when a step completes execution."""
        if result.success:
            click.echo(f"[WORKFLOW] Step completed: {step_name} ({step_id}) - Success")
            click.echo(f"Step Outputs: {result.outputs}")
        else:
            click.echo(f"[WORKFLOW] Step failed: {step_name} ({step_id}) - {result.error}")

    async def on_workflow_start(self, workflow_name: str, execution_id: str) -> None:
        """Called when a workflow starts execution."""
        click.echo(f"[WORKFLOW] Starting workflow: {workflow_name} (ID: {execution_id})")

    async def on_workflow_complete(
        self,
        workflow_name: str,
        execution_id: str,
        status: str,
        error: Optional[str] = None,
    ) -> None:
        """Called when a workflow completes execution."""
        if error:
            click.echo(
                f"[WORKFLOW] Workflow {workflow_name} (ID: {execution_id}) completed with "
                f"status: {status} - Error: {error}"
            )
        else:
            click.echo(f"[WORKFLOW] Workflow {workflow_name} (ID: {execution_id}) completed with status: {status}")
