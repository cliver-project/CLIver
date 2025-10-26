"""
Workflow command for Cliver CLI.
"""
import asyncio
import click
from typing import Optional

from cliver.cli import Cliver, pass_cliver
from cliver.workflow.workflow_executor import WorkflowExecutor
from cliver.commands.console_callback_handler import ConsoleCallbackHandler


@click.group(name="workflow", help="Manage and execute workflows")
def workflow():
    """Workflow management commands."""
    pass


@workflow.command(name="run", help="Run a workflow")
@click.argument("workflow_identifier")
@click.option("--inputs", "-i", multiple=True, type=(str, str), help="Input variables (key=value)")
@click.option("--execution-id", "-e", help="Execution ID for resuming")
@click.option("--dry-run", "-d", is_flag=True, help="Show what would be executed without running")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed execution progress")
@pass_cliver
def run_workflow(
    cliver: Cliver,
    workflow_identifier: str,
    inputs: tuple,
    execution_id: Optional[str],
    dry_run: bool,
    verbose: bool
):
    """Run a workflow with the given inputs."""
    try:
        if not workflow_identifier or len(workflow_identifier) == 0:
            click.echo(f"No workflow identifier provided, aborting.")
            return 1
        # Convert input tuples to dictionary
        _inputs = dict(inputs) if inputs else {}

        # Use Cliver's workflow manager and executor
        workflow_manager = cliver.workflow_manager
        # Create workflow executor with console callback handler if verbose is enabled
        if verbose:
            callback_handler = ConsoleCallbackHandler()
            workflow_executor = WorkflowExecutor(
                task_executor=cliver.task_executor,
                workflow_manager=workflow_manager,
                callback_handler=callback_handler
            )
        else:
            workflow_executor = cliver.workflow_executor

        if dry_run:
            # Load and display workflow info
            _workflow = workflow_manager.load_workflow(workflow_identifier)
            if not _workflow:
                click.echo("No workflow found for workflow {}".format(workflow_identifier))
                return 1
            click.echo(f"Workflow: {_workflow.name}")
            if _workflow.description:
                click.echo(f"Description: {_workflow.description}")
            click.echo(f"Steps: {len(_workflow.steps)}")
            for i, step in enumerate(_workflow.steps):
                click.echo(f"  {i+1}. {step.name} ({step.type.value})")
            return 0

        # Execute workflow
        async def _run():
            _result = await workflow_executor.execute_workflow(
                workflow_name=workflow_identifier,
                inputs=_inputs,
                execution_id=execution_id
            )
            return _result

        result = asyncio.run(_run())

        if result.status == "completed":
            click.echo("Workflow completed successfully!")
            if result.context.outputs:
                click.echo("Outputs:")
                for key, value in result.context.outputs.items():
                    click.echo(f"  {key}: {value}")
            return 0
        else:
            click.echo(f"Workflow failed with status {result.status}: {result.error}")
            return 1

    except Exception as e:
        click.echo(f"Error running workflow: {e}")
        return 1


@workflow.command(name="list", help="List available workflows")
@pass_cliver
def list_workflows(cliver: Cliver):
    """List available workflows."""
    try:
        # Use Cliver's workflow manager
        workflow_manager = cliver.workflow_manager

        # List workflows
        workflows = workflow_manager.list_workflows()

        if not workflows:
            click.echo("No workflows found.")
            return 0

        click.echo("Available workflows:")
        for name, _workflow in workflows.items():
            click.echo(f"  {name}: {_workflow.description}")

    except Exception as e:
        click.echo(f"Error listing workflows: {e}")
        return 1


@workflow.command(name="remove", help="Remove a workflow execution state by workflow name and execution ID")
@click.argument("workflow_name")
@click.argument("execution_id")
@pass_cliver
def remove_execution(cliver: Cliver, workflow_name: str, execution_id: str):
    """Remove a workflow execution state."""
    try:
        # Use Cliver's workflow manager and executor
        workflow_manager = cliver.workflow_manager
        workflow_executor = cliver.workflow_executor

        # Check if the workflow exists
        _workflow = workflow_manager.load_workflow(workflow_name)
        if not _workflow:
            click.echo(f"Workflow {workflow_name} not found.")
            return 1

        # Get execution state to verify it exists
        state = workflow_executor.get_execution_state(workflow_name, execution_id)
        if not state:
            click.echo(f"Execution {execution_id} not found for workflow {workflow_name}.")
            return 1

        # Remove execution
        if workflow_executor.remove_workflow_execution(workflow_name, execution_id):
            click.echo(f"Execution {execution_id} removed successfully from workflow {workflow_name}.")
            return 0
        else:
            click.echo(f"Failed to remove execution {execution_id} from workflow {workflow_name}.")
            return 1

    except Exception as e:
        click.echo(f"Error removing execution: {e}")
        return 1


@workflow.command(name="status", help="Show status of workflow executions")
@pass_cliver
def execution_status(cliver: Cliver):
    """Show status of workflow executions."""
    try:
        # Use Cliver's workflow manager and executor
        workflow_manager = cliver.workflow_manager
        workflow_executor = cliver.workflow_executor

        # List all workflows to get their names
        workflows = workflow_manager.list_workflows()
        workflow_names = list(workflows.keys()) if workflows else []

        # Get cache to list executions for each workflow
        all_executions = {}
        for workflow_name in workflow_names:
            executions = workflow_executor.list_executions(workflow_name)
            all_executions.update(executions)

        if not all_executions:
            click.echo("No workflow executions found.")
            return 0

        click.echo("Workflow executions:")
        for execution_id, metadata in all_executions.items():
            status = metadata.get('status', 'unknown')
            workflow_name = metadata.get('workflow_name', 'unknown')
            completed_steps = len(metadata.get('completed_steps', []))
            click.echo(f"  {execution_id}: {workflow_name} ({status}) - {completed_steps} steps completed")

    except Exception as e:
        click.echo(f"Error getting execution status: {e}")
        return 1


@workflow.command(name="clear", help="Clear all workflow execution states")
@pass_cliver
def clear_executions(cliver: Cliver):
    """Clear all workflow execution states."""
    try:
        # Use Cliver's workflow manager and executor
        workflow_manager = cliver.workflow_manager
        workflow_executor = cliver.workflow_executor

        # List all workflows to get their names
        workflows = workflow_manager.list_workflows()
        workflow_names = list(workflows.keys()) if workflows else []

        # Clear all executions for each workflow
        total_count = 0
        for workflow_name in workflow_names:
            count = workflow_executor.clear_all_executions(workflow_name)
            total_count += count

        click.echo(f"Cleared {total_count} workflow executions.")

    except Exception as e:
        click.echo(f"Error clearing executions: {e}")
        return 1


def post_group():
    """Post group initialization."""
    pass