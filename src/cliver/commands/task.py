"""
Task management commands for CLIver CLI.
"""

import asyncio
import uuid
from typing import Optional

import click

from cliver.cli import Cliver, pass_cliver
from cliver.commands.console_callback_handler import ConsoleCallbackHandler
from cliver.task_manager import TaskDefinition, TaskManager, TaskRun
from cliver.util import parse_key_value_options
from cliver.workflow.workflow_executor import WorkflowExecutor


@click.group(name="task", help="Manage and run agent tasks")
def task():
    """Task management commands."""
    pass


@task.command(name="list", help="List all tasks for the current agent")
@pass_cliver
def list_tasks(cliver: Cliver):
    """List all tasks."""
    manager = TaskManager(cliver.agent_profile.tasks_dir)
    tasks = manager.list_tasks()

    if not tasks:
        click.echo("No tasks defined.")
        return 0

    for t in tasks:
        schedule = f"  schedule: {t.schedule}" if t.schedule else ""
        desc = f"  — {t.description}" if t.description else ""
        click.echo(f"  {t.name}: workflow={t.workflow}{desc}{schedule}")

    return 0


@task.command(name="create", help="Create a new task")
@click.argument("name")
@click.option("--workflow", "-w", required=True, help="Workflow name to execute")
@click.option("--description", "-d", default=None, help="Task description")
@click.option("--input", "-i", "inputs", multiple=True, type=str, help="Input variables (key=value)")
@click.option("--schedule", "-s", default=None, help="Cron expression for recurring execution")
@pass_cliver
def create_task(
    cliver: Cliver,
    name: str,
    workflow: str,
    description: Optional[str],
    inputs: tuple,
    schedule: Optional[str],
):
    """Create a new task definition."""
    # Verify workflow exists
    wf = cliver.workflow_manager.load_workflow(workflow)
    if not wf:
        click.echo(f"Workflow '{workflow}' not found.")
        return 1

    input_dict = parse_key_value_options(inputs) if inputs else None

    task_def = TaskDefinition(
        name=name,
        description=description,
        workflow=workflow,
        inputs=input_dict,
        schedule=schedule,
    )

    manager = TaskManager(cliver.agent_profile.tasks_dir)
    path = manager.save_task(task_def)
    click.echo(f"Task '{name}' created: {path}")
    return 0


@task.command(name="run", help="Run a task")
@click.argument("name")
@click.option("--input", "-i", "inputs", multiple=True, type=str, help="Override input variables (key=value)")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed execution progress")
@pass_cliver
def run_task(cliver: Cliver, name: str, inputs: tuple, verbose: bool):
    """Run a task by executing its workflow."""
    manager = TaskManager(cliver.agent_profile.tasks_dir)
    task_def = manager.get_task(name)
    if not task_def:
        click.echo(f"Task '{name}' not found.")
        return 1

    # Merge task inputs with overrides
    task_inputs = dict(task_def.inputs or {})
    if inputs:
        task_inputs.update(parse_key_value_options(inputs))

    execution_id = str(uuid.uuid4())
    started_at = TaskManager.timestamp_now()

    click.echo(f"Running task '{name}' (workflow: {task_def.workflow})...")

    # Push task-level permissions if defined
    task_perms_pushed = False
    if task_def.permissions and cliver.permission_manager:
        from cliver.permissions import TaskPermissions

        perms = (
            task_def.permissions
            if isinstance(task_def.permissions, TaskPermissions)
            else TaskPermissions(**task_def.permissions)
        )
        cliver.permission_manager.push_task_scope(perms)
        task_perms_pushed = True

    # Create executor with callback handler
    callback = ConsoleCallbackHandler() if verbose else None
    executor = WorkflowExecutor(
        task_executor=cliver.task_executor,
        workflow_manager=cliver.workflow_manager,
        callback_handler=callback,
    )

    try:
        result = asyncio.run(
            executor.execute_workflow(
                workflow_name=task_def.workflow,
                inputs=task_inputs,
                execution_id=execution_id,
            )
        )

        status = result.status if result else "failed"
        error = result.error if result else "No result returned"

        # Record the run
        run = TaskRun(
            task_name=name,
            execution_id=execution_id,
            workflow_name=task_def.workflow,
            status=status,
            started_at=started_at,
            finished_at=TaskManager.timestamp_now(),
            error=error,
        )
        manager.record_run(run)

        if status == "completed":
            click.echo(f"Task '{name}' completed successfully.")
        else:
            click.echo(f"Task '{name}' {status}: {error}")

        return 0 if status == "completed" else 1

    except Exception as e:
        run = TaskRun(
            task_name=name,
            execution_id=execution_id,
            workflow_name=task_def.workflow,
            status="failed",
            started_at=started_at,
            finished_at=TaskManager.timestamp_now(),
            error=str(e),
        )
        manager.record_run(run)
        click.echo(f"Task '{name}' failed: {e}")
        return 1

    finally:
        if task_perms_pushed and cliver.permission_manager:
            cliver.permission_manager.pop_task_scope()


@task.command(name="history", help="Show run history for a task")
@click.argument("name")
@click.option("--limit", "-n", default=10, help="Number of recent runs to show")
@pass_cliver
def task_history(cliver: Cliver, name: str, limit: int):
    """Show execution history for a task."""
    manager = TaskManager(cliver.agent_profile.tasks_dir)
    runs = manager.get_runs(name, limit=limit)

    if not runs:
        click.echo(f"No run history for task '{name}'.")
        return 0

    click.echo(f"Run history for '{name}' (most recent first):")
    for run in runs:
        status_icon = "+" if run.status == "completed" else "x"
        error_info = f" — {run.error}" if run.error else ""
        click.echo(f"  [{status_icon}] {run.started_at}  {run.status}{error_info}")

    return 0


@task.command(name="remove", help="Remove a task")
@click.argument("name")
@pass_cliver
def remove_task(cliver: Cliver, name: str):
    """Remove a task and its run history."""
    manager = TaskManager(cliver.agent_profile.tasks_dir)
    if manager.remove_task(name):
        click.echo(f"Task '{name}' removed.")
    else:
        click.echo(f"Task '{name}' not found.")
    return 0
