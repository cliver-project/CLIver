"""
Task management commands for CLIver CLI.
"""

import asyncio
import uuid
from typing import Optional

import click

from cliver.cli import Cliver, pass_cliver
from cliver.task_manager import TaskDefinition, TaskManager, TaskRun


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
        cliver.output("No tasks defined.")
        return 0

    for t in tasks:
        schedule = f"  schedule: {t.schedule}" if t.schedule else ""
        model = f"  model: {t.model}" if t.model else ""
        workflow = f"  workflow: {t.workflow}" if t.workflow else ""
        skills = f"  skills: {', '.join(t.skills)}" if t.skills else ""
        desc = f"  — {t.description}" if t.description else ""
        cliver.output(f"  {t.name}:{desc}{model}{workflow}{skills}{schedule}")

    return 0


@task.command(name="create", help="Create a new task")
@click.argument("name")
@click.option("--prompt", "-p", required=True, help="Prompt to send to the LLM")
@click.option("--description", "-d", default=None, help="Task description")
@click.option("--model", "-m", default=None, help="Model override for this task")
@click.option("--schedule", "-s", default=None, help="Cron expression for recurring execution")
@click.option("--workflow", "-w", default=None, help="Workflow name to execute (runs workflow instead of chat)")
@click.option("--workflow-inputs", default=None, help="Workflow inputs as JSON string")
@click.option("--skills", multiple=True, help="Skills to pre-activate (can be specified multiple times)")
@pass_cliver
def create_task(
    cliver: Cliver,
    name: str,
    prompt: str,
    description: Optional[str],
    model: Optional[str],
    schedule: Optional[str],
    workflow: Optional[str],
    workflow_inputs: Optional[str],
    skills: tuple,
):
    """Create a new task definition."""
    import json

    # Parse workflow inputs if provided
    parsed_inputs = None
    if workflow_inputs:
        try:
            parsed_inputs = json.loads(workflow_inputs)
        except json.JSONDecodeError as e:
            cliver.output(f"Invalid JSON for --workflow-inputs: {e}")
            return 1

    task_def = TaskDefinition(
        name=name,
        description=description,
        prompt=prompt,
        workflow=workflow,
        workflow_inputs=parsed_inputs,
        skills=list(skills) if skills else None,
        model=model,
        schedule=schedule,
    )

    manager = TaskManager(cliver.agent_profile.tasks_dir)
    path = manager.save_task(task_def)
    cliver.output(f"Task '{name}' created: {path}")
    return 0


@task.command(name="run", help="Run a task")
@click.argument("name")
@click.option("--model", "-m", default=None, help="Override the task's model")
@pass_cliver
def run_task(cliver: Cliver, name: str, model: Optional[str]):
    """Run a task by sending its prompt to the LLM."""
    manager = TaskManager(cliver.agent_profile.tasks_dir)
    task_def = manager.get_task(name)
    if not task_def:
        cliver.output(f"Task '{name}' not found.")
        return 1

    execution_id = str(uuid.uuid4())[:8]
    started_at = TaskManager.timestamp_now()
    use_model = model or task_def.model

    cliver.output(f"Running task '{name}'...")

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

    try:
        result = asyncio.run(
            cliver.task_executor.process_user_input(
                user_input=task_def.prompt,
                model=use_model,
            )
        )

        status = "completed" if result else "failed"
        error = None if result else "No result returned"

        run = TaskRun(
            task_name=name,
            execution_id=execution_id,
            status=status,
            started_at=started_at,
            finished_at=TaskManager.timestamp_now(),
            error=error,
        )
        manager.record_run(run)

        if status == "completed":
            cliver.output(f"Task '{name}' completed.")
        else:
            cliver.output(f"Task '{name}' failed: {error}")

        return 0 if status == "completed" else 1

    except Exception as e:
        run = TaskRun(
            task_name=name,
            execution_id=execution_id,
            status="failed",
            started_at=started_at,
            finished_at=TaskManager.timestamp_now(),
            error=str(e),
        )
        manager.record_run(run)
        cliver.output(f"Task '{name}' failed: {e}")
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
        cliver.output(f"No run history for task '{name}'.")
        return 0

    cliver.output(f"Run history for '{name}' (most recent first):")
    for run in runs:
        status_icon = "+" if run.status == "completed" else "x"
        error_info = f" — {run.error}" if run.error else ""
        cliver.output(f"  [{status_icon}] {run.started_at}  {run.status}{error_info}")

    return 0


@task.command(name="remove", help="Remove a task")
@click.argument("name")
@pass_cliver
def remove_task(cliver: Cliver, name: str):
    """Remove a task and its run history."""
    manager = TaskManager(cliver.agent_profile.tasks_dir)
    if manager.remove_task(name):
        cliver.output(f"Task '{name}' removed.")
    else:
        cliver.output(f"Task '{name}' not found.")
    return 0
