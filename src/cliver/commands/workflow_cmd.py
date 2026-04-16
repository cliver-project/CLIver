"""Workflow management commands."""

import asyncio
import logging

import click
import yaml

from cliver.cli import Cliver, pass_cliver

logger = logging.getLogger(__name__)


@click.group(name="workflow", help="Manage and execute workflows")
def workflow_cmd():
    pass


@workflow_cmd.command(name="list", help="List all saved workflows")
@pass_cliver
def list_workflows(cliver: Cliver):
    from cliver.workflow.persistence import WorkflowStore

    store = WorkflowStore(cliver.agent_profile.workflows_dir)
    names = store.list_workflows()

    if not names:
        cliver.output("No workflows saved.")
        return

    cliver.output("[bold]Saved workflows:[/bold]\n")
    for name in names:
        wf = store.load_workflow(name)
        desc = wf.description or "(no description)" if wf else "(error loading)"
        steps = len(wf.steps) if wf else 0
        cliver.output(f"  [bold]{name}[/bold] — {desc} ({steps} steps)")


@workflow_cmd.command(name="show", help="Show a workflow definition")
@click.argument("name")
@pass_cliver
def show_workflow(cliver: Cliver, name: str):
    from cliver.workflow.persistence import WorkflowStore

    store = WorkflowStore(cliver.agent_profile.workflows_dir)
    wf = store.load_workflow(name)

    if not wf:
        cliver.output(f"Workflow '{name}' not found.")
        return

    cliver.output(yaml.dump(wf.model_dump(exclude_none=True), default_flow_style=False, sort_keys=False))


@workflow_cmd.command(name="run", help="Execute a workflow")
@click.argument("name")
@click.option("--input", "-i", "inputs", multiple=True, type=str, help="Input key=value pairs")
@pass_cliver
def run_workflow(cliver: Cliver, name: str, inputs: tuple):
    from cliver.workflow.persistence import WorkflowStore
    from cliver.workflow.workflow_executor import WorkflowExecutor

    store = WorkflowStore(cliver.agent_profile.workflows_dir)
    wf = store.load_workflow(name)

    if not wf:
        cliver.output(f"Workflow '{name}' not found.")
        return

    # Parse inputs
    parsed_inputs = {}
    for inp in inputs:
        if "=" in inp:
            key, value = inp.split("=", 1)
            parsed_inputs[key.strip()] = value.strip()

    executor = WorkflowExecutor(cliver.task_executor, store)

    cliver.output(f"[bold]Running workflow: {name}[/bold]\n")

    try:
        state = asyncio.run(executor.execute_workflow(name, inputs=parsed_inputs or None))
    except RuntimeError:
        loop = asyncio.get_event_loop()
        state = loop.run_until_complete(executor.execute_workflow(name, inputs=parsed_inputs or None))

    if state:
        cliver.output(f"\n[bold]Status:[/bold] {state.status}")
        if state.error:
            cliver.output(f"[red]Error: {state.error}[/red]")
        if state.completed_steps:
            cliver.output(f"[bold]Completed:[/bold] {', '.join(state.completed_steps)}")
        if state.skipped_steps:
            cliver.output(f"[dim]Skipped:[/dim] {', '.join(state.skipped_steps)}")
        if state.execution_time:
            cliver.output(f"[dim]Duration: {state.execution_time:.1f}s[/dim]")


@workflow_cmd.command(name="resume", help="Resume a paused workflow")
@click.argument("name")
@pass_cliver
def resume_workflow(cliver: Cliver, name: str):
    from cliver.workflow.persistence import WorkflowStore
    from cliver.workflow.workflow_executor import WorkflowExecutor

    store = WorkflowStore(cliver.agent_profile.workflows_dir)
    executor = WorkflowExecutor(cliver.task_executor, store)

    state = store.load_state(name)
    if not state or state.status != "paused":
        cliver.output(f"No paused workflow '{name}' to resume.")
        return

    cliver.output(f"[bold]Resuming workflow: {name}[/bold]\n")

    try:
        result = asyncio.run(executor.resume_workflow(name))
    except RuntimeError:
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(executor.resume_workflow(name))

    if result:
        cliver.output(f"\n[bold]Status:[/bold] {result.status}")
        if result.error:
            cliver.output(f"[red]Error: {result.error}[/red]")


@workflow_cmd.command(name="delete", help="Delete a workflow")
@click.argument("name")
@pass_cliver
def delete_workflow(cliver: Cliver, name: str):
    from cliver.workflow.persistence import WorkflowStore

    store = WorkflowStore(cliver.agent_profile.workflows_dir)
    if store.delete_workflow(name):
        cliver.output(f"Deleted workflow '{name}'.")
    else:
        cliver.output(f"Workflow '{name}' not found.")
