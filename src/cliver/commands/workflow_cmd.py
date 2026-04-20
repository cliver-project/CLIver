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
        agents = len(wf.agents) if wf and wf.agents else 0
        cliver.output(f"  [bold]{name}[/bold] — {desc} ({steps} steps, {agents} agents)")


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

    db_path = cliver.config_dir / "workflow-checkpoints.db"
    executor = WorkflowExecutor(
        task_executor=cliver.task_executor,
        store=store,
        db_path=db_path,
        app_config=cliver.config_manager.config,
        skill_manager=getattr(cliver, "skill_manager", None),
    )

    cliver.output(f"[bold]Running workflow: {name}[/bold]\n")

    try:
        result = asyncio.run(executor.execute_workflow(name, inputs=parsed_inputs or None))
    except RuntimeError:
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(executor.execute_workflow(name, inputs=parsed_inputs or None))

    if result:
        if result.get("error"):
            cliver.output(f"[red]Error: {result['error']}[/red]")

        steps = result.get("steps", [])
        if steps:
            cliver.output("[bold]Execution steps:[/bold]")
            for step_result in steps:
                step_name = step_result.get("step", "unknown")
                status = step_result.get("status", "unknown")
                cliver.output(f"  - {step_name}: {status}")
                if step_result.get("error"):
                    cliver.output(f"    [red]Error: {step_result['error']}[/red]")

        if result.get("outputs_dir"):
            cliver.output(f"\n[dim]Outputs directory: {result['outputs_dir']}[/dim]")


@workflow_cmd.command(name="resume", help="Resume a paused workflow")
@click.argument("name")
@click.option("--thread", "-t", type=str, required=True, help="Checkpoint thread ID")
@click.option("--answer", "-a", type=str, default=None, help="Answer for human input step")
@pass_cliver
def resume_workflow(cliver: Cliver, name: str, thread: str, answer: str | None):
    from cliver.workflow.persistence import WorkflowStore
    from cliver.workflow.workflow_executor import WorkflowExecutor

    store = WorkflowStore(cliver.agent_profile.workflows_dir)
    wf = store.load_workflow(name)

    if not wf:
        cliver.output(f"Workflow '{name}' not found.")
        return

    db_path = cliver.config_dir / "workflow-checkpoints.db"
    executor = WorkflowExecutor(
        task_executor=cliver.task_executor,
        store=store,
        db_path=db_path,
        app_config=cliver.config_manager.config,
        skill_manager=getattr(cliver, "skill_manager", None),
    )

    cliver.output(f"[bold]Resuming workflow: {name}[/bold]\n")

    try:
        result = asyncio.run(executor.resume_workflow(name, thread_id=thread, human_answer=answer))
    except RuntimeError:
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(executor.resume_workflow(name, thread_id=thread, human_answer=answer))

    if result:
        if result.get("error"):
            cliver.output(f"[red]Error: {result['error']}[/red]")

        steps = result.get("steps", [])
        if steps:
            cliver.output("[bold]Execution steps:[/bold]")
            for step_result in steps:
                step_name = step_result.get("step", "unknown")
                status = step_result.get("status", "unknown")
                cliver.output(f"  - {step_name}: {status}")
                if step_result.get("error"):
                    cliver.output(f"    [red]Error: {step_result['error']}[/red]")

        if result.get("outputs_dir"):
            cliver.output(f"\n[dim]Outputs directory: {result['outputs_dir']}[/dim]")


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
