"""Workflow management commands."""

import asyncio
import logging

import click
import yaml

from cliver.cli import Cliver, pass_cliver

logger = logging.getLogger(__name__)


# ── Logic Functions ──


def _list_workflows(cliver: Cliver):
    """List all saved workflows."""
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


def _show_workflow(cliver: Cliver, name: str):
    """Show a workflow definition."""
    from cliver.workflow.persistence import WorkflowStore

    store = WorkflowStore(cliver.agent_profile.workflows_dir)
    wf = store.load_workflow(name)

    if not wf:
        cliver.output(f"Workflow '{name}' not found.")
        return

    cliver.output(yaml.dump(wf.model_dump(exclude_none=True), default_flow_style=False, sort_keys=False))


def _run_workflow(cliver: Cliver, name: str, inputs: tuple):
    """Execute a workflow."""
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


def _resume_workflow(cliver: Cliver, name: str, thread: str, answer: str | None):
    """Resume a paused workflow."""
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


def _delete_workflow(cliver: Cliver, name: str):
    """Delete a workflow."""
    from cliver.workflow.persistence import WorkflowStore

    store = WorkflowStore(cliver.agent_profile.workflows_dir)
    if store.delete_workflow(name):
        cliver.output(f"Deleted workflow '{name}'.")
    else:
        cliver.output(f"Workflow '{name}' not found.")


# ── Dispatch ──


def dispatch(cliver: Cliver, args: str):
    """Parse subcommand and route to logic functions."""
    parts = args.strip().split(None, 1) if args.strip() else []
    sub = parts[0] if parts else "list"
    rest = parts[1] if len(parts) > 1 else ""

    if sub == "list":
        _list_workflows(cliver)
    elif sub == "show":
        if not rest:
            cliver.output("[dim]Usage: workflow show NAME[/dim]")
            return
        _show_workflow(cliver, rest.strip())
    elif sub == "run":
        # Parse: workflow run NAME --input key=value --input key2=value2
        run_parts = rest.split()
        if not run_parts:
            cliver.output("[dim]Usage: workflow run NAME [--input key=value ...][/dim]")
            return
        name = run_parts[0]
        inputs = []
        i = 1
        while i < len(run_parts):
            if run_parts[i] in ("--input", "-i") and i + 1 < len(run_parts):
                inputs.append(run_parts[i + 1])
                i += 2
            else:
                i += 1
        _run_workflow(cliver, name, tuple(inputs))
    elif sub == "resume":
        # Parse: workflow resume NAME --thread THREAD [--answer ANSWER]
        resume_parts = rest.split()
        if not resume_parts:
            cliver.output("[dim]Usage: workflow resume NAME --thread THREAD [--answer ANSWER][/dim]")
            return
        name = resume_parts[0]
        thread = None
        answer = None
        i = 1
        while i < len(resume_parts):
            if resume_parts[i] in ("--thread", "-t") and i + 1 < len(resume_parts):
                thread = resume_parts[i + 1]
                i += 2
            elif resume_parts[i] in ("--answer", "-a") and i + 1 < len(resume_parts):
                answer = resume_parts[i + 1]
                i += 2
            else:
                i += 1
        if not thread:
            cliver.output("[dim]Usage: workflow resume NAME --thread THREAD [--answer ANSWER][/dim]")
            return
        _resume_workflow(cliver, name, thread, answer)
    elif sub == "delete":
        if not rest:
            cliver.output("[dim]Usage: workflow delete NAME[/dim]")
            return
        _delete_workflow(cliver, rest.strip())
    elif sub in ("--help", "help"):
        cliver.output("Usage: /workflow [list|show|run|resume|delete] ...")
    else:
        cliver.output(f"[yellow]Unknown: /workflow {sub}[/yellow]")


@click.group(name="workflow", help="Manage and execute workflows")
def workflow_cmd():
    pass


@workflow_cmd.command(name="list", help="List all saved workflows")
@pass_cliver
def list_workflows(cliver: Cliver):
    """List all saved workflows."""
    _list_workflows(cliver)


@workflow_cmd.command(name="show", help="Show a workflow definition")
@click.argument("name")
@pass_cliver
def show_workflow(cliver: Cliver, name: str):
    """Show a workflow definition."""
    _show_workflow(cliver, name)


@workflow_cmd.command(name="run", help="Execute a workflow")
@click.argument("name")
@click.option("--input", "-i", "inputs", multiple=True, type=str, help="Input key=value pairs")
@pass_cliver
def run_workflow(cliver: Cliver, name: str, inputs: tuple):
    """Execute a workflow."""
    _run_workflow(cliver, name, inputs)


@workflow_cmd.command(name="resume", help="Resume a paused workflow")
@click.argument("name")
@click.option("--thread", "-t", type=str, required=True, help="Checkpoint thread ID")
@click.option("--answer", "-a", type=str, default=None, help="Answer for human input step")
@pass_cliver
def resume_workflow(cliver: Cliver, name: str, thread: str, answer: str | None):
    """Resume a paused workflow."""
    _resume_workflow(cliver, name, thread, answer)


@workflow_cmd.command(name="delete", help="Delete a workflow")
@click.argument("name")
@pass_cliver
def delete_workflow(cliver: Cliver, name: str):
    """Delete a workflow."""
    _delete_workflow(cliver, name)
