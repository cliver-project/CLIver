"""Workflow management commands."""

import asyncio
import logging
from pathlib import Path

import click
import yaml

from cliver.cli import Cliver, pass_cliver
from cliver.commands import click_help, wants_help

logger = logging.getLogger(__name__)


# ── Helpers ──


def _run_async(coro):
    """Run an async coroutine, handling existing event loops."""
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coro)


def _make_executor(cliver: Cliver):
    """Create a WorkflowExecutor from a Cliver instance."""
    from cliver.workflow.persistence import WorkflowStore
    from cliver.workflow.workflow_executor import WorkflowExecutor

    store = WorkflowStore(cliver.agent_profile.workflows_dir)
    return WorkflowExecutor(
        agent_core=cliver.agent_core,
        store=store,
        db_path=cliver.agent_profile.workflow_checkpoints_db,
        app_config=cliver.config_manager.config,
        skill_manager=getattr(cliver, "skill_manager", None),
    ), store


# ── Logic Functions ──


def _list_workflows(cliver: Cliver):
    """List all saved workflows."""
    from cliver.workflow.persistence import WorkflowStore

    store = WorkflowStore(cliver.agent_profile.workflows_dir)
    entries = store.list_all_workflows()

    if not entries:
        cliver.output("No workflows saved.")
        return

    cliver.output("[bold]Saved workflows:[/bold]\n")
    for name, source, path in entries:
        wf = WorkflowStore.load_workflow_from_file(path)
        if wf:
            desc = wf.description or "(no description)"
            steps = len(wf.steps)
            agents = len(wf.agents) if wf.agents else 0
            source_tag = f"[dim][{source}][/dim] " if source == "project" else ""
            cliver.output(f"  {source_tag}[bold]{name}[/bold] — {desc} ({steps} steps, {agents} agents)")
        else:
            source_tag = f"[dim][{source}][/dim] " if source == "project" else ""
            cliver.output(f"  {source_tag}[bold]{name}[/bold] — [red]error loading {path}[/red]")


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
    executor, store = _make_executor(cliver)
    wf = store.load_workflow(name)

    if not wf:
        cliver.output(f"Workflow '{name}' not found.")
        return

    parsed_inputs = {}
    for inp in inputs:
        if "=" in inp:
            key, value = inp.split("=", 1)
            parsed_inputs[key.strip()] = value.strip()

    cliver.output(f"[bold]Running workflow: {name}[/bold]\n")

    async def _execute():
        try:
            return await executor.execute_workflow(name, inputs=parsed_inputs or None)
        finally:
            await executor.close()

    result = _run_async(_execute())

    if result:
        _display_result(cliver, result)


def _history_workflow(cliver: Cliver, name: str):
    """Show execution history for a workflow."""
    from cliver.workflow.persistence import WorkflowStore

    db_path = cliver.agent_profile.workflow_checkpoints_db
    executions = WorkflowStore.list_executions(db_path, name)

    if not executions:
        cliver.output(f"No executions found for workflow '{name}'.")
        return

    cliver.output(f"[bold]Execution history for '{name}':[/bold]\n")
    for ex in executions:
        thread = ex["thread_id"]
        status = ex["status"]
        started = ex.get("started_at", "?")
        finished = ex.get("finished_at")

        status_color = {"completed": "green", "failed": "red", "running": "yellow"}.get(status, "dim")
        cliver.output(f"  [{status_color}]{status:>11}[/{status_color}]  {thread}")
        cliver.output(f"             started: {started}")
        if finished:
            cliver.output(f"             finished: {finished}")
        if ex.get("error"):
            cliver.output(f"             [red]error: {ex['error']}[/red]")
        cliver.output("")


def _status_workflow(cliver: Cliver, name: str, thread_id: str):
    """Show step-by-step status of a specific execution."""
    executor, _ = _make_executor(cliver)

    async def _get_status():
        try:
            return await executor.get_execution_status(name, thread_id)
        finally:
            await executor.close()

    status = _run_async(_get_status())

    if not status:
        cliver.output(f"No execution found for thread '{thread_id}'.")
        return

    cliver.output(f"[bold]Execution status: {thread_id}[/bold]\n")

    for step in status["steps"]:
        sid = step["id"]
        st = step["status"]
        color = {"completed": "green", "failed": "red", "pending": "dim"}.get(st, "white")
        time_str = f" ({step['execution_time']:.1f}s)" if step.get("execution_time") else ""
        error_str = f"\n             [red]Error: {step['error']}[/red]" if step.get("error") else ""
        cliver.output(f"  [{color}]{st:>9}[/{color}]  {sid}{time_str}{error_str}")

    if status["next_steps"]:
        cliver.output(f"\n[bold]Next:[/bold] {', '.join(status['next_steps'])}")
    if status["has_interrupts"]:
        cliver.output("[yellow]Workflow is paused at a human step.[/yellow]")


def _resume_workflow(cliver: Cliver, name: str, thread: str, answer: str | None, step: str | None):
    """Resume a paused workflow or replay from a specific step."""
    executor, store = _make_executor(cliver)
    wf = store.load_workflow(name)

    if not wf:
        cliver.output(f"Workflow '{name}' not found.")
        return

    async def _execute():
        try:
            if step:
                return await executor.resume_from_step(name, thread, step)
            return await executor.resume_workflow(name, thread_id=thread, resume_value=answer)
        finally:
            await executor.close()

    if step:
        cliver.output(f"[bold]Resuming workflow '{name}' from step '{step}'[/bold]\n")
    else:
        cliver.output(f"[bold]Resuming workflow: {name}[/bold]\n")

    result = _run_async(_execute())

    if result:
        _display_result(cliver, result)


def _delete_workflow(cliver: Cliver, name: str):
    """Delete a workflow and its associated checkpoints."""
    from cliver.workflow.persistence import WorkflowStore

    store = WorkflowStore(cliver.agent_profile.workflows_dir)
    db_path = cliver.agent_profile.workflow_checkpoints_db
    checkpointer = _get_sync_checkpointer(db_path)
    if store.delete_workflow(name, checkpointer=checkpointer, db_path=db_path):
        cliver.output(f"Deleted workflow '{name}' and its checkpoints.")
    else:
        cliver.output(f"Workflow '{name}' not found.")


def _prune_workflow(cliver: Cliver, days: int):
    """Prune old execution data."""
    from cliver.workflow.persistence import WorkflowStore

    db_path = cliver.agent_profile.workflow_checkpoints_db
    checkpointer = _get_sync_checkpointer(db_path)
    deleted = WorkflowStore.prune_executions(db_path, retention_days=days, checkpointer=checkpointer)
    if deleted:
        cliver.output(f"Pruned {deleted} executions (older than {days} days).")
    else:
        cliver.output("No stale executions to prune.")


def _delete_execution(cliver: Cliver, thread_id: str):
    """Delete a specific execution and its checkpoints."""
    from cliver.workflow.persistence import WorkflowStore

    db_path = cliver.agent_profile.workflow_checkpoints_db
    checkpointer = _get_sync_checkpointer(db_path)
    deleted = WorkflowStore.delete_execution(thread_id, db_path, checkpointer=checkpointer)
    if deleted:
        cliver.output(f"Deleted execution '{thread_id}' and its checkpoints.")
    else:
        cliver.output(f"No execution found for '{thread_id}'.")


def _get_sync_checkpointer(db_path):
    """Get a sync SqliteSaver for checkpoint cleanup operations."""
    if not Path(db_path).exists():
        return None
    try:
        import sqlite3

        from langgraph.checkpoint.sqlite import SqliteSaver

        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        saver = SqliteSaver(conn)
        saver.setup()
        return saver
    except Exception:
        return None


def _display_result(cliver: Cliver, result: dict):
    """Display workflow execution result."""
    if result.get("error"):
        cliver.output(f"[red]Error: {result['error']}[/red]")

    steps = result.get("steps", {})
    if steps and isinstance(steps, dict):
        cliver.output("[bold]Steps:[/bold]")
        for step_id, step_data in steps.items():
            status = step_data.get("status", "unknown")
            color = {"completed": "green", "failed": "red"}.get(status, "white")
            time_str = f" ({step_data.get('execution_time', 0):.1f}s)" if step_data.get("execution_time") else ""
            cliver.output(f"  [{color}]{status:>9}[/{color}]  {step_id}{time_str}")
            if step_data.get("outputs", {}).get("error"):
                cliver.output(f"             [red]Error: {step_data['outputs']['error']}[/red]")

    if result.get("outputs_dir"):
        cliver.output(f"\n[dim]Outputs: {result['outputs_dir']}[/dim]")


# ── Dispatch ──


def dispatch(cliver: Cliver, args: str):
    """Manage workflows — list, show, run, resume, history, status, delete."""
    parts = args.strip().split(None, 1) if args.strip() else []
    sub = parts[0] if parts else "list"
    rest = parts[1] if len(parts) > 1 else ""

    if sub in ("--help", "-h", "help"):
        cliver.output(click_help(workflow_cmd, "/workflow"))
        return

    if sub in _SUBCOMMANDS and wants_help(rest):
        cliver.output(click_help(_SUBCOMMANDS[sub], f"/workflow {sub}"))
        return

    if sub == "list":
        _list_workflows(cliver)
    elif sub == "show":
        if not rest:
            cliver.output(click_help(_SUBCOMMANDS["show"], "/workflow show"))
            return
        _show_workflow(cliver, rest.strip())
    elif sub == "run":
        run_parts = rest.split()
        if not run_parts:
            cliver.output(click_help(_SUBCOMMANDS["run"], "/workflow run"))
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
    elif sub == "history":
        if not rest:
            cliver.output(click_help(_SUBCOMMANDS["history"], "/workflow history"))
            return
        _history_workflow(cliver, rest.strip())
    elif sub == "status":
        status_parts = rest.split()
        if not status_parts:
            cliver.output(click_help(_SUBCOMMANDS["status"], "/workflow status"))
            return
        name = status_parts[0]
        thread = None
        i = 1
        while i < len(status_parts):
            if status_parts[i] in ("--thread", "-t") and i + 1 < len(status_parts):
                thread = status_parts[i + 1]
                i += 2
            else:
                i += 1
        if not thread:
            cliver.output(click_help(_SUBCOMMANDS["status"], "/workflow status"))
            return
        _status_workflow(cliver, name, thread)
    elif sub == "resume":
        resume_parts = rest.split()
        if not resume_parts:
            cliver.output(click_help(_SUBCOMMANDS["resume"], "/workflow resume"))
            return
        name = resume_parts[0]
        thread = None
        answer = None
        step = None
        i = 1
        while i < len(resume_parts):
            if resume_parts[i] in ("--thread", "-t") and i + 1 < len(resume_parts):
                thread = resume_parts[i + 1]
                i += 2
            elif resume_parts[i] in ("--answer", "-a") and i + 1 < len(resume_parts):
                answer = resume_parts[i + 1]
                i += 2
            elif resume_parts[i] in ("--step", "-s") and i + 1 < len(resume_parts):
                step = resume_parts[i + 1]
                i += 2
            else:
                i += 1
        if not thread:
            cliver.output(click_help(_SUBCOMMANDS["resume"], "/workflow resume"))
            return
        _resume_workflow(cliver, name, thread, answer, step)
    elif sub == "delete":
        if not rest:
            cliver.output(click_help(_SUBCOMMANDS["delete"], "/workflow delete"))
            return
        _delete_workflow(cliver, rest.strip())
    elif sub == "delete-execution":
        if not rest:
            cliver.output(click_help(_SUBCOMMANDS["delete-execution"], "/workflow delete-execution"))
            return
        _delete_execution(cliver, rest.strip())
    elif sub == "prune":
        days = 300
        if rest.strip():
            try:
                days = int(rest.strip())
            except ValueError:
                cliver.output("[red]Invalid number of days.[/red]")
                return
        _prune_workflow(cliver, days)
    else:
        cliver.output(f"[yellow]Unknown: /workflow {sub}[/yellow]")


@click.group(name="workflow", help="Manage and execute LangGraph-powered multi-step workflows with checkpointing")
def workflow_cmd():
    pass


@workflow_cmd.command(name="list", help="List all saved workflows with name, description, step and agent counts")
@pass_cliver
def list_workflows(cliver: Cliver):
    """List all saved workflows."""
    _list_workflows(cliver)


@workflow_cmd.command(name="show", help="Display a workflow definition as YAML")
@click.argument("name")
@pass_cliver
def show_workflow(cliver: Cliver, name: str):
    """Show a workflow definition."""
    _show_workflow(cliver, name)


@workflow_cmd.command(name="run", help="Execute a workflow from the beginning with optional input variables")
@click.argument("name")
@click.option(
    "--input",
    "-i",
    "inputs",
    multiple=True,
    type=str,
    help="Input variable as key=value pair (repeatable, e.g. --input topic=auth)",
)
@pass_cliver
def run_workflow(cliver: Cliver, name: str, inputs: tuple):
    """Execute a workflow."""
    _run_workflow(cliver, name, inputs)


@workflow_cmd.command(name="history", help="Show execution history for a workflow")
@click.argument("name")
@pass_cliver
def history_workflow(cliver: Cliver, name: str):
    """Show execution history."""
    _history_workflow(cliver, name)


@workflow_cmd.command(name="status", help="Show step-by-step status of a specific execution")
@click.argument("name")
@click.option("--thread", "-t", type=str, required=True, help="Thread ID of the execution")
@pass_cliver
def status_workflow(cliver: Cliver, name: str, thread: str):
    """Show execution status."""
    _status_workflow(cliver, name, thread)


@workflow_cmd.command(name="resume", help="Resume a previously paused workflow or replay from a step")
@click.argument("name")
@click.option("--thread", "-t", type=str, required=True, help="Thread ID of the execution")
@click.option("--answer", "-a", type=str, default=None, help="Answer for a paused human step")
@click.option("--step", "-s", type=str, default=None, help="Replay from this step forward")
@pass_cliver
def resume_workflow(cliver: Cliver, name: str, thread: str, answer: str | None, step: str | None):
    """Resume a paused workflow."""
    _resume_workflow(cliver, name, thread, answer, step)


@workflow_cmd.command(name="delete", help="Delete a workflow definition permanently")
@click.argument("name")
@pass_cliver
def delete_workflow(cliver: Cliver, name: str):
    """Delete a workflow."""
    _delete_workflow(cliver, name)


@workflow_cmd.command(name="delete-execution", help="Delete checkpoints for a single execution")
@click.argument("thread_id")
@pass_cliver
def delete_execution(cliver: Cliver, thread_id: str):
    """Delete a specific execution's checkpoints."""
    _delete_execution(cliver, thread_id)


@workflow_cmd.command(name="prune", help="Delete old checkpoint data")
@click.argument("days", type=int, default=300)
@pass_cliver
def prune_checkpoints(cliver: Cliver, days: int):
    """Prune old checkpoints."""
    _prune_workflow(cliver, days)


# Subcommand lookup for dispatch help
_SUBCOMMANDS = {
    "list": list_workflows,
    "show": show_workflow,
    "run": run_workflow,
    "history": history_workflow,
    "status": status_workflow,
    "resume": resume_workflow,
    "delete": delete_workflow,
    "delete-execution": delete_execution,
    "prune": prune_checkpoints,
}
