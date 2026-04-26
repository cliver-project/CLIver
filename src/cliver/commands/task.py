"""
Task management commands for CLIver CLI.
"""

import asyncio
import logging
import uuid
from typing import Optional

import click

from cliver.cli import Cliver, pass_cliver
from cliver.gateway.task_store import TaskStore
from cliver.task_manager import TaskDefinition, TaskManager, TaskRun

logger = logging.getLogger(__name__)

# Business logic (plain functions — no Click, no async)


def _get_store(cliver: Cliver) -> TaskStore:
    return TaskStore(cliver.agent_profile.agent_dir / "gateway.db")


def _get_manager(cliver: Cliver) -> TaskManager:
    store = _get_store(cliver)
    return TaskManager(cliver.agent_profile.tasks_dir, store)


def _list_tasks(cliver: Cliver) -> int:
    """List all tasks (DB-first, with status)."""
    manager = _get_manager(cliver)
    entries = manager.list_task_entries()

    if not entries:
        cliver.output("No tasks defined.")
        return 0

    store = _get_store(cliver)
    for entry in entries:
        if entry.status != "active" or not entry.definition:
            status_tag = f"  [{entry.status}]"
            error_info = f"  ({entry.error})" if entry.error else ""
            cliver.output(f"  {entry.name}:{status_tag}{error_info}")
            continue

        t = entry.definition
        schedule = f"  schedule: {t.schedule}" if t.schedule else ""
        model = f"  model: {t.model}" if t.model else ""
        workflow = f"  workflow: {t.workflow}" if t.workflow else ""
        skills = f"  skills: {', '.join(t.skills)}" if t.skills else ""
        desc = f"  — {t.description}" if t.description else ""
        task_origin = store.get_origin(t.name)
        origin = ""
        if task_origin:
            origin = f"  origin: {task_origin.source}"
            if task_origin.channel_id:
                origin += f" ({task_origin.channel_id})"
        cliver.output(f"  {t.name}:{desc}{model}{workflow}{skills}{schedule}{origin}")

    return 0


def _create_task(
    cliver: Cliver,
    name: str,
    prompt: str,
    description: Optional[str] = None,
    model: Optional[str] = None,
    schedule: Optional[str] = None,
    run_at: Optional[str] = None,
    workflow: Optional[str] = None,
    workflow_inputs: Optional[str] = None,
    skills: Optional[list] = None,
) -> int:
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

    # Validate run_at format
    if run_at:
        from datetime import datetime

        try:
            datetime.fromisoformat(run_at)
        except ValueError:
            cliver.output(f"[red]Invalid datetime format: {run_at}[/red]")
            cliver.output("[dim]Use ISO 8601 format, e.g. 2026-04-25T14:30:00[/dim]")
            return 1

    # Capture current CLI session if available
    session_id = getattr(cliver, "current_session_id", None)

    task_def = TaskDefinition(
        name=name,
        description=description,
        prompt=prompt,
        workflow=workflow,
        workflow_inputs=parsed_inputs,
        skills=skills,
        model=model,
        schedule=schedule,
        run_at=run_at,
        session_id=session_id,
    )

    manager = _get_manager(cliver)
    path = manager.save_task(task_def)

    # Auto-attach IM origin if running inside a gateway IM context
    origin_info = _save_im_origin(cliver, name)

    cliver.output(f"Task '{name}' created: {path}{origin_info}")
    return 0


def _save_im_origin(cliver: Cliver, task_name: str) -> str:
    """If running inside a gateway IM context, save origin so results reply back."""
    try:
        from cliver.gateway.gateway import im_context
        from cliver.task_manager import TaskOrigin

        ctx = im_context.get()
        if not ctx:
            return ""

        origin = TaskOrigin(
            source=ctx["platform"],
            platform=ctx["platform"],
            channel_id=ctx.get("channel_id"),
            thread_id=ctx.get("thread_id"),
            user_id=ctx.get("user_id"),
        )

        from cliver.gateway.task_store import TaskStore

        profile = cliver.agent_profile
        store = TaskStore(profile.agent_dir / "gateway.db")
        store.save_origin(task_name, origin)
        return f" (reply-back: {origin.source})"
    except ImportError:
        return ""
    except Exception as e:
        logger.warning("Failed to save IM origin for task '%s': %s", task_name, e)
        return ""


def _run_task(cliver: Cliver, name: str, model: Optional[str] = None) -> int:
    """Run a task by sending its prompt to the LLM."""
    manager = _get_manager(cliver)
    task_def = manager.get_task(name)
    if not task_def:
        cliver.output(f"Task '{name}' not found.")
        return 1

    execution_id = str(uuid.uuid4())[:8]
    started_at = TaskManager.timestamp_now()
    use_model = model or task_def.model
    store = _get_store(cliver)

    # Load origin from DB
    if not task_def.origin:
        task_def.origin = store.get_origin(name)

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
            cliver.agent_core.process_user_input(
                user_input=task_def.prompt,
                model=use_model,
            )
        )

        from cliver.media_handler import extract_response_text

        response_text = extract_response_text(result, fallback=None) if result else None
        status = "completed" if response_text else "failed"
        error = None if response_text else "No result returned"

        run = TaskRun(
            task_name=name,
            execution_id=execution_id,
            status=status,
            started_at=started_at,
            finished_at=TaskManager.timestamp_now(),
            error=error,
            result=response_text,
        )
        store.record_run(run)

        # Save JSON result for non-IM tasks
        is_im = task_def.origin and task_def.origin.platform and task_def.origin.channel_id
        if not is_im:
            _save_result_json(cliver, name, run)

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
        store.record_run(run)
        _save_result_json(cliver, name, run)
        cliver.output(f"Task '{name}' failed: {e}")
        return 1

    finally:
        if task_perms_pushed and cliver.permission_manager:
            cliver.permission_manager.pop_task_scope()


def _task_history(cliver: Cliver, name: str, limit: int = 10) -> int:
    """Show execution history for a task."""
    store = _get_store(cliver)
    runs = store.get_runs(name, limit=limit)

    if not runs:
        cliver.output(f"No run history for task '{name}'.")
        return 0

    cliver.output(f"Run history for '{name}' (most recent first):")
    task_origin = store.get_origin(name)
    if task_origin:
        cliver.output(f"  Origin: {task_origin.source}")
        if task_origin.channel_id:
            cliver.output(f"  Reply-to: {task_origin.channel_id} / {task_origin.thread_id}")
    for run in runs:
        status_icon = "+" if run.status == "completed" else "x"
        error_info = f" — {run.error}" if run.error else ""
        cliver.output(f"  [{status_icon}] {run.started_at}  {run.status}{error_info}")

    return 0


def _save_result_json(cliver: Cliver, task_name: str, run: TaskRun) -> None:
    """Save execution result as JSON file for non-IM tasks."""
    import json as json_mod

    task_results_dir = cliver.agent_profile.tasks_dir / task_name
    task_results_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{task_name}_execution_{run.execution_id}.json"
    result_path = task_results_dir / filename

    result_data = {
        "task_name": run.task_name,
        "execution_id": run.execution_id,
        "status": run.status,
        "started_at": run.started_at,
        "finished_at": run.finished_at,
        "result": run.result,
        "error": run.error,
    }

    result_path.write_text(
        json_mod.dumps(result_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    cliver.output(f"Result saved: {result_path}")


def _remove_task(cliver: Cliver, name: str) -> int:
    """Remove a task and its run history, origin, state, and result files."""
    import shutil

    manager = _get_manager(cliver)
    if manager.remove_task(name):
        store = _get_store(cliver)
        deleted = store.delete_runs(name)

        # Remove per-task result directory
        results_dir = cliver.agent_profile.tasks_dir / name
        if results_dir.is_dir():
            shutil.rmtree(results_dir)

        cliver.output(f"Task '{name}' removed ({deleted} run records cleaned up).")
    else:
        cliver.output(f"Task '{name}' not found.")
    return 0


# TUI dispatch entry point


def dispatch(cliver: Cliver, args: str):
    """Manage scheduled tasks — list, create, run, remove, history."""
    parts = args.strip().split(None, 1) if args.strip() else []
    sub = parts[0] if parts else "list"  # default subcommand
    rest = parts[1] if len(parts) > 1 else ""

    if sub == "list":
        _list_tasks(cliver)
    elif sub == "run":
        if not rest:
            cliver.output("[yellow]Usage: /task run <name> [--model <model>][/yellow]")
            return
        # Parse name and optional --model
        task_name = rest.split()[0]
        model = None
        if "--model" in rest or "-m" in rest:
            parts_split = rest.split()
            for i, part in enumerate(parts_split):
                if part in ("--model", "-m") and i + 1 < len(parts_split):
                    model = parts_split[i + 1]
                    break
        _run_task(cliver, task_name, model)
    elif sub == "history":
        if not rest:
            cliver.output("[yellow]Usage: /task history <name> [--limit <n>][/yellow]")
            return
        # Parse name and optional --limit
        task_name = rest.split()[0]
        limit = 10
        if "--limit" in rest or "-n" in rest:
            parts_split = rest.split()
            for i, part in enumerate(parts_split):
                if part in ("--limit", "-n") and i + 1 < len(parts_split):
                    try:
                        limit = int(parts_split[i + 1])
                    except ValueError:
                        pass
                    break
        _task_history(cliver, task_name, limit)
    elif sub == "remove":
        if not rest:
            cliver.output("[yellow]Usage: /task remove <name>[/yellow]")
            return
        task_name = rest.strip()
        _remove_task(cliver, task_name)
    elif sub == "create":
        if not rest:
            cliver.output(
                "[yellow]Usage: /task create <name> [--prompt P] [--schedule CRON] [--run-at DATETIME][/yellow]"
            )
            return
        from shlex import split as shlex_split

        try:
            parts_split = shlex_split(rest)
        except ValueError:
            parts_split = rest.split()
        task_name = parts_split[0]
        prompt = None
        model = None
        schedule = None
        run_at = None
        workflow = None
        skills = None
        i = 1
        while i < len(parts_split):
            if parts_split[i] == "--prompt" and i + 1 < len(parts_split):
                prompt = parts_split[i + 1]
                i += 2
            elif parts_split[i] == "--model" and i + 1 < len(parts_split):
                model = parts_split[i + 1]
                i += 2
            elif parts_split[i] == "--schedule" and i + 1 < len(parts_split):
                schedule = parts_split[i + 1]
                i += 2
            elif parts_split[i] == "--run-at" and i + 1 < len(parts_split):
                run_at = parts_split[i + 1]
                i += 2
            elif parts_split[i] == "--workflow" and i + 1 < len(parts_split):
                workflow = parts_split[i + 1]
                i += 2
            elif parts_split[i] == "--skills" and i + 1 < len(parts_split):
                skills = [s.strip() for s in parts_split[i + 1].split(",")]
                i += 2
            else:
                i += 1

        if not prompt:
            prompt = cliver.ui.ask_input(f"  Enter prompt for task '{task_name}': ")
            if not prompt or not prompt.strip():
                cliver.output("[yellow]Cancelled — prompt is required.[/yellow]")
                return
            prompt = prompt.strip()

        _create_task(
            cliver,
            task_name,
            prompt,
            model=model,
            schedule=schedule,
            run_at=run_at,
            workflow=workflow,
            skills=skills,
        )
    elif sub in ("--help", "help"):
        cliver.output("Usage: /task [list|create|run|history|remove] ...")
        cliver.output("  list                — list all tasks")
        cliver.output("  create <name> [--prompt P] [--model M] [--schedule S] [--run-at DATETIME]")
        cliver.output("  run <name>          — run a task")
        cliver.output("  history <name>      — show run history")
        cliver.output("  remove <name>       — remove a task")
        cliver.output("")
        cliver.output("Note: To create tasks programmatically, use the CreateTask tool directly.")
    else:
        cliver.output(f"[yellow]Unknown: /task {sub}[/yellow]")


# Click wrappers (thin — just call logic functions)


@click.group(name="task", help="Manage and run agent tasks")
def task():
    """Task management commands."""
    pass


@task.command(name="list", help="List all tasks for the current agent")
@pass_cliver
def list_tasks(cliver: Cliver):
    _list_tasks(cliver)


@task.command(name="create", help="Create a new task")
@click.argument("name")
@click.option("--prompt", "-p", required=True, help="Prompt to send to the LLM")
@click.option("--description", "-d", default=None, help="Task description")
@click.option("--model", "-m", default=None, help="Model override for this task")
@click.option("--schedule", "-s", default=None, help="Cron expression for recurring execution")
@click.option("--run-at", default=None, help="ISO 8601 datetime for one-shot execution (e.g. 2026-04-25T14:30:00)")
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
    run_at: Optional[str],
    workflow: Optional[str],
    workflow_inputs: Optional[str],
    skills: tuple,
):
    _create_task(
        cliver,
        name,
        prompt,
        description,
        model,
        schedule,
        run_at,
        workflow,
        workflow_inputs,
        list(skills) if skills else None,
    )


@task.command(name="run", help="Run a task")
@click.argument("name")
@click.option("--model", "-m", default=None, help="Override the task's model")
@pass_cliver
def run_task(cliver: Cliver, name: str, model: Optional[str]):
    _run_task(cliver, name, model)


@task.command(name="history", help="Show run history for a task")
@click.argument("name")
@click.option("--limit", "-n", default=10, help="Number of recent runs to show")
@pass_cliver
def task_history(cliver: Cliver, name: str, limit: int):
    _task_history(cliver, name, limit)


@task.command(name="remove", help="Remove a task")
@click.argument("name")
@pass_cliver
def remove_task(cliver: Cliver, name: str):
    _remove_task(cliver, name)
