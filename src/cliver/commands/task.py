"""
Task management commands for CLIver CLI.
"""

import uuid
from typing import Optional

import click

from cliver.cli import Cliver, pass_cliver
from cliver.commands import click_help, wants_help
from cliver.gateway.task_store import TaskStore
from cliver.task_manager import TaskDefinition, TaskManager, TaskRun, resolve_task_prompt

# Business logic (plain functions — no Click, no async)


def _get_store(cliver: Cliver) -> TaskStore:
    return TaskStore(cliver.agent_profile.gateway_db)


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
        agent = f"  agent: {t.agent}" if t.agent else ""
        ctx = f"  context: {t.context}" if t.context else ""
        schedule = f"  schedule: {t.schedule}" if t.schedule else ""
        workflow = f"  workflow: {t.workflow}" if t.workflow else ""
        skills = f"  skills: {', '.join(t.skills)}" if t.skills else ""
        desc = f"  — {t.description}" if t.description else ""
        task_origin = store.get_origin(t.name)
        origin = ""
        if task_origin:
            origin = f"  origin: {task_origin.source}"
            if task_origin.channel_id:
                origin += f" ({task_origin.channel_id})"
        cliver.output(f"  {t.name}:{desc}{agent}{ctx}{workflow}{skills}{schedule}{origin}")

    return 0


def _create_task(
    cliver: Cliver,
    name: str,
    prompt: str,
    description: Optional[str] = None,
    agent: Optional[str] = None,
    context: Optional[str] = None,
    schedule: Optional[str] = None,
    run_at: Optional[str] = None,
    workflow: Optional[str] = None,
    workflow_inputs: Optional[str] = None,
    skills: Optional[list] = None,
    reply_to: Optional[str] = None,
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
            cliver.output(f"Invalid datetime format: {run_at}")
            cliver.output("Use ISO 8601 format, e.g. 2026-04-25T14:30:00")
            return 1

    # Capture current CLI session if available
    session_id = getattr(cliver, "current_session_id", None)

    task_def = TaskDefinition(
        name=name,
        description=description,
        prompt=prompt,
        agent=agent,
        context=context,
        workflow=workflow,
        workflow_inputs=parsed_inputs,
        skills=skills,
        schedule=schedule,
        run_at=run_at,
        session_id=session_id,
    )

    manager = _get_manager(cliver)
    path = manager.save_task(task_def)

    # Save IM origin from --reply-to so task results route back
    origin_info = ""
    if reply_to:
        origin_info = _save_reply_to_origin(cliver, name, reply_to)

    cliver.output(f"Task '{name}' created: {path}{origin_info}")
    return 0


def _save_reply_to_origin(cliver: Cliver, task_name: str, reply_to: str) -> str:
    """Parse ``--reply-to platform:channel:thread`` and save as TaskOrigin."""
    from cliver.task_manager import TaskOrigin

    parts = reply_to.split(":", 2)
    if len(parts) < 2:
        cliver.output(f"Warning: invalid --reply-to format '{reply_to}', expected platform:channel[:thread]")
        return ""

    platform = parts[0]
    channel_id = parts[1]
    thread_id = parts[2] if len(parts) > 2 else None

    origin = TaskOrigin(
        source=platform,
        platform=platform,
        channel_id=channel_id,
        thread_id=thread_id,
    )
    store = _get_store(cliver)
    store.save_origin(task_name, origin)
    return f" (reply-back: {platform})"


def _run_task(cliver: Cliver, name: str) -> int:
    """Run a task by sending its prompt to the LLM."""
    manager = _get_manager(cliver)
    task_def = manager.get_task(name)
    if not task_def:
        cliver.output(f"Task '{name}' not found.")
        return 1

    execution_id = str(uuid.uuid4())[:8]
    started_at = TaskManager.timestamp_now()
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
        from cliver.cli_llm_call import LLMCallOptions, llm_call

        effective_prompt = resolve_task_prompt(task_def)
        llm_result = llm_call(
            cliver,
            LLMCallOptions(
                user_input=effective_prompt,
                agent_name=task_def.agent,
            ),
        )

        response_text = llm_result.text if llm_result.success else None
        status = "completed" if response_text else "failed"
        error = llm_result.error if not llm_result.success else (None if response_text else "No result returned")

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

    tasks_dir = cliver.agent_profile.tasks_dir
    cliver.output(f"Run history for '{name}' (most recent first):")
    for run in runs:
        status_icon = "+" if run.status == "completed" else "x"
        duration = ""
        if run.started_at and run.finished_at:
            try:
                from datetime import datetime

                s = datetime.fromisoformat(run.started_at)
                f = datetime.fromisoformat(run.finished_at)
                secs = (f - s).total_seconds()
                duration = f"  ({secs:.1f}s)" if secs < 60 else f"  ({secs / 60:.1f}m)"
            except Exception:
                pass
        cliver.output(f"  [{status_icon}] {run.started_at}  {run.status}{duration}")
        cliver.output(f"      id: {run.execution_id}")
        if run.session_id:
            cliver.output(f"      session: {run.session_id}")
        if run.error:
            cliver.output(f"      error: {run.error}")
        if run.result:
            preview = run.result[:120] + "..." if len(run.result) > 120 else run.result
            cliver.output(f"      result: {preview}")
        result_file = tasks_dir / name / f"{name}_execution_{run.execution_id}.json"
        if result_file.exists():
            cliver.output(f"      file: {result_file}")

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


def _show_task(cliver: Cliver, name: str) -> int:
    """Display a task's YAML definition."""
    import yaml

    manager = _get_manager(cliver)
    task_def = manager.get_task(name)
    if not task_def:
        cliver.output(f"Task '{name}' not found.")
        return 1

    data = task_def.model_dump(exclude_none=True, exclude={"origin", "session_id"})
    cliver.output(yaml.dump(data, default_flow_style=False, sort_keys=False, allow_unicode=True))
    return 0


def _edit_task(cliver: Cliver, name: str) -> int:
    """Open a task's YAML file in $EDITOR, validate before saving."""
    import os
    import subprocess

    import yaml

    manager = _get_manager(cliver)
    task_def = manager.get_task(name)
    if not task_def:
        cliver.output(f"Task '{name}' not found.")
        return 1

    yaml_path = cliver.agent_profile.tasks_dir / f"{name}.yaml"
    if not yaml_path.exists():
        cliver.output(f"Task file not found: {yaml_path}")
        return 1

    original = yaml_path.read_text(encoding="utf-8")
    editor = os.environ.get("EDITOR", "vi")
    try:
        subprocess.run([editor, str(yaml_path)], check=True)
    except Exception as e:
        cliver.output(f"Editor failed: {e}")
        return 1

    edited = yaml_path.read_text(encoding="utf-8")
    if edited == original:
        cliver.output("No changes.")
        return 0

    # Validate the edited YAML
    try:
        data = yaml.safe_load(edited)
        if not isinstance(data, dict):
            raise ValueError("YAML must be a mapping")
        if not data.get("name"):
            raise ValueError("'name' is required")
        if not data.get("prompt"):
            raise ValueError("'prompt' is required")
        if data.get("run_at"):
            from datetime import datetime

            datetime.fromisoformat(data["run_at"])
        TaskDefinition(**data)
    except Exception as e:
        yaml_path.write_text(original, encoding="utf-8")
        cliver.output(f"Validation failed: {e}")
        cliver.output("Changes reverted.")
        return 1

    cliver.output(f"Task '{name}' updated: {yaml_path}")
    return 0


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

    if sub in ("--help", "-h", "help"):
        cliver.output(click_help(task, "/task"))
        return

    if sub in _SUBCOMMANDS and wants_help(rest):
        cliver.output(click_help(_SUBCOMMANDS[sub], f"/task {sub}"))
        return

    if sub == "list":
        _list_tasks(cliver)
    elif sub == "show":
        if not rest:
            cliver.output("Usage: /task show <name>")
            return
        _show_task(cliver, rest.strip())
    elif sub == "edit":
        if not rest:
            cliver.output("Usage: /task edit <name>")
            return
        _edit_task(cliver, rest.strip())
    elif sub == "run":
        if not rest:
            cliver.output(click_help(_SUBCOMMANDS["run"], "/task run"))
            return
        task_name = rest.split()[0]
        _run_task(cliver, task_name)
    elif sub == "history":
        if not rest:
            cliver.output(click_help(_SUBCOMMANDS["history"], "/task history"))
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
            cliver.output(click_help(_SUBCOMMANDS["remove"], "/task remove"))
            return
        task_name = rest.strip()
        _remove_task(cliver, task_name)
    elif sub == "create":
        if not rest:
            cliver.output(click_help(_SUBCOMMANDS["create"], "/task create"))
            return
        from shlex import split as shlex_split

        try:
            parts_split = shlex_split(rest)
        except ValueError:
            parts_split = rest.split()
        task_name = parts_split[0]
        prompt = None
        agent = None
        context = None
        schedule = None
        run_at = None
        workflow = None
        skills = None
        reply_to = None
        i = 1
        while i < len(parts_split):
            if parts_split[i] == "--prompt" and i + 1 < len(parts_split):
                prompt = parts_split[i + 1]
                i += 2
            elif parts_split[i] == "--agent" and i + 1 < len(parts_split):
                agent = parts_split[i + 1]
                i += 2
            elif parts_split[i] == "--context" and i + 1 < len(parts_split):
                context = parts_split[i + 1]
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
            elif parts_split[i] == "--reply-to" and i + 1 < len(parts_split):
                reply_to = parts_split[i + 1]
                i += 2
            else:
                i += 1

        if not prompt:
            prompt = cliver.ui.ask_input(f"  Enter prompt for task '{task_name}': ")
            if not prompt or not prompt.strip():
                cliver.output("Cancelled — prompt is required.")
                return
            prompt = prompt.strip()

        _create_task(
            cliver,
            task_name,
            prompt,
            agent=agent,
            context=context,
            schedule=schedule,
            run_at=run_at,
            workflow=workflow,
            skills=skills,
            reply_to=reply_to,
        )
    else:
        cliver.output(f"Unknown: /task {sub}")


# Click wrappers (thin — just call logic functions)


@click.group(
    name="task",
    help="Manage and run agent tasks (named prompts, optionally scheduled via gateway cron)",
)
def task():
    """Task management commands."""
    pass


@task.command(
    name="list",
    help="List all tasks for the current agent with status, schedule, and origin",
)
@pass_cliver
def list_tasks(cliver: Cliver):
    _list_tasks(cliver)


@task.command(name="create", help="Create a new task definition with prompt and optional schedule")
@click.argument("name")
@click.option(
    "--prompt",
    "-p",
    required=True,
    help="Prompt text to send to the LLM when the task runs (required)",
)
@click.option(
    "--description",
    "-d",
    default=None,
    help="Human-readable task description (optional, for display only)",
)
@click.option(
    "--agent",
    "-a",
    default=None,
    help="Agent name to run this task (null = default agent). Must match a name from 'agent list'",
)
@click.option(
    "--context",
    "-c",
    default=None,
    help="URL or file path with additional context (read and appended to prompt at execution time)",
)
@click.option(
    "--schedule",
    "-s",
    default=None,
    help="Cron expression for recurring execution (e.g. '0 9 * * 1-5'). Requires gateway daemon",
)
@click.option(
    "--run-at",
    default=None,
    help="ISO 8601 datetime for one-shot execution (e.g. '2026-04-25T14:30:00')",
)
@click.option(
    "--workflow",
    "-w",
    default=None,
    help="Workflow name to execute instead of chat. Must match a name from 'workflow list'",
)
@click.option(
    "--workflow-inputs",
    default=None,
    help='Workflow input variables as JSON (e.g. \'{"env":"staging"}\')',
)
@click.option(
    "--skills",
    multiple=True,
    help="Skill to pre-activate (repeatable, e.g. --skills brainstorm --skills write-plan)",
)
@click.option(
    "--reply-to",
    default=None,
    help="IM origin for result delivery as platform:channel[:thread]",
)
@pass_cliver
def create_task(
    cliver: Cliver,
    name: str,
    prompt: str,
    description: Optional[str],
    agent: Optional[str],
    context: Optional[str],
    schedule: Optional[str],
    run_at: Optional[str],
    workflow: Optional[str],
    workflow_inputs: Optional[str],
    skills: tuple,
    reply_to: Optional[str],
):
    _create_task(
        cliver,
        name,
        prompt,
        description,
        agent=agent,
        context=context,
        schedule=schedule,
        run_at=run_at,
        workflow=workflow,
        workflow_inputs=workflow_inputs,
        skills=list(skills) if skills else None,
        reply_to=reply_to,
    )


@task.command(name="run", help="Execute a task immediately by sending its prompt to the LLM")
@click.argument("name")
@pass_cliver
def run_task(cliver: Cliver, name: str):
    _run_task(cliver, name)


@task.command(name="history", help="Show execution history for a task, most recent first")
@click.argument("name")
@click.option("--limit", "-n", default=10, help="Maximum number of recent runs to display (default: 10)")
@pass_cliver
def task_history(cliver: Cliver, name: str, limit: int):
    _task_history(cliver, name, limit)


@task.command(name="show", help="Display a task's YAML definition")
@click.argument("name")
@pass_cliver
def show_task(cliver: Cliver, name: str):
    _show_task(cliver, name)


@task.command(name="edit", help="Open a task's YAML file in $EDITOR for editing (validates before saving)")
@click.argument("name")
@pass_cliver
def edit_task(cliver: Cliver, name: str):
    _edit_task(cliver, name)


@task.command(name="remove", help="Remove a task, its run history, origin, and result files permanently")
@click.argument("name")
@pass_cliver
def remove_task(cliver: Cliver, name: str):
    _remove_task(cliver, name)


# Subcommand lookup for dispatch help
_SUBCOMMANDS = {
    "list": list_tasks,
    "show": show_task,
    "edit": edit_task,
    "create": create_task,
    "run": run_task,
    "history": task_history,
    "remove": remove_task,
}
