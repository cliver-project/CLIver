"""Workflow management commands."""

import logging

import click
import yaml

from cliver.cli import Cliver, pass_cliver

logger = logging.getLogger(__name__)


def _run_async(coro):
    from cliver.util import run_async

    return run_async(coro)


def _make_executor(cliver: Cliver):
    from cliver.workflow.executor import WorkflowExecutor
    from cliver.workflow.persistence import WorkflowStore

    workflows_dir = cliver.agent_profile.workflows_dir
    store = WorkflowStore(workflows_dir)
    executor = WorkflowExecutor(
        app_config=cliver.config_manager.config,
        workflows_dir=workflows_dir,
    )
    return executor, store


_HELP_TEXT = """\
Usage: /workflow <command> [args]

Commands:
  list                              List all workflows (global + project)
  show <name>                       Display workflow YAML definition
  run <name> [-i key=val ...]       Execute a workflow with optional inputs
  history [name]                    List past runs (all or filtered by name)
  status <name> <execution-id>         Show per-step status of a run
  resume <name> <execution-id>         Resume from last checkpoint
  delete <name>                     Delete workflow definition + all runs
  delete-run <name> <execution-id>     Delete a single run + artifacts
  prune [--days N]                  Clean up runs older than N days (default 300)
"""


def dispatch(cliver: Cliver, args: str) -> None:
    parts = args.strip().split(None, 1)
    if not parts:
        cliver.output(_HELP_TEXT)
        return

    subcmd = parts[0].lower()
    rest = parts[1] if len(parts) > 1 else ""

    if subcmd in ("--help", "-h", "help"):
        cliver.output(_HELP_TEXT)
        return

    handlers = {
        "list": lambda: _list_workflows(cliver),
        "show": lambda: _show_workflow(cliver, rest.strip()),
        "run": lambda: _run_workflow(cliver, rest),
        "history": lambda: _history(cliver, rest.strip() or None),
        "status": lambda: _status(cliver, rest),
        "resume": lambda: _resume(cliver, rest),
        "delete": lambda: _delete_workflow(cliver, rest.strip()),
        "delete-run": lambda: _delete_run(cliver, rest),
        "prune": lambda: _prune(cliver, rest),
    }

    handler = handlers.get(subcmd)
    if handler:
        handler()
    else:
        cliver.output(f"Unknown subcommand: {subcmd}\n")
        cliver.output(_HELP_TEXT)


def _list_workflows(cliver: Cliver):
    from cliver.workflow.persistence import WorkflowStore

    store = WorkflowStore(cliver.agent_profile.workflows_dir)
    entries = store.list_all_workflows()
    if not entries:
        cliver.output("No workflows saved.")
        return
    cliver.output("Saved workflows:\n")
    for name, source, path in entries:
        wf = WorkflowStore.load_workflow_from_file(path)
        if wf:
            desc = wf.description or "(no description)"
            steps = len(wf.steps)
            tag = f"[{source}] " if source == "project" else ""
            cliver.output(f"  {tag}{name} — {desc} ({steps} steps)")
        else:
            cliver.output(f"  {name} — error loading")


def _show_workflow(cliver: Cliver, name: str):
    from cliver.workflow.persistence import WorkflowStore

    store = WorkflowStore(cliver.agent_profile.workflows_dir)
    wf = store.load_workflow(name)
    if not wf:
        cliver.output(f"Workflow '{name}' not found.")
        return
    cliver.output(
        yaml.dump(
            wf.model_dump(exclude_none=True),
            default_flow_style=False,
            sort_keys=False,
        )
    )


def _run_workflow(cliver: Cliver, rest: str):
    parts = rest.split()
    name = parts[0] if parts else ""
    if not name:
        cliver.output("Usage: /workflow run <name> [-i key=val ...]")
        return

    inputs = {}
    i = 1
    while i < len(parts):
        if parts[i] == "-i" and i + 1 < len(parts):
            kv = parts[i + 1]
            if "=" in kv:
                k, v = kv.split("=", 1)
                inputs[k.strip()] = v.strip()
            i += 2
        else:
            i += 1

    executor, store = _make_executor(cliver)
    wf = store.load_workflow(name)
    if not wf:
        cliver.output(f"Workflow '{name}' not found.")
        return

    cliver.output(f"Running workflow: {name}")
    if wf.description:
        cliver.output(f"  {wf.description}")
    cliver.output("")

    try:
        result = _run_async(executor.execute(wf, inputs=inputs))
        _display_result(cliver, result)
    except KeyboardInterrupt:
        cliver.output("\nWorkflow interrupted. Checkpoint saved.")
    except Exception as e:
        cliver.output(f"Workflow failed: {e}")


def _history(cliver: Cliver, name: str = None):
    executor, _ = _make_executor(cliver)
    execs = _run_async(executor.get_history(name))
    if not execs:
        cliver.output("No execution history found.")
        return
    for e in execs:
        status = e.get("status", "?")
        started = e.get("started_at", "?")
        thread_id = e.get("thread_id", "?")
        wf_name = e.get("workflow_name", "?")
        cliver.output(f"  [{status}] {wf_name} — {thread_id} (started {started})")


def _status(cliver: Cliver, rest: str):
    parts = rest.strip().split()
    if len(parts) < 2:
        cliver.output("Usage: /workflow status <workflow-name> <execution-id>")
        return
    wf_name, thread_id = parts[0], parts[1]
    executor, _ = _make_executor(cliver)
    status = _run_async(executor.get_status(thread_id, wf_name))
    if not status:
        cliver.output(f"No status found for {thread_id}")
        return
    cliver.output(f"Status: {status.get('status')}")
    for step_id, step_status in status.get("step_statuses", {}).items():
        cliver.output(f"  {step_id}: {step_status}")


def _resume(cliver: Cliver, rest: str):
    parts = rest.strip().split()
    if len(parts) < 2:
        cliver.output("Usage: /workflow resume <workflow-name> <execution-id>")
        return
    wf_name, thread_id = parts[0], parts[1]
    executor, _ = _make_executor(cliver)
    try:
        result = _run_async(executor.resume(wf_name, thread_id))
        _display_result(cliver, result)
    except Exception as e:
        cliver.output(f"Resume failed: {e}")


def _delete_workflow(cliver: Cliver, name: str):
    if not name:
        cliver.output("Usage: /workflow delete <name>")
        return

    import shutil
    from pathlib import Path

    from cliver.workflow.persistence import WorkflowStore

    deleted = []

    store = WorkflowStore(cliver.agent_profile.workflows_dir)
    if store.delete_workflow(name):
        deleted.append(f"  definition: {store.workflows_dir / f'{name}.yaml'}")

    project_path = Path.cwd() / ".cliver" / "workflows" / f"{name}.yaml"
    if project_path.exists():
        project_path.unlink()
        deleted.append(f"  definition: {project_path}")

    executor, _ = _make_executor(cliver)
    runs_dir = executor._runs_dir / name
    if runs_dir.exists():
        run_count = sum(1 for d in runs_dir.iterdir() if d.is_dir() and d.name != "checkpoints")
        shutil.rmtree(runs_dir)
        deleted.append(f"  runs: {run_count} execution(s) + checkpoints")

    if deleted:
        cliver.output(f"Deleted workflow '{name}':")
        for line in deleted:
            cliver.output(line)
    else:
        cliver.output(f"Workflow '{name}' not found.")


def _delete_run(cliver: Cliver, rest: str):
    parts = rest.strip().split()
    if len(parts) < 2:
        cliver.output("Usage: /workflow delete-run <workflow-name> <execution-id>")
        return
    wf_name, thread_id = parts[0], parts[1]
    executor, _ = _make_executor(cliver)
    _run_async(executor.delete_run(thread_id, wf_name))
    cliver.output(f"Run '{thread_id}' deleted.")


def _prune(cliver: Cliver, rest: str):
    days = 300
    parts = rest.strip().split()
    for i, p in enumerate(parts):
        if p == "--days" and i + 1 < len(parts):
            days = int(parts[i + 1])
    executor, _ = _make_executor(cliver)
    count = _run_async(executor.prune(days))
    cliver.output(f"Pruned {count} old execution(s).")


# ---------------------------------------------------------------------------
# Click commands (thin wrappers for CLI `cliver workflow ...`)
# ---------------------------------------------------------------------------


@click.group(
    name="workflow",
    help="Manage and run workflows (multi-step LLM + Python pipelines)",
    invoke_without_command=True,
)
@pass_cliver
@click.pass_context
def workflow_cmd(ctx, cliver: Cliver):
    if ctx.invoked_subcommand is None:
        _list_workflows(cliver)


@workflow_cmd.command(name="list", help="List all workflows (global + project)")
@pass_cliver
def list_workflows(cliver: Cliver):
    _list_workflows(cliver)


@workflow_cmd.command(name="show", help="Display workflow YAML definition")
@click.argument("name")
@pass_cliver
def show_workflow(cliver: Cliver, name: str):
    _show_workflow(cliver, name)


@workflow_cmd.command(name="run", help="Execute a workflow with optional inputs")
@click.argument("name")
@click.option("-i", "--input", "input_pairs", multiple=True, help="Input as key=value (repeatable)")
@pass_cliver
def run_workflow(cliver: Cliver, name: str, input_pairs: tuple):
    rest = name
    for kv in input_pairs:
        rest += f" -i {kv}"
    _run_workflow(cliver, rest)


@workflow_cmd.command(name="history", help="List past execution runs")
@click.argument("name", required=False, default=None)
@pass_cliver
def history(cliver: Cliver, name: str):
    _history(cliver, name)


@workflow_cmd.command(name="status", help="Show per-step status of a run")
@click.argument("name")
@click.argument("execution_id")
@pass_cliver
def status(cliver: Cliver, name: str, execution_id: str):
    _status(cliver, f"{name} {execution_id}")


@workflow_cmd.command(name="resume", help="Resume from last checkpoint")
@click.argument("name")
@click.argument("execution_id")
@pass_cliver
def resume(cliver: Cliver, name: str, execution_id: str):
    _resume(cliver, f"{name} {execution_id}")


@workflow_cmd.command(name="delete", help="Delete workflow definition + all runs")
@click.argument("name")
@pass_cliver
def delete_workflow(cliver: Cliver, name: str):
    _delete_workflow(cliver, name)


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def _display_result(cliver: Cliver, result: dict):
    if not result:
        cliver.output("No result returned.")
        return

    steps = result.get("steps", {})
    cliver.output(f"\nCompleted — {len(steps)} step(s):")
    for step_id, output in steps.items():
        if isinstance(output, dict) and "error" in output:
            cliver.output(f"  {step_id}: FAILED — {output['error']}")
        else:
            files = output.get("files", []) if isinstance(output, dict) else []
            file_info = f" ({len(files)} files)" if files else ""
            cliver.output(f"  {step_id}: OK{file_info}")

    outputs_dir = result.get("outputs_dir")
    if outputs_dir:
        cliver.output(f"\nOutputs: {outputs_dir}")
