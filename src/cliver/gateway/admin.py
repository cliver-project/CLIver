"""
Admin portal for CLIver gateway.

Provides a web-based admin interface with cookie-based session auth
for monitoring and managing the gateway: tasks, sessions, workflows,
skills, adapters, agent info, and configuration.

React SPA served from admin_dist/ with JSON API endpoints.

Usage:
    from cliver.gateway.admin import get_admin_routes
    routes = get_admin_routes(username="admin", password="secret", context={...})
"""

import asyncio
import base64
import functools
import hashlib
import hmac
import inspect
import json
import logging
import secrets
from pathlib import Path
from typing import Optional

from starlette.requests import Request
from starlette.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response, StreamingResponse
from starlette.routing import Route

logger = logging.getLogger(__name__)


def _find_admin_dist() -> Path:
    pkg_dist = Path(__file__).resolve().parent / "admin_dist"
    if pkg_dist.is_dir():
        return pkg_dist
    return Path(__file__).resolve().parent.parent.parent.parent / "admin" / "dist"


_ADMIN_DIST = _find_admin_dist()


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------


def _check_basic_auth(request, username: str, password: str) -> bool:
    """Decode Basic Auth header and compare credentials."""
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Basic "):
        return False
    try:
        decoded = base64.b64decode(auth[6:]).decode("utf-8")
    except Exception:
        return False
    parts = decoded.split(":", 1)
    if len(parts) != 2:
        return False
    return parts[0] == username and parts[1] == password


def _make_session_token(username: str, secret: str) -> str:
    """Create an HMAC-signed session token."""
    sig = hmac.new(secret.encode(), username.encode(), hashlib.sha256).hexdigest()
    return f"{username}:{sig}"


def _check_session_cookie(request, username: str, secret: str) -> bool:
    """Validate the session cookie."""
    cookie = request.cookies.get("cliver_session", "")
    if not cookie or ":" not in cookie:
        return False
    expected = _make_session_token(username, secret)
    return hmac.compare_digest(cookie, expected)


def _mask_secret(value: Optional[str]) -> str:
    """Mask a secret string for safe display.

    - If value contains ``{{`` and ``}}``, return as-is (template expression).
    - If <= 8 chars, return ``****``.
    - Otherwise return ``first4****last4``.
    """
    if value is None:
        return ""
    if "{{" in value and "}}" in value:
        return value
    if len(value) <= 8:
        return "****"
    return value[:4] + "****" + value[-4:]


# ---------------------------------------------------------------------------
# Data-access helpers (module-level, take ctx dict)
# ---------------------------------------------------------------------------


def _get_tasks(ctx: dict) -> list:
    """List all registered tasks (DB-first) with YAML load status."""
    try:
        from cliver.agent_profile import CliverProfile
        from cliver.gateway.task_store import TaskStore
        from cliver.task_manager import TaskManager

        config_dir = ctx.get("config_dir")
        if not config_dir:
            return []
        profile = CliverProfile(config_dir)
        store = TaskStore(profile.gateway_db)
        tm = TaskManager(profile.tasks_dir, store)
        entries = tm.list_task_entries()

        result = []
        for entry in entries:
            if entry.status == "active" and entry.definition:
                data = entry.definition.model_dump(exclude_none=True)
            else:
                data = {"name": entry.name}
            data["task_status"] = entry.status
            if entry.error:
                data["task_error"] = entry.error
            if entry.created_at:
                data["created_at"] = entry.created_at
            if entry.updated_at:
                data["updated_at"] = entry.updated_at
            state = store.get_task_state(entry.name)
            if state:
                data["live_status"] = state
            runs = store.get_runs(entry.name, limit=1)
            if runs:
                data["last_run"] = runs[0].model_dump(exclude_none=True)
            origin = store.get_origin(entry.name)
            if origin:
                data["origin"] = origin.model_dump(exclude_none=True)
            if entry.definition:
                last_run_ts = store.get_last_run_time(entry.name)
                data["schedule_info"] = _compute_schedule_info(entry.definition, last_run_ts)
            result.append(data)

        store.close()
        return result
    except Exception as e:
        logger.warning("Failed to get tasks: %s", e)
        return []


async def _run_task(ctx: dict, task_name: str) -> dict:
    """Load and fire a task via the gateway (fire-and-forget)."""
    try:
        from cliver.agent_profile import CliverProfile
        from cliver.gateway.task_store import TaskStore
        from cliver.task_manager import TaskManager

        config_dir = ctx.get("config_dir")
        gateway = ctx.get("gateway")
        if not config_dir or not gateway:
            return {"status": "error", "message": "gateway or config_dir not available"}

        profile = CliverProfile(config_dir)
        store = TaskStore(profile.gateway_db)
        tm = TaskManager(profile.tasks_dir, store)
        task = tm.get_task(task_name)
        if not task:
            return {"status": "error", "message": f"task '{task_name}' not found"}

        asyncio.create_task(gateway._run_task(task))
        return {"status": "started"}
    except Exception as e:
        logger.warning("Failed to run task: %s", e)
        return {"status": "error", "message": str(e)}


# ---------------------------------------------------------------------------
# Source-aware session helpers
# ---------------------------------------------------------------------------


def _record_inference(ctx: dict, title: str, user_text: str, assistant_text: str) -> Optional[str]:
    """Record an LLM inference as a session turn pair. Returns the session_id."""
    try:
        gw = ctx.get("gateway")
        sm = gw._session_manager if gw else None
        if not sm:
            return None
        session_id = sm.create_session(title=title)
        sm.append_turn(session_id, "user", user_text, msg_type="human")
        sm.append_turn(session_id, "assistant", assistant_text, msg_type="ai")
        return session_id
    except Exception as e:
        logger.warning("Failed to record inference session: %s", e)
        return None


async def _run_in_thread(fn, *args):
    """Run a blocking function in a thread executor to avoid blocking the event loop."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, fn, *args)


def _get_session_manager(ctx, source):
    """Get the session manager for a given source ('cli' or 'gateway')."""
    if source == "gateway":
        gw = ctx.get("gateway")
        return gw._session_manager if gw else None
    elif source == "cli":
        return ctx.get("cli_session_manager")
    return None


def _get_sessions_by_source(ctx, source):
    """List sessions from the specified source.

    For gateway sessions, extracts the platform from the session_key title
    and fetches the first user message as a display title.
    """
    sm = _get_session_manager(ctx, source)
    if not sm:
        return []
    try:
        sessions = sm.list_sessions()
        if source == "gateway":
            for s in sessions:
                title = s.get("title") or ""
                parts = title.split(":", 2)
                if len(parts) >= 2:
                    s["platform"] = parts[0]
                    turns = sm.load_turns(s["id"])
                    first_user = next((t["content"] for t in turns if t["role"] == "user"), None)
                    if first_user:
                        s["display_title"] = first_user[:80]
        return sessions
    except Exception as e:
        logger.warning("Failed to get %s sessions: %s", source, e)
        return []


def _get_session_turns(ctx, source, session_id):
    """Load turns for a specific session from the specified source."""
    sm = _get_session_manager(ctx, source)
    if not sm:
        return []
    try:
        return sm.load_turns(session_id)
    except Exception as e:
        logger.warning("Failed to get session turns: %s", e)
        return []


def _delete_session(ctx, source, session_id):
    """Delete a session from the specified source."""
    sm = _get_session_manager(ctx, source)
    if not sm:
        return False
    try:
        return sm.delete_session(session_id)
    except Exception as e:
        logger.warning("Failed to delete session: %s", e)
        return False


# ---------------------------------------------------------------------------
# Detail helpers
# ---------------------------------------------------------------------------


def _compute_schedule_info(task_def, last_run_ts: float | None) -> dict:
    """Compute scheduling metadata for a task definition."""
    from datetime import datetime, timezone

    from cliver.util import get_effective_timezone

    eff_tz = get_effective_timezone()
    info: dict = {}
    tz_name = str(eff_tz)
    if tz_name:
        info["timezone"] = tz_name

    if task_def.run_at:
        info["type"] = "one-shot"
        info["run_at"] = task_def.run_at
        try:
            scheduled = datetime.fromisoformat(task_def.run_at)
            if scheduled.tzinfo is None:
                scheduled = scheduled.replace(tzinfo=eff_tz)
            now = datetime.now(timezone.utc)
            scheduled_utc = scheduled.astimezone(timezone.utc)
            if scheduled_utc > now:
                info["next_run"] = scheduled.astimezone(eff_tz).isoformat()
                delta = scheduled_utc - now
                info["time_until"] = _fmt_timedelta(delta)
            else:
                info["status"] = "completed" if last_run_ts else "overdue"
        except ValueError:
            info["status"] = "invalid"
        return info

    if task_def.schedule:
        info["type"] = "cron"
        info["expression"] = task_def.schedule
        try:
            from croniter import croniter

            now_local = datetime.now(eff_tz)
            base_dt = datetime.fromtimestamp(last_run_ts, tz=eff_tz) if last_run_ts else now_local
            cron = croniter(task_def.schedule, base_dt)
            next_dt = cron.get_next(datetime)
            if next_dt.tzinfo is None:
                next_dt = next_dt.replace(tzinfo=eff_tz)
            info["next_run"] = next_dt.astimezone(eff_tz).isoformat()
            delta = next_dt.astimezone(timezone.utc) - datetime.now(timezone.utc)
            if delta.total_seconds() > 0:
                info["time_until"] = _fmt_timedelta(delta)

            upcoming = [next_dt.astimezone(eff_tz).isoformat()]
            for _ in range(2):
                nxt = cron.get_next(datetime)
                if nxt.tzinfo is None:
                    nxt = nxt.replace(tzinfo=eff_tz)
                upcoming.append(nxt.astimezone(eff_tz).isoformat())
            info["upcoming"] = upcoming

            info["summary"] = _describe_cron(task_def.schedule)
        except (ValueError, KeyError):
            info["status"] = "invalid"
        return info

    info["type"] = "manual"
    return info


def _fmt_timedelta(delta) -> str:
    """Format a timedelta as a human-readable string."""
    total = int(delta.total_seconds())
    if total < 0:
        return "overdue"
    if total < 60:
        return f"{total}s"
    if total < 3600:
        return f"{total // 60}m"
    if total < 86400:
        h = total // 3600
        m = (total % 3600) // 60
        return f"{h}h {m}m" if m else f"{h}h"
    d = total // 86400
    h = (total % 86400) // 3600
    return f"{d}d {h}h" if h else f"{d}d"


def _describe_cron(expr: str) -> str:
    """Generate a simple human-readable summary of a cron expression."""
    parts = expr.strip().split()
    if len(parts) != 5:
        return expr

    minute, hour, dom, month, dow = parts

    dow_names = {
        "0": "Sun",
        "1": "Mon",
        "2": "Tue",
        "3": "Wed",
        "4": "Thu",
        "5": "Fri",
        "6": "Sat",
        "7": "Sun",
    }

    if minute == "*" and hour == "*":
        return "Every minute"
    if minute.startswith("*/"):
        n = minute[2:]
        return f"Every {n} minutes"
    if hour.startswith("*/"):
        n = hour[2:]
        return f"Every {n} hours at :{minute.zfill(2)}"

    time_str = ""
    if hour != "*" and minute != "*":
        time_str = f"{hour.zfill(2)}:{minute.zfill(2)}"

    if dom == "*" and month == "*" and dow == "*":
        return f"Daily at {time_str}" if time_str else expr

    if dow != "*" and dom == "*" and month == "*":
        day_tokens = dow.replace(",", ", ")
        for k, v in dow_names.items():
            day_tokens = day_tokens.replace(k, v)
        prefix = f"At {time_str} on " if time_str else "On "
        return f"{prefix}{day_tokens}"

    if dom != "*" and month == "*" and dow == "*":
        prefix = f"At {time_str} on day " if time_str else "On day "
        return f"{prefix}{dom} of every month"

    return f"At {time_str}" if time_str else expr


def _get_task_detail(ctx, task_name):
    """Get detailed info for a single task including run history."""
    try:
        from cliver.agent_profile import CliverProfile
        from cliver.gateway.task_store import TaskStore
        from cliver.task_manager import TaskManager

        config_dir = ctx.get("config_dir")
        if not config_dir:
            return None
        profile = CliverProfile(config_dir)
        store = TaskStore(profile.gateway_db)
        tm = TaskManager(profile.tasks_dir, store)
        task_entry = tm.get_task_entry(task_name)
        if not task_entry:
            return None
        if task_entry.status == "active" and task_entry.definition:
            result = task_entry.definition.model_dump(exclude_none=True)
        else:
            result = {"name": task_entry.name}
        result["task_status"] = task_entry.status
        if task_entry.error:
            result["task_error"] = task_entry.error
        if task_entry.created_at:
            result["created_at"] = task_entry.created_at
        if task_entry.updated_at:
            result["updated_at"] = task_entry.updated_at
        state = store.get_task_state(task_name)
        if state:
            result["live_status"] = state
        session_id = store.get_session_id(task_name)
        if session_id:
            result["session_id"] = session_id
        runs = store.get_runs(task_name, limit=10)
        result["runs"] = [r.model_dump(exclude_none=True) for r in runs]
        origin = store.get_origin(task_name)
        if origin:
            result["origin"] = origin.model_dump(exclude_none=True)

        # Schedule metadata
        if task_entry.definition:
            last_run_ts = store.get_last_run_time(task_name)
            result["schedule_info"] = _compute_schedule_info(task_entry.definition, last_run_ts)

        store.close()
        return result
    except Exception as e:
        logger.warning("Failed to get task detail: %s", e)
        return None


def _get_workflow_detail(ctx, workflow_id):
    """Get detailed info for a single workflow by file stem or internal name."""
    try:
        from cliver.agent_profile import CliverProfile
        from cliver.workflow.persistence import WorkflowStore

        config_dir = ctx.get("config_dir")
        if not config_dir:
            return None
        profile = CliverProfile(config_dir)
        store = WorkflowStore(profile.workflows_dir)
        # Get default model and model list from gateway config
        default_model = None
        model_names = []
        gateway = ctx.get("gateway")
        agent_core = getattr(gateway, "_agent_core", None) if gateway else None
        if agent_core:
            default_model = agent_core.default_model
            model_names = sorted(agent_core.llm_models.keys())

        for stem, _source, path in store.list_all_workflows():
            if stem == workflow_id:
                wf = WorkflowStore.load_workflow_from_file(path)
                if wf:
                    data = wf.model_dump(exclude_none=True, mode="json")
                    data["id"] = stem
                    data["_path"] = str(path)
                    if default_model:
                        data["_default_model"] = default_model
                    data["_models"] = model_names
                    if not wf.outputs_dir:
                        # Check app config for workflow_runs_dir
                        config_runs_dir = None
                        if gateway:
                            ac = getattr(gateway, "_agent_core", None)
                            if ac and hasattr(ac, "_app_config") and ac._app_config:
                                config_runs_dir = getattr(ac._app_config, "workflow_runs_dir", None)
                        if config_runs_dir:
                            default_runs = Path(config_runs_dir) / stem
                        else:
                            default_runs = profile.config_dir / "workflow-runs" / stem
                        data["_default_outputs_dir"] = str(default_runs)
                    return data
        return None
    except Exception as e:
        logger.warning("Failed to get workflow detail: %s", e)
        return None


def _get_workflows(ctx: dict) -> list:
    """List workflows with descriptions (global + project-local)."""
    try:
        from cliver.agent_profile import CliverProfile
        from cliver.workflow.persistence import WorkflowStore

        config_dir = ctx.get("config_dir")
        if not config_dir:
            return []
        profile = CliverProfile(config_dir)
        store = WorkflowStore(profile.workflows_dir)
        entries = store.list_all_workflows()
        result = []
        for stem, source, path in entries:
            wf = WorkflowStore.load_workflow_from_file(path)
            if wf:
                entry = {
                    "id": stem,
                    "name": wf.name,
                    "description": wf.description or "",
                    "source": source,
                }
                if hasattr(wf, "steps") and wf.steps:
                    entry["steps"] = len(wf.steps)
                result.append(entry)
        return result
    except Exception as e:
        logger.warning("Failed to get workflows: %s", e)
        return []


def _get_workflow_executions(ctx: dict, workflow_id: str = None) -> list:
    """List workflow executions from per-workflow databases."""
    try:
        from cliver.workflow.executor import WorkflowExecutor

        gateway = ctx.get("gateway")
        if not gateway:
            return []
        config_manager = gateway._get_config_manager()
        executor = WorkflowExecutor(app_config=config_manager.config)
        import asyncio

        return asyncio.get_event_loop().run_until_complete(executor.get_history(workflow_id))
    except RuntimeError:
        import asyncio

        return asyncio.run(_get_workflow_executions_async(ctx, workflow_id))
    except Exception as e:
        logger.warning("Failed to get workflow executions: %s", e)
        return []


async def _get_workflow_executions_async(ctx: dict, workflow_id: str = None) -> list:
    from cliver.workflow.executor import WorkflowExecutor

    gateway = ctx.get("gateway")
    if not gateway:
        return []
    config_manager = gateway._get_config_manager()
    executor = WorkflowExecutor(app_config=config_manager.config)
    return await executor.get_history(workflow_id)


async def _get_execution_status(ctx: dict, workflow_id: str, thread_id: str) -> Optional[dict]:
    """Get step-by-step status of a specific workflow execution."""
    try:
        from cliver.agent_profile import CliverProfile
        from cliver.workflow.executor import WorkflowExecutor
        from cliver.workflow.persistence import WorkflowStore

        config_dir = ctx.get("config_dir")
        if not config_dir:
            return None
        profile = CliverProfile(config_dir)
        gateway = ctx.get("gateway")
        if not gateway:
            return None

        store = WorkflowStore(profile.workflows_dir)

        # Resolve file stem to internal workflow name
        internal_name = workflow_id
        for stem, _source, path in store.list_all_workflows():
            if stem == workflow_id:
                wf = WorkflowStore.load_workflow_from_file(path)
                if wf:
                    internal_name = wf.name
                break

        config_manager = gateway._get_config_manager()
        executor = WorkflowExecutor(
            app_config=config_manager.config,
            workflows_dir=profile.workflows_dir,
        )
        return await executor.get_status(thread_id, internal_name)
    except Exception as e:
        logger.warning("Failed to get execution status: %s", e)
        return None


def _save_workflow(ctx: dict, workflow_id: str, data: dict) -> Optional[str]:
    """Save workflow changes back to the YAML file.

    Returns None on success, or an error message string.
    """
    try:
        import yaml as _yaml

        from cliver.agent_profile import CliverProfile
        from cliver.workflow.persistence import WorkflowStore
        from cliver.workflow.workflow_models import Workflow

        config_dir = ctx.get("config_dir")
        if not config_dir:
            return "No config dir"
        profile = CliverProfile(config_dir)
        store = WorkflowStore(profile.workflows_dir)

        # Find the file path
        file_path = None
        for stem, _source, path in store.list_all_workflows():
            if stem == workflow_id:
                file_path = path
                break
        if not file_path:
            return f"Workflow '{workflow_id}' not found"

        # Strip internal metadata fields before saving
        clean = {k: v for k, v in data.items() if not k.startswith("_") and k != "id"}

        # Validate through the Pydantic model
        try:
            wf = Workflow(**clean)
        except Exception as e:
            return f"Validation error: {e}"

        # Validate semantics + compilation
        from cliver.tools.workflow_validate import _compile_check, _semantic_checks

        errors = _semantic_checks(wf)
        if errors:
            return "Validation errors:\n" + "\n".join(f"- {e}" for e in errors)
        compile_errors = _compile_check(wf)
        if compile_errors:
            return "Compilation errors:\n" + "\n".join(f"- {e}" for e in compile_errors)

        # Write back to YAML
        dump_data = wf.model_dump(exclude_none=True, mode="json")
        with open(file_path, "w", encoding="utf-8") as f:
            _yaml.dump(dump_data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

        return None
    except Exception as e:
        logger.warning("Failed to save workflow: %s", e)
        return str(e)


def _get_skills() -> list:
    """List all discovered skills with full detail."""
    try:
        from cliver.skill_manager import SkillManager

        manager = SkillManager()
        skills = manager.list_skills()
        return [
            {
                "name": s.name,
                "description": s.description,
                "source": s.source,
                "path": str(s.base_dir),
                "allowed_tools": s.allowed_tools,
                "body": s.body or "",
            }
            for s in skills
        ]
    except Exception as e:
        logger.warning("Failed to get skills: %s", e)
        return []


def _get_adapters(ctx: dict) -> list:
    """Get adapter statuses merged with platform config (secrets masked)."""
    try:
        from cliver.config import ConfigManager

        gateway = ctx.get("gateway")
        statuses = {}
        if gateway and gateway._adapter_manager:
            for s in gateway._adapter_manager.platform_statuses:
                statuses[s["name"]] = s

        config_dir = ctx.get("config_dir")
        if not config_dir:
            return list(statuses.values())

        cm = ConfigManager(config_dir)
        gw_cfg = cm.config.gateway
        if not gw_cfg or not gw_cfg.platforms:
            return list(statuses.values())

        result = []
        for name, pc in gw_cfg.platforms.items():
            entry = statuses.get(name) or statuses.get(pc.type) or {"name": name, "state": "not loaded", "error": ""}
            entry["type"] = pc.type
            entry["token"] = _mask_secret(pc.token)
            if pc.app_token:
                entry["app_token"] = _mask_secret(pc.app_token)
            entry["home_channel"] = pc.home_channel or ""
            entry["allowed_users"] = pc.allowed_users or []
            extra = {}
            if hasattr(pc, "model_extra") and pc.model_extra:
                for k, v in pc.model_extra.items():
                    if isinstance(v, str) and ("token" in k.lower() or "secret" in k.lower() or "key" in k.lower()):
                        extra[k] = _mask_secret(v)
                    else:
                        extra[k] = v
            if extra:
                entry["extra"] = extra
            result.append(entry)
        return result
    except Exception as e:
        logger.warning("Failed to get adapters: %s", e)
        return []


def _get_agent_info(ctx: dict) -> dict:
    """Get agent name, identity, and memory."""
    try:
        from cliver.agent_profile import CliverProfile

        agent_name = ctx.get("agent_name", "CLIver")
        config_dir = ctx.get("config_dir")
        info = {"name": agent_name, "identity": "", "memory": ""}

        if config_dir:
            profile = CliverProfile(config_dir)
            if profile.identity_file.exists():
                content = profile.identity_file.read_text(encoding="utf-8")
                info["identity"] = content[:2000]
            if profile.memory_file.exists():
                content = profile.memory_file.read_text(encoding="utf-8")
                info["memory"] = content[:2000]

        return info
    except Exception as e:
        logger.warning("Failed to get agent info: %s", e)
        return {"name": ctx.get("agent_name", "CLIver"), "identity": "", "memory": ""}


def _get_config(ctx: dict) -> dict:
    """Get full config with secrets masked."""
    try:
        from cliver.config import ConfigManager

        config_dir = ctx.get("config_dir")
        if not config_dir:
            return {}

        cm = ConfigManager(config_dir)
        cfg = cm.config

        # Providers with nested models
        providers = {}
        for name, pc in cfg.providers.items():
            prov: dict = {
                "type": pc.type,
                "api_url": pc.api_url,
                "api_key": _mask_secret(pc.api_key),
            }
            if pc.rate_limit:
                prov["rate_limit"] = {
                    "requests": pc.rate_limit.requests,
                    "period": pc.rate_limit.period,
                    "margin": pc.rate_limit.margin,
                }
            if pc.pricing:
                prov["pricing"] = {
                    k: v
                    for k, v in {
                        "currency": pc.pricing.currency,
                        "input": pc.pricing.input,
                        "output": pc.pricing.output,
                        "cached_input": pc.pricing.cached_input,
                    }.items()
                    if v is not None
                }
            for f in ("image_url", "image_model", "audio_url", "audio_model"):
                val = getattr(pc, f, None)
                if val:
                    prov[f] = val

            models = []
            for mc in cfg.models.values():
                if mc.provider != name:
                    continue
                m: dict = {"name": mc.api_model_name}
                if mc.options:
                    opts = {k: v for k, v in mc.options.model_dump().items() if v is not None}
                    if opts:
                        m["options"] = opts
                if mc.think_mode is not None:
                    m["think_mode"] = mc.think_mode
                if mc.context_window is not None:
                    m["context_window"] = mc.context_window
                if mc.pricing:
                    m["pricing"] = {
                        k: v
                        for k, v in {
                            "currency": mc.pricing.currency,
                            "input": mc.pricing.input,
                            "output": mc.pricing.output,
                            "cached_input": mc.pricing.cached_input,
                        }.items()
                        if v is not None
                    }
                if mc.capabilities:
                    m["capabilities"] = sorted(c.value if hasattr(c, "value") else str(c) for c in mc.capabilities)
                models.append(m)
            prov["models"] = models
            providers[name] = prov

        # MCP servers
        mcp_servers = {}
        for name, sc in cfg.mcpServers.items():
            entry: dict = {"transport": sc.transport}
            if hasattr(sc, "command"):
                entry["command"] = sc.command
            if hasattr(sc, "args") and sc.args:
                entry["args"] = sc.args
            if hasattr(sc, "env") and sc.env:
                entry["env"] = sc.env
            if hasattr(sc, "url"):
                entry["url"] = sc.url
            if hasattr(sc, "headers") and sc.headers:
                entry["headers"] = sc.headers
            mcp_servers[name] = entry

        # Gateway
        gateway = None
        if cfg.gateway:
            gw = cfg.gateway
            gateway = {
                "host": gw.host,
                "port": gw.port,
                "admin_username": gw.admin_username or "",
                "admin_password": _mask_secret(gw.admin_password),
                "log_file": gw.log_file or "",
                "log_max_bytes": gw.log_max_bytes,
                "log_backup_count": gw.log_backup_count,
            }
            if gw.api_key:
                gateway["api_key"] = _mask_secret(gw.api_key)
            platforms = {}
            for pname, pc in gw.platforms.items():
                plat: dict = {"type": pc.type}
                plat["token"] = _mask_secret(pc.token)
                plat["app_token"] = _mask_secret(pc.app_token)
                if pc.home_channel:
                    plat["home_channel"] = pc.home_channel
                if pc.allowed_users:
                    plat["allowed_users"] = pc.allowed_users
                platforms[pname] = plat
            gateway["platforms"] = platforms

        # Session
        session = {
            "max_sessions": cfg.session.max_sessions,
            "max_turns_per_session": cfg.session.max_turns_per_session,
            "max_age_days": cfg.session.max_age_days,
        }

        from cliver.gateway.adapters import BUILTIN_ADAPTERS

        agents = {}
        for aname, ac in cfg.agents.items():
            agents[aname] = ac.model_dump(exclude_none=True)

        return {
            "default_model": cfg.default_model or "",
            "default_agent": cfg.default_agent or "",
            "user_agent": cfg.user_agent or "",
            "timezone": cfg.timezone or "",
            "search_engines": cfg.search_engines or [],
            "skill_auto_learn": cfg.skill_auto_learn,
            "model_auto_fallback": cfg.model_auto_fallback,
            "workflow_runs_dir": cfg.workflow_runs_dir or "",
            "default_workflow_runs_dir": str(config_dir / "workflow-runs"),
            "adapter_types": sorted(BUILTIN_ADAPTERS.keys()),
            "agents": agents,
            "providers": providers,
            "mcpServers": mcp_servers,
            "gateway": gateway,
            "session": session,
        }
    except Exception as e:
        logger.warning("Failed to get config: %s", e)
        return {}


def _is_masked(value: str) -> bool:
    """Return True if value looks like a masked secret (contains ****)."""
    return "****" in value


def _save_config_data(ctx: dict, data: dict) -> Optional[str]:
    """Apply config changes from the admin UI and save to config.yaml.

    Returns an error message string, or None on success.
    Masked secret fields (containing ****) are skipped to preserve existing values.
    """
    try:
        from cliver.config import (
            ConfigManager,
            GatewayConfig,
            ModelConfig,
            ModelOptions,
            PlatformConfig,
            PricingConfig,
            RateLimitConfig,
            SessionConfig,
        )

        config_dir = ctx.get("config_dir")
        if not config_dir:
            return "No config directory"

        cm = ConfigManager(config_dir)
        cfg = cm.config

        # General settings
        if "default_model" in data:
            cfg.default_model = data["default_model"] or None
        if "user_agent" in data:
            cfg.user_agent = data["user_agent"] or None
        if "timezone" in data:
            cfg.timezone = data["timezone"] or None
        if "search_engines" in data:
            val = data["search_engines"]
            cfg.search_engines = val if val else None
        if "skill_auto_learn" in data:
            cfg.skill_auto_learn = bool(data["skill_auto_learn"])
        if "model_auto_fallback" in data:
            cfg.model_auto_fallback = bool(data["model_auto_fallback"])
        if "workflow_runs_dir" in data:
            cfg.workflow_runs_dir = data["workflow_runs_dir"] or None
        if "default_agent" in data:
            cfg.default_agent = data["default_agent"] or None

        # Agents
        if "agents" in data:
            from cliver.config import AgentConfig

            incoming = data["agents"]
            removed = set(cfg.agents.keys()) - set(incoming.keys())
            for name in removed:
                cfg.agents.pop(name, None)
            for aname, adata in incoming.items():
                cfg.agents[aname] = AgentConfig(
                    description=adata.get("description") or None,
                    role=adata.get("role") or None,
                    system_prompt=adata.get("system_prompt") or None,
                    model=adata.get("model") or None,
                    skills=adata.get("skills") or [],
                )

        # Providers & models
        if "providers" in data:
            incoming_providers = data["providers"]
            removed = set(cfg.providers.keys()) - set(incoming_providers.keys())
            for name in removed:
                refs = [m.name for m in cfg.models.values() if m.provider == name]
                for ref in refs:
                    cfg.models.pop(ref, None)
                cfg.providers.pop(name, None)

            for pname, pdata in incoming_providers.items():
                api_key_val = pdata.get("api_key")
                if api_key_val and _is_masked(api_key_val):
                    api_key_val = None  # skip, keep existing

                rl = None
                if pdata.get("rate_limit"):
                    rld = pdata["rate_limit"]
                    rl = RateLimitConfig(
                        requests=rld.get("requests", 100),
                        period=rld.get("period", "1h"),
                        margin=rld.get("margin", 0.1),
                    )

                pricing = None
                if pdata.get("pricing"):
                    pricing = PricingConfig(**pdata["pricing"])

                cm.add_or_update_provider(
                    name=pname,
                    type=pdata.get("type", "openai"),
                    api_url=pdata.get("api_url", ""),
                    api_key=api_key_val,
                    rate_limit=rl,
                    image_url=pdata.get("image_url"),
                    image_model=pdata.get("image_model"),
                    audio_url=pdata.get("audio_url"),
                    audio_model=pdata.get("audio_model"),
                )
                if pricing and pname in cfg.providers:
                    cfg.providers[pname].pricing = pricing

                # Models under this provider
                incoming_models = pdata.get("models", [])
                existing_model_names = {m.name for m in cfg.models.values() if m.provider == pname}
                incoming_model_names = {f"{pname}/{md['name']}" for md in incoming_models if md.get("name")}
                for gone in existing_model_names - incoming_model_names:
                    cfg.models.pop(gone, None)

                for md in incoming_models:
                    mname = md.get("name", "")
                    if not mname:
                        continue
                    canonical = f"{pname}/{mname}"
                    opts = None
                    if md.get("options"):
                        opts = ModelOptions(**md["options"])
                    m_pricing = None
                    if md.get("pricing"):
                        m_pricing = PricingConfig(**md["pricing"])
                    caps = None
                    if md.get("capabilities"):
                        caps = set(md["capabilities"])

                    mc = cfg.models.get(canonical)
                    if mc:
                        mc.options = opts
                        mc.pricing = m_pricing
                        mc.think_mode = md.get("think_mode")
                        mc.context_window = md.get("context_window")
                        if caps is not None:
                            mc.capabilities = caps
                    else:
                        cfg.models[canonical] = ModelConfig(
                            name=canonical,
                            provider=pname,
                            options=opts,
                            pricing=m_pricing,
                            think_mode=md.get("think_mode"),
                            context_window=md.get("context_window"),
                            capabilities=caps,
                        )
                    if pname in cfg.providers:
                        cfg.models[canonical]._provider_config = cfg.providers[pname]

        # MCP servers
        if "mcpServers" in data:
            incoming = data["mcpServers"]
            removed = set(cfg.mcpServers.keys()) - set(incoming.keys())
            for name in removed:
                cfg.mcpServers.pop(name, None)
            for sname, sdata in incoming.items():
                cm.add_or_update_server(sname, sdata)

        # Gateway
        if "gateway" in data and data["gateway"] is not None:
            gd = data["gateway"]
            if cfg.gateway is None:
                cfg.gateway = GatewayConfig()
            gw = cfg.gateway
            if "host" in gd:
                gw.host = gd["host"]
            if "port" in gd:
                gw.port = int(gd["port"])
            if "admin_username" in gd:
                gw.admin_username = gd["admin_username"] or None
            if "admin_password" in gd:
                pw = gd["admin_password"]
                if not _is_masked(pw):
                    gw.admin_password = pw or None
            if "log_file" in gd:
                gw.log_file = gd["log_file"] or None
            if "log_max_bytes" in gd:
                gw.log_max_bytes = int(gd["log_max_bytes"])
            if "log_backup_count" in gd:
                gw.log_backup_count = int(gd["log_backup_count"])
            if "api_key" in gd:
                ak = gd["api_key"]
                if not _is_masked(ak):
                    gw.api_key = ak or None

            if "platforms" in gd:
                incoming_plats = gd["platforms"]
                removed_plats = set(gw.platforms.keys()) - set(incoming_plats.keys())
                for pn in removed_plats:
                    gw.platforms.pop(pn, None)
                for pn, pd in incoming_plats.items():
                    existing = gw.platforms.get(pn)
                    token = pd.get("token", "")
                    app_token = pd.get("app_token", "")
                    if existing:
                        existing.type = pd.get("type", existing.type)
                        if not _is_masked(token):
                            existing.token = token or None
                        if not _is_masked(app_token):
                            existing.app_token = app_token or None
                        existing.home_channel = pd.get("home_channel") or None
                        existing.allowed_users = pd.get("allowed_users") or None
                    else:
                        gw.platforms[pn] = PlatformConfig(
                            name=pn,
                            type=pd.get("type", ""),
                            token=None if _is_masked(token) else (token or None),
                            app_token=None if _is_masked(app_token) else (app_token or None),
                            home_channel=pd.get("home_channel") or None,
                            allowed_users=pd.get("allowed_users") or None,
                        )

        # Session
        if "session" in data:
            sd = data["session"]
            cfg.session = SessionConfig(
                max_sessions=int(sd.get("max_sessions", 300)),
                max_turns_per_session=int(sd.get("max_turns_per_session", 100)),
                max_age_days=int(sd.get("max_age_days", 365)),
            )

        cm._save_config()
        return None
    except Exception as e:
        logger.warning("Failed to save config: %s", e)
        return str(e)


# ---------------------------------------------------------------------------
# Route registration
# ---------------------------------------------------------------------------


def get_admin_routes(
    username: Optional[str],
    password: Optional[str],
    context: dict,
) -> list:
    """Return admin portal routes as a list of Starlette Route objects.

    Args:
        username: Admin username (None = portal disabled)
        password: Admin password (None = portal disabled)
        context: Dict with keys:
            - get_status: async callable or dict returning status info
            - agent_name: str
            - config_dir: Path or None
            - gateway: Gateway instance (optional)
            - cli_session_manager: SessionManager for CLI sessions (optional)

    Returns:
        list of starlette.routing.Route
    """
    session_secret = secrets.token_hex(32)

    # --- Auth decorator (cookie-first, Basic Auth fallback) ---

    def require_auth(handler):
        @functools.wraps(handler)
        async def wrapper(request: Request):
            if username is None or password is None:
                return JSONResponse(
                    {"error": "Admin portal is disabled"},
                    status_code=403,
                )
            if _check_session_cookie(request, username, session_secret):
                return await handler(request)
            if _check_basic_auth(request, username, password):
                return await handler(request)
            is_api = request.url.path.startswith("/admin/api/")
            if is_api:
                return JSONResponse({"error": "Unauthorized"}, status_code=401)
            return RedirectResponse("/admin/login", status_code=302)

        return wrapper

    # --- Route handlers (closures capturing context) ---

    async def handle_spa(request: Request):
        """Serve the SPA index.html for all non-API routes."""
        index_path = _ADMIN_DIST / "index.html"
        if not index_path.exists():
            return HTMLResponse(
                "<h1>Admin UI not built. Run: make admin-build</h1>",
                status_code=503,
            )
        return FileResponse(str(index_path))

    async def handle_spa_assets(request: Request):
        """Serve static assets (JS, CSS, etc.) from the SPA dist directory."""
        import mimetypes as _mt

        file_path = request.path_params["path"]
        full_path = _ADMIN_DIST / "assets" / file_path
        # Prevent path traversal
        try:
            full_path.resolve().relative_to(_ADMIN_DIST.resolve())
        except ValueError:
            return Response("Forbidden", status_code=403)
        if not full_path.exists() or not full_path.is_file():
            return Response("Not found", status_code=404)
        content_type = _mt.guess_type(str(full_path))[0] or "application/octet-stream"
        return Response(full_path.read_bytes(), media_type=content_type)

    async def handle_login_submit(request: Request):
        try:
            data = await request.json()
        except Exception:
            return JSONResponse({"error": "Invalid request"}, status_code=400)
        u = data.get("username", "")
        p = data.get("password", "")
        if u == username and p == password:
            token = _make_session_token(username, session_secret)
            resp = JSONResponse({"status": "ok"})
            resp.set_cookie("cliver_session", token, httponly=True, samesite="lax", path="/admin")
            return resp
        return JSONResponse({"error": "Invalid credentials"}, status_code=401)

    async def handle_logout(request: Request):
        resp = RedirectResponse("/admin/login", status_code=302)
        resp.delete_cookie("cliver_session", path="/admin")
        return resp

    @require_auth
    async def handle_status(request: Request):
        get_status = context.get("get_status")
        if get_status is None:
            return JSONResponse({})
        if callable(get_status):
            result = get_status()
            if inspect.isawaitable(result):
                result = await result
            return JSONResponse(result)
        return JSONResponse(get_status)

    @require_auth
    async def handle_tasks(request: Request):
        tasks = await _run_in_thread(_get_tasks, context)
        return JSONResponse(tasks)

    @require_auth
    async def handle_task_detail_api(request: Request):
        name = request.path_params["name"]
        detail = await _run_in_thread(_get_task_detail, context, name)
        return JSONResponse(detail)

    @require_auth
    async def handle_run_task(request: Request):
        task_name = request.path_params["name"]
        logger.info("[admin] Task '%s' triggered via admin portal", task_name)
        result = await _run_task(context, task_name)
        status_code = 200 if result.get("status") == "started" else 400
        return JSONResponse(result, status_code=status_code)

    @require_auth
    async def handle_create_task(request: Request):
        """POST /admin/api/tasks — create a new task."""
        from cliver.agent_profile import CliverProfile
        from cliver.gateway.task_store import TaskStore
        from cliver.task_manager import TaskDefinition, TaskManager

        try:
            data = await request.json()
        except Exception:
            return JSONResponse({"error": "Invalid JSON"}, status_code=400)

        name = (data.get("name") or "").strip()
        prompt = (data.get("prompt") or "").strip()
        if not name:
            return JSONResponse({"error": "Task name is required"}, status_code=400)
        if not prompt:
            return JSONResponse({"error": "Prompt is required"}, status_code=400)

        config_dir = context.get("config_dir")
        if not config_dir:
            return JSONResponse({"error": "No config dir"}, status_code=500)

        try:
            profile = CliverProfile(config_dir)
            store = TaskStore(profile.gateway_db)
            tm = TaskManager(profile.tasks_dir, store)

            if tm.get_task(name):
                store.close()
                return JSONResponse({"error": f"Task '{name}' already exists"}, status_code=409)

            task = TaskDefinition(
                name=name,
                prompt=prompt,
                description=data.get("description") or None,
                model=data.get("model") or None,
                agent=data.get("agent") or None,
                schedule=data.get("schedule") or None,
                run_at=data.get("run_at") or None,
                skills=data.get("skills") or None,
                workflow=data.get("workflow") or None,
            )
            tm.save_task(task)
            store.close()

            logger.info("[admin] Task '%s' created via admin portal", name)
            detail = _get_task_detail(context, name)
            return JSONResponse(detail, status_code=201)
        except Exception as e:
            logger.error("Failed to create task: %s", e)
            return JSONResponse({"error": str(e)}, status_code=500)

    @require_auth
    async def handle_update_task(request: Request):
        """PUT /admin/api/tasks/{name} — update an existing task."""
        from cliver.agent_profile import CliverProfile
        from cliver.gateway.task_store import TaskStore
        from cliver.task_manager import TaskManager

        task_name = request.path_params["name"]
        try:
            data = await request.json()
        except Exception:
            return JSONResponse({"error": "Invalid JSON"}, status_code=400)

        config_dir = context.get("config_dir")
        if not config_dir:
            return JSONResponse({"error": "No config dir"}, status_code=500)

        try:
            profile = CliverProfile(config_dir)
            store = TaskStore(profile.gateway_db)
            tm = TaskManager(profile.tasks_dir, store)

            task = tm.get_task(task_name)
            if not task:
                store.close()
                return JSONResponse({"error": "Task not found"}, status_code=404)

            if "prompt" in data:
                task.prompt = data["prompt"]
            if "description" in data:
                task.description = data["description"] or None
            if "model" in data:
                task.model = data["model"] or None
            if "agent" in data:
                task.agent = data["agent"] or None
            if "schedule" in data:
                task.schedule = data["schedule"] or None
            if "run_at" in data:
                task.run_at = data["run_at"] or None
            if "skills" in data:
                task.skills = data["skills"] or None
            if "workflow" in data:
                task.workflow = data["workflow"] or None

            tm.save_task(task)
            store.close()

            logger.info("[admin] Task '%s' updated via admin portal", task_name)
            detail = _get_task_detail(context, task_name)
            return JSONResponse(detail)
        except Exception as e:
            logger.error("Failed to update task '%s': %s", task_name, e)
            return JSONResponse({"error": str(e)}, status_code=500)

    @require_auth
    async def handle_delete_task(request: Request):
        task_name = request.path_params["name"]
        logger.info("[admin] Task '%s' deleted via admin portal", task_name)
        try:
            from cliver.agent_profile import CliverProfile
            from cliver.gateway.task_store import TaskStore
            from cliver.task_manager import TaskManager

            config_dir = context.get("config_dir")
            if not config_dir:
                return JSONResponse({"error": "No config dir"}, status_code=500)

            profile = CliverProfile(config_dir)
            store = TaskStore(profile.gateway_db)
            tm = TaskManager(profile.tasks_dir, store)

            session_id = store.get_session_id(task_name)
            removed = tm.remove_task(task_name)

            deleted_runs = 0
            if removed:
                deleted_runs = store.delete_runs(task_name)
                store.delete_task_state(task_name)
                store.delete_origin(task_name)
                if session_id:
                    try:
                        from cliver.session_manager import SessionManager

                        sm = SessionManager(config_dir / "gateway-sessions")
                        sm.delete_session(session_id)
                    except Exception as e:
                        logger.warning("Failed to delete session for task '%s': %s", task_name, e)
            store.close()

            if not removed:
                return JSONResponse({"error": "Task not found"}, status_code=404)

            return JSONResponse(
                {
                    "status": "deleted",
                    "runs_removed": deleted_runs,
                }
            )
        except Exception as e:
            logger.error("Failed to delete task '%s': %s", task_name, e)
            return JSONResponse({"error": str(e)}, status_code=500)

    @require_auth
    async def handle_sessions_by_source(request: Request):
        source = request.path_params["source"]
        if source not in ("cli", "gateway"):
            return JSONResponse(
                {"error": f"Invalid source '{source}'. Use 'cli' or 'gateway'."},
                status_code=400,
            )
        sessions = await _run_in_thread(_get_sessions_by_source, context, source)
        return JSONResponse(sessions)

    @require_auth
    async def handle_session_turns(request: Request):
        source = request.path_params["source"]
        session_id = request.path_params["id"]
        if source not in ("cli", "gateway"):
            return JSONResponse(
                {"error": f"Invalid source '{source}'."},
                status_code=400,
            )
        turns = await _run_in_thread(_get_session_turns, context, source, session_id)
        return JSONResponse(turns)

    @require_auth
    async def handle_delete_session(request: Request):
        source = request.path_params["source"]
        session_id = request.path_params["id"]
        if source not in ("cli", "gateway"):
            return JSONResponse(
                {"error": f"Invalid source '{source}'."},
                status_code=400,
            )
        logger.info(
            "[admin] Session '%s' (%s) deleted via admin portal",
            session_id,
            source,
        )
        deleted = _delete_session(context, source, session_id)
        if deleted:
            return JSONResponse({"status": "deleted"})
        return JSONResponse({"error": "session not found"}, status_code=404)

    @require_auth
    async def handle_workflows(request: Request):
        workflows = _get_workflows(context)
        return JSONResponse(workflows)

    @require_auth
    async def handle_workflow_detail_api(request: Request):
        name = request.path_params["name"]
        detail = _get_workflow_detail(context, name)
        return JSONResponse(detail)

    @require_auth
    async def handle_update_workflow(request: Request):
        name = request.path_params["name"]
        try:
            data = await request.json()
        except Exception:
            return JSONResponse({"error": "Invalid JSON"}, status_code=400)
        error = _save_workflow(context, name, data)
        if error:
            return JSONResponse({"error": error}, status_code=400)
        return JSONResponse({"status": "saved"})

    @require_auth
    async def handle_workflow_executions(request: Request):
        name = request.path_params.get("name")
        execs = _get_workflow_executions(context, name)
        return JSONResponse(execs)

    @require_auth
    async def handle_all_workflow_executions(request: Request):
        execs = _get_workflow_executions(context)
        return JSONResponse(execs)

    @require_auth
    async def handle_execution_status(request: Request):
        name = request.path_params["name"]
        tid = request.path_params["tid"]
        status = await _get_execution_status(context, name, tid)
        if status is None:
            return JSONResponse({"error": "Execution not found"}, status_code=404)
        return JSONResponse(status)

    _running_workflow_tasks = {}

    @require_auth
    async def handle_run_workflow(request: Request):
        name = request.path_params["name"]
        gateway = context.get("gateway")
        if not gateway or not hasattr(gateway, "run_workflow"):
            return JSONResponse({"error": "Gateway not available"}, status_code=500)
        try:
            body = await request.json()
        except Exception:
            body = {}
        inputs = body.get("inputs")
        task = asyncio.create_task(gateway.run_workflow(name, inputs=inputs))
        _running_workflow_tasks[name] = task
        task.add_done_callback(lambda _: _running_workflow_tasks.pop(name, None))
        return JSONResponse({"status": "started", "workflow": name})

    @require_auth
    async def handle_stop_workflow(request: Request):
        name = request.path_params["name"]
        task = _running_workflow_tasks.get(name)
        if not task:
            return JSONResponse({"error": "No running execution found"}, status_code=404)
        task.cancel()
        return JSONResponse({"status": "stopped", "workflow": name})

    @require_auth
    async def handle_resume_from_step(request: Request):
        """Resume a workflow execution from a specific step.

        Uses LangGraph checkpoint history to find the state right before
        the target step, then re-runs from that point forward.

        Path params: name (workflow), step_id
        Query/body: thread_id (required — the execution to resume)
        """
        wf_name = request.path_params["name"]
        step_id = request.path_params["step_id"]

        try:
            body = await request.json()
        except Exception:
            body = {}
        thread_id = body.get("thread_id", "")
        if not thread_id:
            return JSONResponse({"error": "'thread_id' is required"}, status_code=400)

        gateway = context.get("gateway")
        if not gateway or not getattr(gateway, "_agent_core", None):
            return JSONResponse({"error": "Gateway not available"}, status_code=503)

        async def _do_resume():
            from cliver.agent_profile import CliverProfile
            from cliver.workflow.executor import WorkflowExecutor

            config_dir = context.get("config_dir")
            profile = CliverProfile(config_dir)
            config_manager = gateway._get_config_manager()

            executor = WorkflowExecutor(
                app_config=config_manager.config,
                workflows_dir=profile.workflows_dir,
            )
            try:
                result = await executor.resume(wf_name, thread_id)
                if result and isinstance(result, dict) and "error" in result:
                    logger.warning("Resume from step failed: %s", result["error"])
                else:
                    logger.info("Resumed workflow '%s' from step '%s'", wf_name, step_id)
            except Exception as e:
                logger.error("Resume failed: %s", e)

        asyncio.create_task(_do_resume())
        return JSONResponse(
            {
                "status": "started",
                "workflow": wf_name,
                "step_id": step_id,
                "thread_id": thread_id,
            }
        )

    @require_auth
    async def handle_browse_files(request: Request):
        """Browse server filesystem for file selection (restricted to home directory)."""
        home = Path.home().resolve()
        dir_path = request.query_params.get("dir", "")
        file_filter = request.query_params.get("filter", "")
        if not dir_path:
            dir_path = str(home)
        target = Path(dir_path).expanduser().resolve()

        # Restrict browsing to home directory
        try:
            target.relative_to(home)
        except ValueError:
            return JSONResponse({"error": "Access restricted to home directory"}, status_code=403)

        if not target.is_dir():
            return JSONResponse({"error": "Not a directory", "path": str(target)}, status_code=400)

        filter_exts = set()
        if file_filter == "image":
            filter_exts = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".bmp"}
        elif file_filter == "audio":
            filter_exts = {".mp3", ".wav", ".ogg", ".aac", ".flac", ".m4a"}
        elif file_filter == "video":
            filter_exts = {".mp4", ".webm", ".mov", ".avi", ".mkv"}

        items = []
        try:
            # Parent directory (only if still within home)
            parent = target.parent.resolve()
            try:
                parent.relative_to(home)
                if parent != target:
                    items.append({"name": "..", "path": str(parent), "type": "dir"})
            except ValueError:
                pass
            for entry in sorted(target.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower())):
                if entry.name.startswith("."):
                    continue
                if entry.is_dir():
                    items.append({"name": entry.name, "path": str(entry), "type": "dir"})
                elif entry.is_file():
                    if filter_exts and entry.suffix.lower() not in filter_exts:
                        continue
                    items.append({"name": entry.name, "path": str(entry), "type": "file", "size": entry.stat().st_size})
        except PermissionError:
            return JSONResponse({"error": "Permission denied", "path": str(target)}, status_code=403)

        return JSONResponse({"path": str(target), "items": items})

    @require_auth
    async def handle_delete_execution(request: Request):
        wf_name = request.path_params["name"]
        tid = request.path_params["tid"]
        try:
            from cliver.workflow.executor import WorkflowExecutor

            gateway = context.get("gateway")
            if not gateway:
                return JSONResponse({"error": "No gateway"}, status_code=500)
            config_manager = gateway._get_config_manager()
            executor = WorkflowExecutor(app_config=config_manager.config)
            deleted = await executor.delete_run(tid, wf_name)
            if deleted:
                return JSONResponse({"status": "deleted"})
            return JSONResponse({"error": "Execution not found"}, status_code=404)
        except Exception as e:
            logger.warning("Failed to delete execution: %s", e)
            return JSONResponse({"error": str(e)}, status_code=500)

    @require_auth
    async def handle_workflow_media(request: Request):
        """Serve media files from workflow output directories."""
        import mimetypes as _mt

        file_path = request.path_params["path"]
        full_path = Path(file_path)
        if not full_path.is_absolute():
            return Response("Invalid path", status_code=400)
        if not full_path.exists() or not full_path.is_file():
            return Response("Not found", status_code=404)
        # Only serve files from workflow-runs directories
        if "workflow-runs" not in str(full_path):
            return Response("Forbidden", status_code=403)
        content_type = _mt.guess_type(str(full_path))[0] or "application/octet-stream"
        return Response(full_path.read_bytes(), media_type=content_type)

    @require_auth
    async def handle_skills(request: Request):
        skills = _get_skills()
        return JSONResponse(skills)

    @require_auth
    async def handle_skill_detail_api(request: Request):
        name = request.path_params["name"]
        skills = _get_skills()
        skill = next((s for s in skills if s["name"] == name), None)
        if not skill:
            return Response("Skill not found", status_code=404)
        editable = skill.get("source", "") != "builtin"
        raw_content = ""
        skill_file = Path(skill["path"]) / "SKILL.md"
        if skill_file.is_file():
            try:
                raw_content = skill_file.read_text(encoding="utf-8")
            except Exception:
                pass
        return JSONResponse({**skill, "editable": editable, "raw_content": raw_content})

    @require_auth
    async def handle_update_skill(request: Request):
        body = await request.json()
        content = body.get("content", "")
        skill_path = body.get("path", "")
        if not skill_path:
            return Response("Missing path", status_code=400)
        skill_dir = Path(skill_path).resolve()
        skill_file = skill_dir / "SKILL.md"
        if not skill_file.is_file():
            return Response("Skill file not found", status_code=404)
        from cliver.skill_manager import _BUILTIN_SKILLS_DIR

        if str(skill_dir).startswith(str(_BUILTIN_SKILLS_DIR)):
            return Response("Cannot edit builtin skills", status_code=403)
        try:
            skill_file.write_text(content, encoding="utf-8")
        except Exception as e:
            return Response(f"Failed to save: {e}", status_code=500)
        return JSONResponse({"ok": True})

    @require_auth
    async def handle_create_skill(request: Request):
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "Invalid JSON"}, status_code=400)
        content = body.get("content", "")
        skill_name = body.get("name", "").strip()
        if not skill_name:
            return JSONResponse({"error": "Skill name is required"}, status_code=400)
        from cliver.skill_manager import validate_skill_name

        errors = validate_skill_name(skill_name)
        if errors:
            return JSONResponse({"error": "; ".join(errors)}, status_code=400)
        work_dir = context.get("work_dir") or Path.cwd()
        skills_dir = Path(work_dir) / ".cliver" / "skills" / skill_name
        if skills_dir.exists():
            return JSONResponse({"error": f"Skill '{skill_name}' already exists"}, status_code=409)
        try:
            skills_dir.mkdir(parents=True, exist_ok=True)
            skill_file = skills_dir / "SKILL.md"
            skill_file.write_text(content, encoding="utf-8")
        except Exception as e:
            return JSONResponse({"error": f"Failed to create: {e}"}, status_code=500)
        return JSONResponse({"ok": True, "name": skill_name}, status_code=201)

    @require_auth
    async def handle_adapters(request: Request):
        adapters = _get_adapters(context)
        return JSONResponse(adapters)

    @require_auth
    async def handle_agent(request: Request):
        info = _get_agent_info(context)
        return JSONResponse(info)

    @require_auth
    async def handle_agents_list(request: Request):
        """GET /admin/api/agents — list configured agents."""
        from cliver.config import ConfigManager

        config_dir = context.get("config_dir")
        if not config_dir:
            return JSONResponse([])
        cm = ConfigManager(config_dir)
        cfg = cm.config
        cfg.ensure_default_agent()
        result = []
        for name, ac in cfg.agents.items():
            entry = ac.model_dump(exclude_none=True)
            entry["name"] = name
            entry["is_default"] = name == cfg.default_agent
            result.append(entry)
        return JSONResponse(result)

    @require_auth
    async def handle_config(request: Request):
        config = _get_config(context)
        return JSONResponse(config)

    @require_auth
    async def handle_update_config(request: Request):
        """Save config changes to config.yaml."""
        try:
            data = await request.json()
        except Exception:
            return JSONResponse({"error": "Invalid JSON"}, status_code=400)
        try:
            err = _save_config_data(context, data)
            if err:
                return JSONResponse({"error": err}, status_code=400)
            return JSONResponse({"status": "saved"})
        except Exception as e:
            logger.warning("Failed to save config: %s", e)
            return JSONResponse({"error": str(e)}, status_code=500)

    # --- Provider connectivity test ---

    @require_auth
    async def handle_test_provider(request: Request):
        """POST /admin/api/providers/{provider_name}/test — send a hello to verify connectivity."""
        import asyncio

        from langchain_core.messages import HumanMessage

        from cliver.config import ConfigManager
        from cliver.llm.unified_engine import UnifiedInferenceEngine

        provider_name = request.path_params["provider_name"]
        try:
            body = await request.json()
        except Exception:
            body = {}

        config_dir = context.get("config_dir")
        if not config_dir:
            return JSONResponse({"status": "error", "error": "No config directory"}, status_code=500)

        cm = ConfigManager(config_dir)
        cfg = cm.config

        prov = cfg.providers.get(provider_name)
        if not prov:
            return JSONResponse({"status": "error", "error": f"Provider '{provider_name}' not found"}, status_code=404)

        requested_model = body.get("model")
        model_config = None
        for mc in cfg.models.values():
            if mc.provider != provider_name:
                continue
            if requested_model and mc.api_model_name != requested_model:
                continue
            model_config = mc
            break

        if not model_config:
            return JSONResponse(
                {"status": "error", "error": f"No models configured for provider '{provider_name}'"},
                status_code=404,
            )

        try:
            engine = UnifiedInferenceEngine(model_config)
            response = await asyncio.wait_for(
                engine.infer([HumanMessage(content="hello")], tools=None),
                timeout=15,
            )
            text = response.content if hasattr(response, "content") else str(response)
            if isinstance(text, list):
                text = " ".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in text)
            return JSONResponse({"status": "ok", "message": text[:200]})
        except asyncio.TimeoutError:
            return JSONResponse({"status": "error", "error": "Connection timed out after 15 seconds"})
        except Exception as e:
            return JSONResponse({"status": "error", "error": str(e)})

    # --- Gateway restart ---

    @require_auth
    async def handle_restart(request: Request):
        """POST /admin/api/restart — restart the gateway process.

        Spawns a shell background job that waits for the current process to
        exit, then starts a fresh gateway.  This avoids the flock race where
        the new process starts before the old one has released the PID file.
        """
        import os
        import signal
        import subprocess
        import sys

        pid = os.getpid()
        py = sys.executable
        # Shell one-liner: wait for old PID to exit, then start a new gateway.
        subprocess.Popen(
            ["sh", "-c", f"while kill -0 {pid} 2>/dev/null; do sleep 0.2; done; exec {py} -m cliver.gateway.main"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

        async def _delayed_kill():
            await asyncio.sleep(0.5)
            os.kill(pid, signal.SIGTERM)

        asyncio.create_task(_delayed_kill())
        return JSONResponse({"status": "restarting"})

    # --- Chat API (streaming LLM inference for admin portal) ---

    @require_auth
    async def handle_models(request: Request):
        """Return available LLM models for the chat UI."""
        gateway = context.get("gateway")
        if not gateway or not getattr(gateway, "_agent_core", None):
            return JSONResponse({"models": [], "default": None})
        models = list(gateway._agent_core.llm_models.keys())
        return JSONResponse({"models": models, "default": gateway._agent_core.default_model})

    @require_auth
    async def handle_chat(request: Request):
        """Streaming chat endpoint via SSE.

        Accepts JSON body:
            model: str (optional)
            prompt: str (required)
            system_message: str (optional)
            conversation_history: list of {role, content} (optional)
            filter_tools: list of tool names to allow (optional)
            save_media_dir: str - directory to save generated media (optional)
        """
        gateway = context.get("gateway")
        if not gateway or not getattr(gateway, "_agent_core", None):
            return JSONResponse({"error": "Agent not available"}, status_code=503)

        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

        prompt = body.get("prompt", "").strip()
        if not prompt:
            return JSONResponse({"error": "'prompt' is required"}, status_code=400)

        executor = gateway._agent_core
        model = body.get("model") or executor.default_model
        system_message = body.get("system_message")
        raw_history = body.get("conversation_history") or []
        tool_names = body.get("filter_tools")
        save_media_dir = body.get("save_media_dir")

        # Build conversation history, preserving reasoning_content for
        # thinking-mode models (DeepSeek requires it on all assistant messages).
        conversation_history = None
        if raw_history:
            from langchain_core.messages import AIMessage, HumanMessage

            conversation_history = []
            for msg in raw_history:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "user":
                    conversation_history.append(HumanMessage(content=content))
                elif role == "assistant":
                    extra = {}
                    if "reasoning_content" in msg:
                        extra["reasoning_content"] = msg["reasoning_content"]
                    conversation_history.append(AIMessage(content=content, additional_kwargs=extra))

        # System message appender
        def _system_appender():
            parts = []
            if system_message:
                parts.append(system_message)
            parts.append(
                "\n## Server Mode\n\n"
                "You are running as a backend API service via the admin portal. "
                "Do NOT use Ask — there is no human to respond. "
                "Make autonomous decisions. Be concise and direct. "
                "After a tool call completes, summarize the result briefly — "
                "do NOT repeat the tool input or arguments you already provided."
            )
            return "\n".join(parts)

        # Tool filter — if tool_names is provided (even empty list), filter tools
        _tool_filter = None
        if tool_names is not None:
            allowed = set(tool_names)

            async def _tool_filter(user_input, tools):
                return [t for t in tools if t.name in allowed]

        # Configure server mode
        from cliver.permissions import PermissionMode

        if executor.permission_manager:
            executor.permission_manager.set_mode(PermissionMode.YOLO)
        executor.on_permission_prompt = lambda tool, args: "allow"

        # Detect thinking mode to signal frontend to preserve reasoning_content
        uses_thinking = False
        try:
            from cliver.model_capabilities import ModelCapability

            effective_model = model or executor.default_model
            mc = executor._get_llm_model(effective_model) if effective_model else None
            if mc and ModelCapability.THINK_MODE in mc.get_capabilities():
                uses_thinking = True
        except Exception:
            pass

        async def generate():
            full_text = ""
            media_files = []
            stream_media = []
            try:
                async for chunk in executor.stream_user_input(
                    user_input=prompt,
                    model=model,
                    system_message_appender=_system_appender,
                    filter_tools=_tool_filter,
                    conversation_history=conversation_history,
                    outputs_dir=save_media_dir,
                ):
                    if hasattr(chunk, "content") and chunk.content:
                        text = str(chunk.content)
                        full_text += text
                        data = json.dumps({"type": "chunk", "content": text})
                        yield f"data: {data}\n\n".encode()
                    chunk_kwargs = getattr(chunk, "additional_kwargs", None) or {}
                    if "media_content" in chunk_kwargs:
                        stream_media.extend(chunk_kwargs["media_content"])

                # Save media generated by tools (captured from streaming chunks)
                if save_media_dir and stream_media:
                    try:
                        from cliver.media_handler import MultimediaResponse, MultimediaResponseHandler

                        handler = MultimediaResponseHandler(save_directory=save_media_dir)
                        multimedia = MultimediaResponse(media_content=stream_media)
                        media_files = handler.save_media_content(multimedia, prefix="chat")
                    except Exception as e:
                        logger.warning("Failed to save media: %s", e)

                # Record inference as a session
                if full_text:
                    _record_inference(
                        context,
                        "admin:chat",
                        prompt,
                        full_text,
                    )

                done_data = {
                    "type": "done",
                    "content": full_text,
                    "media_files": media_files,
                }
                if uses_thinking:
                    done_data["reasoning_content"] = ""
                done = json.dumps(done_data)
                yield f"data: {done}\n\n".encode()

            except Exception as e:
                logger.error("Chat streaming error: %s", e)
                error = json.dumps({"type": "error", "message": str(e)})
                yield f"data: {error}\n\n".encode()

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    @require_auth
    async def handle_save_step_output(request: Request):
        """Save chat output as a workflow step's output file.

        Accepts JSON body:
            result: str (required) — text result
            media_files: list of str (optional) — file paths of generated media
            output_format: str (optional, default 'md')
        """
        wf_name = request.path_params["name"]
        step_id = request.path_params["step_id"]

        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "Invalid JSON"}, status_code=400)

        result_text = body.get("result", "")
        media_files = body.get("media_files", [])
        output_format = body.get("output_format", "md")

        if not result_text and not media_files:
            return JSONResponse({"error": "No content to save"}, status_code=400)

        try:
            detail = await _run_in_thread(_get_workflow_detail, context, wf_name)
            if not detail or "error" in detail:
                return JSONResponse({"error": "Workflow not found"}, status_code=404)

            outputs_dir = detail.get("outputs_dir") or detail.get("_default_outputs_dir")
            if not outputs_dir:
                return JSONResponse({"error": "No outputs directory configured"}, status_code=400)

            from cliver.workflow.compiler import _save_step_output

            _save_step_output(outputs_dir, step_id, result_text, output_format)

            return JSONResponse(
                {
                    "status": "saved",
                    "outputs_dir": outputs_dir,
                    "step_id": step_id,
                    "media_files": media_files,
                }
            )
        except Exception as e:
            logger.error("Failed to save step output: %s", e)
            return JSONResponse({"error": str(e)}, status_code=500)

    @require_auth
    async def handle_step_log(request: Request):
        """Read a step's log file. Returns JSON lines from {outputs_dir}/{step_id}.log.

        Query params:
            after: int — line offset to read from (for incremental polling)
        """
        wf_name = request.path_params["name"]
        step_id = request.path_params["step_id"]
        after = int(request.query_params.get("after", "0"))

        detail = await _run_in_thread(_get_workflow_detail, context, wf_name)
        if not detail or "error" in detail:
            return JSONResponse({"error": "Workflow not found"}, status_code=404)

        outputs_dir = detail.get("outputs_dir") or detail.get("_default_outputs_dir")
        if not outputs_dir:
            return JSONResponse({"lines": [], "total": 0})

        log_path = Path(outputs_dir) / f"{step_id}.log"
        if not log_path.exists():
            return JSONResponse({"lines": [], "total": 0})

        try:
            all_lines = log_path.read_text(encoding="utf-8").splitlines()
            new_lines = []
            for line in all_lines[after:]:
                try:
                    new_lines.append(json.loads(line))
                except Exception:
                    new_lines.append({"type": "raw", "result": line})
            return JSONResponse({"lines": new_lines, "total": len(all_lines)})
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    @require_auth
    async def handle_run_step(request: Request):
        """Run a single workflow step with context from existing outputs.

        Reads prior step outputs from the outputs directory, renders the
        step prompt with Jinja2, streams LLM inference via SSE, and saves
        the result to the outputs directory.
        """
        wf_name = request.path_params["name"]
        step_id = request.path_params["step_id"]

        gateway = context.get("gateway")
        if not gateway or not getattr(gateway, "_agent_core", None):
            return JSONResponse({"error": "Agent not available"}, status_code=503)

        executor = gateway._agent_core

        # Load workflow definition
        detail = await _run_in_thread(_get_workflow_detail, context, wf_name)
        if not detail or "error" in detail:
            return JSONResponse({"error": "Workflow not found"}, status_code=404)

        steps = detail.get("steps", [])
        step = None
        for s in steps:
            if s.get("id") == step_id:
                step = s
                break
        if not step:
            return JSONResponse({"error": f"Step '{step_id}' not found"}, status_code=404)

        outputs_dir = detail.get("outputs_dir") or detail.get("_default_outputs_dir", "")
        inputs = detail.get("inputs") or {}

        # Build execution context from existing output files on disk
        step_outputs = {}
        if outputs_dir:
            from pathlib import Path as _P

            _text_exts = {".md", ".json", ".txt", ".yaml"}
            _media_exts = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".mp3", ".wav", ".mp4"}
            out_path = _P(outputs_dir)
            if out_path.exists():
                # Collect all media files in the directory
                all_media = [f for f in out_path.iterdir() if f.is_file() and f.suffix in _media_exts]

                # Read text output files (one per step)
                for f in out_path.iterdir():
                    if not f.is_file() or f.suffix not in _text_exts:
                        continue
                    sid = f.stem
                    if sid == step_id:
                        continue
                    try:
                        result_text = f.read_text(encoding="utf-8")
                    except Exception:
                        continue
                    # Match media files: prefix matches step ID (e.g., step_id_timestamp_0.png)
                    media = [str(m) for m in all_media if m.stem.startswith(sid)]
                    step_outputs[sid] = {
                        "outputs": {"result": result_text, "media_files": media},
                        "status": "completed",
                    }

        # Render prompt with ref substitution
        from cliver.workflow.ref_resolver import resolve_refs

        raw_prompt = step.get("prompt", "")
        ref_state = {"inputs": inputs, "steps": step_outputs}
        rendered_prompt = resolve_refs(raw_prompt, ref_state)

        model = step.get("model") or detail.get("_default_model") or executor.default_model

        # Build system message from agent config
        system_message = None
        agent_name = step.get("agent")
        agents = detail.get("agents") or {}
        if agent_name and agent_name in agents:
            agent_cfg = agents[agent_name]
            parts = []
            if agent_cfg.get("role"):
                parts.append("You are: " + agent_cfg["role"])
            if agent_cfg.get("instructions"):
                parts.append(agent_cfg["instructions"])
            if agent_cfg.get("system_message"):
                parts.append(agent_cfg["system_message"])
            if parts:
                system_message = "\n\n".join(parts)

        def _sys_appender():
            p = [
                "You are executing a workflow step autonomously. "
                "Do NOT ask for user confirmation or clarification. "
                "Make the best decision based on the information available and proceed. "
                "Do NOT use the Ask tool. Complete the task directly. "
                "After a tool call, summarize the result briefly — do NOT repeat the tool input.\n\n"
                "ALL output files (text, images, audio, video, code, data) MUST be saved "
                "to the designated outputs directory provided below. Do NOT use any other directory. "
                "Reference files from prior steps using the paths listed under Available Files."
            ]

            overview = detail.get("overview")
            if overview:
                p.append(f"# Workflow Overview\n\n{overview}")

            if system_message:
                p.append(system_message)

            # Outputs directory and prior step files
            file_section = []
            if outputs_dir:
                file_section.append(f"**Outputs directory:** `{outputs_dir}`")
                file_section.append("Save ALL generated files to this directory.")
            prior_files = []
            for sid, sdata in step_outputs.items():
                s_out = sdata.get("outputs", {})
                if s_out.get("media_files"):
                    for f in s_out["media_files"]:
                        prior_files.append(f"- `{f}` (from step '{sid}')")
            if prior_files:
                file_section.append("\n**Files from prior steps:**")
                file_section.extend(prior_files)
            if file_section:
                p.append("# Output Directory & Available Files\n\n" + "\n".join(file_section))

            # Workflow inputs
            if inputs:
                p.append("# Workflow Inputs\n\n" + "\n".join(f"- **{k}**: {v}" for k, v in inputs.items()))

            return "\n\n".join(p)

        # Tool filter from agent config
        _tf = None
        if agent_name and agent_name in agents:
            tools = agents[agent_name].get("tools")
            if tools:
                allowed = set(tools)

                async def _tf(_ui, t):
                    return [x for x in t if x.name in allowed]

        # Server mode
        from cliver.permissions import PermissionMode

        if executor.permission_manager:
            executor.permission_manager.set_mode(PermissionMode.YOLO)
        executor.on_permission_prompt = lambda tool, args: "allow"

        output_format = step.get("output_format", "md")

        # Log events to file + forward to SSE queue
        event_queue = asyncio.Queue()
        step_logger = None
        if outputs_dir:
            from cliver.workflow.compiler import create_step_logger

            step_logger, _ = create_step_logger(outputs_dir, step_id)

        def _event_handler(event):
            from cliver.tool_events import ToolEventType

            if step_logger:
                step_logger(event)

            evt = {"tool": event.tool_name}
            if event.event_type == ToolEventType.TOOL_START:
                evt["type"] = "tool_start"
                if event.args:
                    summary = {}
                    for k, v in event.args.items():
                        val = str(v)
                        summary[k] = val[:200] + "..." if len(val) > 200 else val
                    evt["args"] = summary
            elif event.event_type == ToolEventType.TOOL_END:
                evt["type"] = "tool_end"
                if event.duration_ms:
                    evt["duration_ms"] = round(event.duration_ms)
                if event.result:
                    evt["result"] = event.result[:500]
            elif event.event_type == ToolEventType.TOOL_ERROR:
                evt["type"] = "tool_error"
                evt["error"] = event.error
            else:
                return
            try:
                event_queue.put_nowait(evt)
            except Exception:
                pass

        executor.on_tool_event = _event_handler

        async def generate():
            full_text = ""
            stream_media = []
            try:
                async for chunk in executor.stream_user_input(
                    user_input=rendered_prompt,
                    model=model,
                    system_message_appender=_sys_appender,
                    filter_tools=_tf,
                    outputs_dir=outputs_dir,
                ):
                    # Drain any queued tool events
                    while not event_queue.empty():
                        try:
                            evt = event_queue.get_nowait()
                            yield f"data: {json.dumps(evt)}\n\n".encode()
                        except asyncio.QueueEmpty:
                            break

                    if hasattr(chunk, "content") and chunk.content:
                        text = str(chunk.content)
                        full_text += text
                        data = json.dumps({"type": "chunk", "content": text})
                        yield f"data: {data}\n\n".encode()
                    chunk_kwargs = getattr(chunk, "additional_kwargs", None) or {}
                    if "media_content" in chunk_kwargs:
                        stream_media.extend(chunk_kwargs["media_content"])

                # Drain remaining tool events
                while not event_queue.empty():
                    try:
                        evt = event_queue.get_nowait()
                        yield f"data: {json.dumps(evt)}\n\n".encode()
                    except asyncio.QueueEmpty:
                        break

                # Record inference as a session
                if full_text:
                    _record_inference(
                        context,
                        f"admin:workflow:{wf_name}:{step_id}",
                        rendered_prompt,
                        full_text,
                    )

                # Save output to disk
                media_files = []
                if outputs_dir and full_text:
                    from cliver.workflow.compiler import _save_step_output

                    _save_step_output(outputs_dir, step_id, full_text, output_format)

                # Save media from ctx.generated_media (via stream chunks)
                if outputs_dir and stream_media:
                    try:
                        from cliver.media_handler import MultimediaResponse, MultimediaResponseHandler

                        handler = MultimediaResponseHandler(save_directory=outputs_dir)
                        multimedia = MultimediaResponse(media_content=stream_media)
                        media_files = handler.save_media_content(multimedia, prefix=step_id)
                    except Exception as e:
                        logger.warning("Failed to save step media: %s", e)

                done = json.dumps(
                    {
                        "type": "done",
                        "content": full_text,
                        "media_files": media_files,
                        "outputs_dir": outputs_dir,
                    }
                )
                yield f"data: {done}\n\n".encode()

            except Exception as e:
                logger.error("Step execution error: %s", e)
                error = json.dumps({"type": "error", "message": str(e)})
                yield f"data: {error}\n\n".encode()

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    # --- Return routes ---

    return [
        # Auth
        Route("/admin/api/login", handle_login_submit, methods=["POST"]),
        Route("/admin/logout", handle_logout),
        # API routes
        Route("/admin/api/status", handle_status),
        Route("/admin/api/tasks", handle_tasks),
        Route("/admin/api/tasks", handle_create_task, methods=["POST"]),
        Route("/admin/api/tasks/{name}", handle_task_detail_api),
        Route("/admin/api/tasks/{name}", handle_update_task, methods=["PUT"]),
        Route("/admin/api/tasks/{name}/run", handle_run_task, methods=["POST"]),
        Route("/admin/api/tasks/{name}", handle_delete_task, methods=["DELETE"]),
        Route("/admin/api/sessions/{source}", handle_sessions_by_source),
        Route("/admin/api/sessions/{source}/{id}", handle_session_turns),
        Route("/admin/api/sessions/{source}/{id}", handle_delete_session, methods=["DELETE"]),
        Route("/admin/api/browse", handle_browse_files),
        Route("/admin/api/workflow-executions", handle_all_workflow_executions),
        Route("/admin/api/workflows", handle_workflows),
        # Step-level routes (before catch-all {name} routes)
        Route("/admin/api/workflows/{name}/steps/{step_id}/output", handle_save_step_output, methods=["POST"]),
        Route("/admin/api/workflows/{name}/steps/{step_id}/log", handle_step_log),
        Route("/admin/api/workflows/{name}/steps/{step_id}/run", handle_run_step, methods=["POST"]),
        Route("/admin/api/workflows/{name}/steps/{step_id}/resume", handle_resume_from_step, methods=["POST"]),
        Route("/admin/api/workflows/{name}/executions/{tid}", handle_execution_status),
        Route("/admin/api/workflows/{name}/executions/{tid}", handle_delete_execution, methods=["DELETE"]),
        Route("/admin/api/workflows/{name}/executions", handle_workflow_executions),
        Route("/admin/api/workflows/{name}/run", handle_run_workflow, methods=["POST"]),
        Route("/admin/api/workflows/{name}/stop", handle_stop_workflow, methods=["POST"]),
        Route("/admin/api/workflows/{name}", handle_workflow_detail_api),
        Route("/admin/api/workflows/{name}", handle_update_workflow, methods=["PUT"]),
        Route("/admin/api/media/{path:path}", handle_workflow_media),
        Route("/admin/api/skills/{name}", handle_skill_detail_api),
        Route("/admin/api/skills/{name}", handle_update_skill, methods=["PUT"]),
        Route("/admin/api/skills", handle_create_skill, methods=["POST"]),
        Route("/admin/api/skills", handle_skills),
        Route("/admin/api/adapters", handle_adapters),
        Route("/admin/api/agent", handle_agent),
        Route("/admin/api/agents", handle_agents_list),
        Route("/admin/api/config", handle_config),
        Route("/admin/api/config", handle_update_config, methods=["PUT"]),
        Route("/admin/api/restart", handle_restart, methods=["POST"]),
        Route("/admin/api/providers/{provider_name}/test", handle_test_provider, methods=["POST"]),
        Route("/admin/api/models", handle_models),
        Route("/admin/api/chat", handle_chat, methods=["POST"]),
        # SPA static assets
        Route("/admin/assets/{path:path}", handle_spa_assets),
        # SPA catch-all (MUST be after all API routes)
        Route("/admin/login", handle_spa),
        Route("/admin/{path:path}", handle_spa),
        Route("/admin", handle_spa),
        Route("/admin/", handle_spa),
    ]
