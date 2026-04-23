"""
Admin portal for CLIver gateway.

Provides a web-based admin interface with Basic Auth for monitoring
and managing the gateway: tasks, sessions, workflows, skills, adapters,
agent info, and configuration.

Usage:
    app = web.Application()
    register_admin_routes(app, username="admin", password="secret", context={...})
"""

import asyncio
import functools
import inspect
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_TEMPLATE_DIR = Path(__file__).parent / "templates"


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------


def _check_basic_auth(request, username: str, password: str) -> bool:
    """Decode Basic Auth header and compare credentials."""
    import base64

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
    """List tasks with live state and last run info."""
    try:
        from cliver.agent_profile import CliverProfile
        from cliver.gateway.task_run_store import TaskRunStore
        from cliver.task_manager import TaskManager

        config_dir = ctx.get("config_dir")
        if not config_dir:
            return []
        profile = CliverProfile(ctx["agent_name"], config_dir)
        tm = TaskManager(profile.tasks_dir)
        tasks = tm.list_tasks()
        if not tasks:
            return []

        db_path = profile.agent_dir / "gateway.db"
        run_store = None
        if db_path.exists():
            run_store = TaskRunStore(db_path)

        result = []
        for task in tasks:
            entry = task.model_dump(exclude_none=True)
            if run_store:
                state = run_store.get_task_state(task.name)
                if state:
                    entry["live_status"] = state
                runs = run_store.get_runs(task.name, limit=1)
                if runs:
                    entry["last_run"] = runs[0].model_dump(exclude_none=True)
            result.append(entry)

        if run_store:
            run_store.close()
        return result
    except Exception as e:
        logger.warning("Failed to get tasks: %s", e)
        return []


def _get_sessions(ctx: dict) -> list:
    """List sessions from the gateway's session manager."""
    try:
        gateway = ctx.get("gateway")
        if gateway and gateway._session_manager:
            return gateway._session_manager.list_sessions()
        return []
    except Exception as e:
        logger.warning("Failed to get sessions: %s", e)
        return []


def _get_session_turns(ctx: dict, session_id: str) -> list:
    """Load turns for a specific session."""
    try:
        gateway = ctx.get("gateway")
        if gateway and gateway._session_manager:
            return gateway._session_manager.load_turns(session_id)
        return []
    except Exception as e:
        logger.warning("Failed to get session turns: %s", e)
        return []


def _delete_session(ctx: dict, session_id: str) -> bool:
    """Delete a session."""
    try:
        gateway = ctx.get("gateway")
        if gateway and gateway._session_manager:
            return gateway._session_manager.delete_session(session_id)
        return False
    except Exception as e:
        logger.warning("Failed to delete session: %s", e)
        return False


async def _run_task(ctx: dict, task_name: str) -> dict:
    """Load and fire a task via the gateway (fire-and-forget)."""
    try:
        from cliver.agent_profile import CliverProfile
        from cliver.task_manager import TaskManager

        config_dir = ctx.get("config_dir")
        gateway = ctx.get("gateway")
        if not config_dir or not gateway:
            return {"status": "error", "message": "gateway or config_dir not available"}

        profile = CliverProfile(ctx["agent_name"], config_dir)
        tm = TaskManager(profile.tasks_dir)
        task = tm.get_task(task_name)
        if not task:
            return {"status": "error", "message": f"task '{task_name}' not found"}

        asyncio.create_task(gateway._run_task(task))
        return {"status": "started"}
    except Exception as e:
        logger.warning("Failed to run task: %s", e)
        return {"status": "error", "message": str(e)}


def _get_workflows(ctx: dict) -> list:
    """List workflows with descriptions."""
    try:
        from cliver.agent_profile import CliverProfile
        from cliver.workflow.persistence import WorkflowStore

        config_dir = ctx.get("config_dir")
        if not config_dir:
            return []
        profile = CliverProfile(ctx["agent_name"], config_dir)
        store = WorkflowStore(profile.workflows_dir)
        names = store.list_workflows()
        result = []
        for name in names:
            wf = store.load_workflow(name)
            if wf:
                entry = {"name": wf.name, "description": wf.description or ""}
                if hasattr(wf, "steps") and wf.steps:
                    entry["steps"] = len(wf.steps)
                result.append(entry)
        return result
    except Exception as e:
        logger.warning("Failed to get workflows: %s", e)
        return []


def _get_skills() -> list:
    """List all discovered skills."""
    try:
        from cliver.skill_manager import SkillManager

        manager = SkillManager()
        skills = manager.list_skills()
        return [
            {
                "name": s.name,
                "description": s.description,
                "source": s.source,
            }
            for s in skills
        ]
    except Exception as e:
        logger.warning("Failed to get skills: %s", e)
        return []


def _get_adapters(ctx: dict) -> list:
    """Get adapter platform statuses."""
    try:
        gateway = ctx.get("gateway")
        if gateway and gateway._adapter_manager:
            return gateway._adapter_manager.platform_statuses
        return []
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
            profile = CliverProfile(agent_name, config_dir)
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
    """Get config overview with secrets masked."""
    try:
        from cliver.config import ConfigManager

        config_dir = ctx.get("config_dir")
        if not config_dir:
            return {"models": {}, "providers": {}, "mcp_servers": {}}

        cm = ConfigManager(config_dir)

        # Models
        models = {}
        for name, mc in cm.config.models.items():
            models[name] = {
                "provider": mc.provider,
                "api_url": mc.get_resolved_url() or "",
                "model_id": mc.name_in_provider or name,
            }

        # Providers
        providers = {}
        for name, pc in cm.config.providers.items():
            providers[name] = {
                "type": pc.type,
                "api_url": pc.api_url,
                "api_key": _mask_secret(pc.api_key),
            }

        # MCP servers
        mcp_servers = {}
        for name, sc in cm.config.mcpServers.items():
            entry = {"transport": sc.transport}
            if hasattr(sc, "command"):
                entry["command"] = sc.command
            if hasattr(sc, "url"):
                entry["url"] = sc.url
            mcp_servers[name] = entry

        return {
            "models": models,
            "providers": providers,
            "mcp_servers": mcp_servers,
        }
    except Exception as e:
        logger.warning("Failed to get config: %s", e)
        return {"models": {}, "providers": {}, "mcp_servers": {}}


# ---------------------------------------------------------------------------
# Route registration
# ---------------------------------------------------------------------------


def register_admin_routes(
    app,
    username: Optional[str],
    password: Optional[str],
    context: dict,
) -> None:
    """Register admin portal routes on an aiohttp application.

    Args:
        app: aiohttp.web.Application
        username: Admin username (None = portal disabled)
        password: Admin password (None = portal disabled)
        context: Dict with keys:
            - get_status: async callable or dict returning status info
            - agent_name: str
            - config_dir: Path or None
            - gateway: Gateway instance (optional)
    """
    from aiohttp import web

    # --- Auth decorator (closure capturing username/password) ---

    def require_auth(handler):
        @functools.wraps(handler)
        async def wrapper(request):
            if username is None or password is None:
                return web.json_response(
                    {"error": "Admin portal is disabled"},
                    status=403,
                )
            if not _check_basic_auth(request, username, password):
                return web.Response(
                    status=401,
                    headers={"WWW-Authenticate": 'Basic realm="CLIver Admin"'},
                    text="Unauthorized",
                )
            return await handler(request)

        return wrapper

    # --- Route handlers (closures capturing context) ---

    @require_auth
    async def handle_admin_page(request):
        template_path = _TEMPLATE_DIR / "admin.html"
        try:
            html = template_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return web.Response(text="Admin template not found", status=500)

        agent_name = context.get("agent_name", "CLIver")
        base_url = str(request.url.origin())
        html = html.replace("{{AGENT_NAME}}", agent_name)
        html = html.replace("{{BASE_URL}}", base_url)
        return web.Response(text=html, content_type="text/html")

    @require_auth
    async def handle_status(request):
        get_status = context.get("get_status")
        if get_status is None:
            return web.json_response({})
        if callable(get_status):
            result = get_status()
            if inspect.isawaitable(result):
                result = await result
            return web.json_response(result)
        return web.json_response(get_status)

    @require_auth
    async def handle_tasks(request):
        tasks = _get_tasks(context)
        return web.json_response(tasks)

    @require_auth
    async def handle_run_task(request):
        task_name = request.match_info["name"]
        result = await _run_task(context, task_name)
        status_code = 200 if result.get("status") == "started" else 400
        return web.json_response(result, status=status_code)

    @require_auth
    async def handle_sessions(request):
        sessions = _get_sessions(context)
        return web.json_response(sessions)

    @require_auth
    async def handle_session_detail(request):
        session_id = request.match_info["id"]
        turns = _get_session_turns(context, session_id)
        return web.json_response(turns)

    @require_auth
    async def handle_delete_session(request):
        session_id = request.match_info["id"]
        deleted = _delete_session(context, session_id)
        if deleted:
            return web.json_response({"status": "deleted"})
        return web.json_response({"error": "session not found"}, status=404)

    @require_auth
    async def handle_workflows(request):
        workflows = _get_workflows(context)
        return web.json_response(workflows)

    @require_auth
    async def handle_skills(request):
        skills = _get_skills()
        return web.json_response(skills)

    @require_auth
    async def handle_adapters(request):
        adapters = _get_adapters(context)
        return web.json_response(adapters)

    @require_auth
    async def handle_agent(request):
        info = _get_agent_info(context)
        return web.json_response(info)

    @require_auth
    async def handle_config(request):
        config = _get_config(context)
        return web.json_response(config)

    # --- Register routes ---

    app.router.add_get("/admin", handle_admin_page)
    app.router.add_get("/admin/api/status", handle_status)
    app.router.add_get("/admin/api/tasks", handle_tasks)
    app.router.add_post("/admin/api/tasks/{name}/run", handle_run_task)
    app.router.add_get("/admin/api/sessions", handle_sessions)
    app.router.add_get("/admin/api/sessions/{id}", handle_session_detail)
    app.router.add_delete("/admin/api/sessions/{id}", handle_delete_session)
    app.router.add_get("/admin/api/workflows", handle_workflows)
    app.router.add_get("/admin/api/skills", handle_skills)
    app.router.add_get("/admin/api/adapters", handle_adapters)
    app.router.add_get("/admin/api/agent", handle_agent)
    app.router.add_get("/admin/api/config", handle_config)
