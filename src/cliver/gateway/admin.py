"""
Admin portal for CLIver gateway.

Provides a web-based admin interface with Basic Auth for monitoring
and managing the gateway: tasks, sessions, workflows, skills, adapters,
agent info, and configuration.

Multi-page routing: base.html + pages/{page_name}.html assembly pattern.

Usage:
    app = web.Application()
    register_admin_routes(app, username="admin", password="secret", context={...})
"""

import asyncio
import base64
import functools
import inspect
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_TEMPLATE_DIR = Path(__file__).parent / "templates"
_PAGES_DIR = _TEMPLATE_DIR / "pages"


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
# Page rendering
# ---------------------------------------------------------------------------


def _render_page(page_name, context, username, password, extra_replacements=None):
    """Assemble base.html + pages/{page_name}.html with substitutions."""
    base_path = _TEMPLATE_DIR / "base.html"
    page_path = _PAGES_DIR / f"{page_name}.html"
    if not page_path.exists():
        return None
    base_html = base_path.read_text(encoding="utf-8")
    page_html = page_path.read_text(encoding="utf-8")
    auth_token = base64.b64encode(f"{username}:{password}".encode()).decode()
    html = base_html.replace("{{CONTENT}}", page_html)
    html = html.replace("{{AGENT_NAME}}", context.get("agent_name", "CLIver"))
    html = html.replace("{{BASE_URL}}", "/admin")
    html = html.replace("{{AUTH_TOKEN}}", auth_token)
    html = html.replace("{{CURRENT_PAGE}}", page_name.split("_")[0])
    if extra_replacements:
        for key, value in extra_replacements.items():
            html = html.replace(key, value)
    return html


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


# ---------------------------------------------------------------------------
# Source-aware session helpers
# ---------------------------------------------------------------------------


def _get_session_manager(ctx, source):
    """Get the session manager for a given source ('cli' or 'gateway')."""
    if source == "gateway":
        gw = ctx.get("gateway")
        return gw._session_manager if gw else None
    elif source == "cli":
        return ctx.get("cli_session_manager")
    return None


def _get_sessions_by_source(ctx, source):
    """List sessions from the specified source."""
    sm = _get_session_manager(ctx, source)
    if not sm:
        return []
    try:
        return sm.list_sessions()
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


def _get_task_detail(ctx, task_name):
    """Get detailed info for a single task including run history."""
    try:
        from cliver.agent_profile import CliverProfile
        from cliver.gateway.task_run_store import TaskRunStore
        from cliver.task_manager import TaskManager

        config_dir = ctx.get("config_dir")
        if not config_dir:
            return None
        profile = CliverProfile(ctx["agent_name"], config_dir)
        tm = TaskManager(profile.tasks_dir)
        task = tm.get_task(task_name)
        if not task:
            return None
        entry = task.model_dump(exclude_none=True)
        db_path = profile.agent_dir / "gateway.db"
        if db_path.exists():
            rs = TaskRunStore(db_path)
            state = rs.get_task_state(task_name)
            if state:
                entry["live_status"] = state
            runs = rs.get_runs(task_name, limit=10)
            entry["runs"] = [r.model_dump(exclude_none=True) for r in runs]
            rs.close()
        return entry
    except Exception as e:
        logger.warning("Failed to get task detail: %s", e)
        return None


def _get_workflow_detail(ctx, workflow_name):
    """Get detailed info for a single workflow."""
    try:
        from cliver.agent_profile import CliverProfile
        from cliver.workflow.persistence import WorkflowStore

        config_dir = ctx.get("config_dir")
        if not config_dir:
            return None
        profile = CliverProfile(ctx["agent_name"], config_dir)
        store = WorkflowStore(profile.workflows_dir)
        wf = store.load_workflow(workflow_name)
        if not wf:
            return None
        return wf.model_dump(exclude_none=True, mode="json")
    except Exception as e:
        logger.warning("Failed to get workflow detail: %s", e)
        return None


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
                "license": s.license,
                "compatibility": s.compatibility,
                "allowed_tools": s.allowed_tools,
                "body_preview": s.body[:500] if s.body else "",
                "body_length": len(s.body) if s.body else 0,
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
            - cli_session_manager: SessionManager for CLI sessions (optional)
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
    async def handle_admin_root(request):
        raise web.HTTPFound("/admin/gateway")

    @require_auth
    async def handle_admin_page(request):
        page = request.match_info["page"]
        html = _render_page(page, context, username, password)
        if html is None:
            return web.Response(text="Page not found", status=404)
        return web.Response(text=html, content_type="text/html")

    @require_auth
    async def handle_task_detail_page(request):
        name = request.match_info["name"]
        html = _render_page(
            "task_detail",
            context,
            username,
            password,
            extra_replacements={"{{ITEM_NAME}}": name},
        )
        if html is None:
            return web.Response(text="Page not found", status=404)
        return web.Response(text=html, content_type="text/html")

    @require_auth
    async def handle_workflow_detail_page(request):
        name = request.match_info["name"]
        html = _render_page(
            "workflow_detail",
            context,
            username,
            password,
            extra_replacements={"{{ITEM_NAME}}": name},
        )
        if html is None:
            return web.Response(text="Page not found", status=404)
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
    async def handle_task_detail_api(request):
        name = request.match_info["name"]
        detail = _get_task_detail(context, name)
        return web.json_response(detail)

    @require_auth
    async def handle_run_task(request):
        task_name = request.match_info["name"]
        logger.info("[admin] Task '%s' triggered via admin portal", task_name)
        result = await _run_task(context, task_name)
        status_code = 200 if result.get("status") == "started" else 400
        return web.json_response(result, status=status_code)

    @require_auth
    async def handle_sessions_by_source(request):
        source = request.match_info["source"]
        if source not in ("cli", "gateway"):
            return web.json_response(
                {"error": f"Invalid source '{source}'. Use 'cli' or 'gateway'."},
                status=400,
            )
        sessions = _get_sessions_by_source(context, source)
        return web.json_response(sessions)

    @require_auth
    async def handle_session_turns(request):
        source = request.match_info["source"]
        session_id = request.match_info["id"]
        if source not in ("cli", "gateway"):
            return web.json_response(
                {"error": f"Invalid source '{source}'."},
                status=400,
            )
        turns = _get_session_turns(context, source, session_id)
        return web.json_response(turns)

    @require_auth
    async def handle_delete_session(request):
        source = request.match_info["source"]
        session_id = request.match_info["id"]
        if source not in ("cli", "gateway"):
            return web.json_response(
                {"error": f"Invalid source '{source}'."},
                status=400,
            )
        logger.info(
            "[admin] Session '%s' (%s) deleted via admin portal",
            session_id,
            source,
        )
        deleted = _delete_session(context, source, session_id)
        if deleted:
            return web.json_response({"status": "deleted"})
        return web.json_response({"error": "session not found"}, status=404)

    @require_auth
    async def handle_workflows(request):
        workflows = _get_workflows(context)
        return web.json_response(workflows)

    @require_auth
    async def handle_workflow_detail_api(request):
        name = request.match_info["name"]
        detail = _get_workflow_detail(context, name)
        return web.json_response(detail)

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

    # Root redirect
    app.router.add_get("/admin", handle_admin_root)

    # Detail page routes (before catch-all)
    app.router.add_get("/admin/tasks/{name}", handle_task_detail_page)
    app.router.add_get("/admin/workflows/{name}", handle_workflow_detail_page)

    # Catch-all section route
    app.router.add_get("/admin/{page}", handle_admin_page)

    # API routes
    app.router.add_get("/admin/api/status", handle_status)
    app.router.add_get("/admin/api/tasks", handle_tasks)
    app.router.add_get("/admin/api/tasks/{name}", handle_task_detail_api)
    app.router.add_post("/admin/api/tasks/{name}/run", handle_run_task)
    app.router.add_get("/admin/api/sessions/{source}", handle_sessions_by_source)
    app.router.add_get("/admin/api/sessions/{source}/{id}", handle_session_turns)
    app.router.add_delete("/admin/api/sessions/{source}/{id}", handle_delete_session)
    app.router.add_get("/admin/api/workflows", handle_workflows)
    app.router.add_get("/admin/api/workflows/{name}", handle_workflow_detail_api)
    app.router.add_get("/admin/api/skills", handle_skills)
    app.router.add_get("/admin/api/adapters", handle_adapters)
    app.router.add_get("/admin/api/agent", handle_agent)
    app.router.add_get("/admin/api/config", handle_config)
