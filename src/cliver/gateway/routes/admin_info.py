"""Info/config routes for the admin portal (status, skills, adapters, agent, config, keys, models)."""

from __future__ import annotations

import inspect
import logging
from typing import Callable, Optional

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

logger = logging.getLogger(__name__)


def _mask_secret(value: Optional[str]) -> str:
    if value is None:
        return ""
    if "{{" in value and "}}" in value:
        return value
    if len(value) <= 8:
        return "****"
    return value[:4] + "****" + value[-4:]


def _get_skills() -> list:
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
                "body": s.body or "",
            }
            for s in skills
        ]
    except Exception as e:
        logger.warning("Failed to get skills: %s", e)
        return []


def _get_adapters(ctx: dict, config_manager=None) -> list:
    try:
        gateway = ctx.get("gateway")
        statuses = {}
        if gateway and gateway._adapter_manager:
            for s in gateway._adapter_manager.platform_statuses:
                statuses[s["name"]] = s

        cm = config_manager
        if cm is None:
            from cliver.config import ConfigManager

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
    try:
        gateway = ctx.get("gateway")
        agent_name = ctx.get("agent_name", "CLIver")
        info = {"name": agent_name, "identity": "", "memory": ""}

        if gateway and hasattr(gateway, "_agent_profile"):
            profile = gateway._agent_profile
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


def _get_config(ctx: dict, config_manager=None) -> dict:
    try:
        cm = config_manager
        if cm is None:
            from cliver.config import ConfigManager

            config_dir = ctx.get("config_dir")
            if not config_dir:
                return {}
            cm = ConfigManager(config_dir)

        gw = cm.config.gateway
        return {
            "gateway": gw.model_dump() if gw else None,
            "session": cm.config.session.model_dump(),
            "theme": cm.config.theme,
            "timezone": cm.config.timezone,
            "user_agent": cm.config.user_agent,
            "default_model": cm.config.default_model,
            "default_agent": cm.config.default_agent,
            "search_engines": cm.config.search_engines,
            "skill_auto_learn": cm.config.skill_auto_learn,
            "model_auto_fallback": cm.config.model_auto_fallback,
            "providers": {n: p.model_dump() for n, p in cm.config.providers.items()},
            "mcp_servers": {n: s.model_dump() for n, s in cm.config.mcpServers.items()},
            "models": {n: m.model_dump() for n, m in cm.config.models.items()},
            "agents": {n: a.model_dump() for n, a in (cm.config.agents or {}).items()},
        }
    except Exception as e:
        logger.warning("Failed to get config: %s", e)
        return {}


def get_info_routes(context: dict, require_auth: Callable, config_manager=None) -> list:
    """Return info/config/keys/models API routes.

    config_manager is a pre-loaded ConfigManager instance (avoids re-reading
    config.yaml from disk on every request). Falls back to creating one from
    config_dir if not provided.
    """

    async def handle_icon(request: Request):
        """Serve the CLIver icon for the admin UI."""
        import importlib.resources

        try:
            data = importlib.resources.files("cliver").joinpath("gateway/admin_dist/icon.png").read_bytes()
        except Exception:
            try:
                # Development fallback: read from repo docs/images/
                from pathlib import Path as _Path

                icon_path = _Path(__file__).parent.parent.parent.parent.parent / "docs" / "images" / "icon.png"
                if icon_path.exists():
                    data = icon_path.read_bytes()
                else:
                    from starlette.responses import Response

                    return Response("", status_code=404)
            except Exception:
                from starlette.responses import Response

                return Response("", status_code=404)
        from starlette.responses import Response

        return Response(data, media_type="image/png", headers={"Cache-Control": "public, max-age=86400"})

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
    async def handle_skills(request: Request):
        skills = _get_skills()
        return JSONResponse(skills)

    @require_auth
    async def handle_adapters(request: Request):
        adapters = _get_adapters(context, config_manager)
        return JSONResponse(adapters)

    @require_auth
    async def handle_agent(request: Request):
        info = _get_agent_info(context)
        return JSONResponse(info)

    @require_auth
    async def handle_config(request: Request):
        config = _get_config(context, config_manager)
        return JSONResponse(config)

    @require_auth
    async def handle_config_update(request: Request):
        from cliver.config import (
            ConfigManager,
            GatewayConfig,
            SessionConfig,
        )

        cm = config_manager
        if cm is None:
            config_dir = context.get("config_dir")
            if not config_dir:
                return JSONResponse({"error": "No config dir"}, status_code=500)
            cm = ConfigManager(config_dir)

        try:
            data = await request.json()

            for key, value in data.items():
                if key == "gateway":
                    cm.config.gateway = GatewayConfig(**value) if value else None

                elif key == "session":
                    cm.config.session = SessionConfig(**value)

                elif hasattr(cm.config, key):
                    setattr(cm.config, key, value)

            cm._save_config()
            logger.info("Configuration updated via admin API")
            return JSONResponse({"status": "ok"})
        except Exception as e:
            logger.error("Failed to save config: %s", e)
            return JSONResponse({"error": str(e)}, status_code=500)

    @require_auth
    async def handle_keys_list(request: Request):
        ks = context.get("key_store")
        if not ks:
            return JSONResponse([])
        keys = ks.list_keys()
        return JSONResponse(
            [
                {"name": k.name, "description": k.description, "created_at": k.created_at, "updated_at": k.updated_at}
                for k in keys
            ]
        )

    @require_auth
    async def handle_keys_create(request: Request):
        ks = context.get("key_store")
        if not ks:
            return JSONResponse({"error": "Key store not available"}, status_code=500)
        data = await request.json()
        name = data.get("name", "").strip()
        value = data.get("value", "")
        description = data.get("description", "")
        if not name or not value:
            return JSONResponse({"error": "name and value are required"}, status_code=400)
        ks.set(name, value, description=description)
        return JSONResponse({"status": "ok", "name": name})

    @require_auth
    async def handle_keys_delete(request: Request):
        ks = context.get("key_store")
        if not ks:
            return JSONResponse({"error": "Key store not available"}, status_code=500)
        name = request.path_params["name"]
        if ks.delete(name):
            return JSONResponse({"status": "ok"})
        return JSONResponse({"error": "not found"}, status_code=404)

    @require_auth
    async def handle_restart(request: Request):
        """Trigger gateway restart by stopping the uvicorn server.

        The process manager (systemd / launchd / CLI wrapper) restarts
        the process after clean exit.
        """
        import asyncio
        import os
        import signal

        logger.info("Gateway restart requested via admin API")

        # Schedule shutdown after response is sent
        def _do_restart():
            os.kill(os.getpid(), signal.SIGTERM)

        asyncio.get_event_loop().call_later(0.5, _do_restart)
        return JSONResponse({"status": "restarting"})

    return [
        Route("/admin/icon.png", handle_icon),
        Route("/admin/api/restart", handle_restart, methods=["POST"]),
        Route("/admin/api/status", handle_status),
        Route("/admin/api/skills", handle_skills),
        Route("/admin/api/adapters", handle_adapters),
        Route("/admin/api/agent", handle_agent),
        Route("/admin/api/config", handle_config),
        Route("/admin/api/config", handle_config_update, methods=["PUT"]),
        Route("/admin/api/keys", handle_keys_list),
        Route("/admin/api/keys", handle_keys_create, methods=["POST"]),
        Route("/admin/api/keys/{name}", handle_keys_delete, methods=["DELETE"]),
    ]
