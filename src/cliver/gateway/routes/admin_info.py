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


def _get_adapters(ctx: dict) -> list:
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
    try:
        from cliver.config import ConfigManager

        config_dir = ctx.get("config_dir")
        if not config_dir:
            return {"models": {}, "providers": {}, "mcp_servers": {}}

        cm = ConfigManager(config_dir)

        providers = {}
        for name, pc in cm.config.providers.items():
            provider_models = [mc.api_model_name for mc in cm.config.models.values() if mc.provider == name]
            providers[name] = {
                "type": pc.type,
                "api_url": pc.api_url,
                "api_key": _mask_secret(pc.api_key),
                "models": provider_models,
            }
            if pc.image_url:
                providers[name]["image_url"] = pc.image_url
            if pc.image_model:
                providers[name]["image_model"] = pc.image_model

        mcp_servers = {}
        for name, sc in cm.config.mcpServers.items():
            entry = {"transport": sc.transport}
            if hasattr(sc, "command"):
                entry["command"] = sc.command
            if hasattr(sc, "url"):
                entry["url"] = sc.url
            mcp_servers[name] = entry

        return {"providers": providers, "mcp_servers": mcp_servers}
    except Exception as e:
        logger.warning("Failed to get config: %s", e)
        return {"providers": {}, "mcp_servers": {}}


def get_info_routes(context: dict, require_auth: Callable) -> list:
    """Return info/config/keys/models API routes."""

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
        adapters = _get_adapters(context)
        return JSONResponse(adapters)

    @require_auth
    async def handle_agent(request: Request):
        info = _get_agent_info(context)
        return JSONResponse(info)

    @require_auth
    async def handle_config(request: Request):
        config = _get_config(context)
        return JSONResponse(config)

    @require_auth
    async def handle_config_update(request: Request):
        config_dir = context.get("config_dir")
        if not config_dir:
            return JSONResponse({"error": "No config dir"}, status_code=500)

        try:
            data = await request.json()
            from cliver.config import (
                AgentConfig,
                ConfigManager,
                GatewayConfig,
                ProviderConfig,
                SessionConfig,
                SSEMCPServerConfig,
                StdioMCPServerConfig,
                StreamableHttpMCPServerConfig,
                WebSocketMCPServerConfig,
            )

            cm = ConfigManager(config_dir)

            for key, value in data.items():
                if key == "providers":
                    providers = {}
                    for pname, pdata in value.items():
                        if isinstance(pdata, dict):
                            cfg = {"name": pname}
                            cfg.update(pdata)
                            cfg.pop("models", None)
                            providers[pname] = ProviderConfig(**cfg)
                    cm.config.providers = providers

                elif key in ("mcpServers", "mcp_servers"):
                    servers = {}
                    for sname, sdata in value.items():
                        if isinstance(sdata, dict):
                            cfg = {"name": sname}
                            cfg.update(sdata)
                            transport = cfg.get("transport")
                            if transport == "stdio":
                                servers[sname] = StdioMCPServerConfig(**cfg)
                            elif transport == "sse":
                                servers[sname] = SSEMCPServerConfig(**cfg)
                            elif transport == "streamable_http":
                                servers[sname] = StreamableHttpMCPServerConfig(**cfg)
                            elif transport == "websocket":
                                servers[sname] = WebSocketMCPServerConfig(**cfg)
                    cm.config.mcpServers = servers

                elif key == "agents":
                    cm.config.agents = {k: AgentConfig(**v) for k, v in value.items()}

                elif key == "gateway":
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
        from cliver.key_store import KeyStore

        config_dir = context.get("config_dir")
        if not config_dir:
            return JSONResponse([])
        ks = KeyStore(config_dir / "cliver.db")
        keys = ks.list_keys()
        return JSONResponse(
            [
                {"name": k.name, "description": k.description, "created_at": k.created_at, "updated_at": k.updated_at}
                for k in keys
            ]
        )

    @require_auth
    async def handle_keys_create(request: Request):
        from cliver.key_store import KeyStore

        config_dir = context.get("config_dir")
        if not config_dir:
            return JSONResponse({"error": "No config dir"}, status_code=500)
        ks = KeyStore(config_dir / "cliver.db")
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
        from cliver.key_store import KeyStore

        config_dir = context.get("config_dir")
        if not config_dir:
            return JSONResponse({"error": "No config dir"}, status_code=500)
        ks = KeyStore(config_dir / "cliver.db")
        name = request.path_params["name"]
        if ks.delete(name):
            return JSONResponse({"status": "ok"})
        return JSONResponse({"error": "not found"}, status_code=404)

    @require_auth
    async def handle_agents(request: Request):
        from cliver.config import ConfigManager

        config_dir = context.get("config_dir")
        if not config_dir:
            return JSONResponse([])
        cm = ConfigManager(config_dir)
        return JSONResponse(
            [
                {"name": name, "type": ac.type, "model": ac.model, "description": ac.description}
                for name, ac in cm.config.agents.items()
            ]
        )

    @require_auth
    async def handle_models(request: Request):
        gateway = context.get("gateway")
        if not gateway or not getattr(gateway, "_agent_core", None):
            return JSONResponse({"models": [], "default": None})
        models = list(gateway._agent_core.llm_models.keys())
        return JSONResponse({"models": models, "default": gateway._agent_core.default_model})

    return [
        Route("/admin/api/agents", handle_agents),
        Route("/admin/api/status", handle_status),
        Route("/admin/api/skills", handle_skills),
        Route("/admin/api/adapters", handle_adapters),
        Route("/admin/api/agent", handle_agent),
        Route("/admin/api/config", handle_config),
        Route("/admin/api/config", handle_config_update, methods=["PUT"]),
        Route("/admin/api/keys", handle_keys_list),
        Route("/admin/api/keys", handle_keys_create, methods=["POST"]),
        Route("/admin/api/keys/{name}", handle_keys_delete, methods=["DELETE"]),
        Route("/admin/api/models", handle_models),
    ]
