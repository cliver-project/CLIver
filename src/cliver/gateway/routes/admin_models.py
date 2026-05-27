"""Model and Provider API routes for the admin portal (config.yaml backed)."""

from __future__ import annotations

import logging
from typing import Callable

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

logger = logging.getLogger(__name__)


def get_model_routes(config_manager, require_auth: Callable) -> list:
    """Return model and provider CRUD API routes backed by config.yaml."""
    from cliver.gateway.admin import _run_in_thread

    # -- Models ---------------------------------------------------------------

    @require_auth
    async def handle_list_models(request: Request):
        models = await _run_in_thread(config_manager.list_llm_models)
        cat_filter = (request.query_params.get("category") or "").strip().lower()
        result = []
        for name, mc in models.items():
            if cat_filter and mc.category.lower() != cat_filter:
                continue
            d = {
                "id": name,
                "name": name,
                "category": mc.category,
                "model": mc.model,
                "provider": mc.provider,
                "api_url": mc.api_url,
                "options": mc.options.model_dump() if mc.options else {},
                "is_default": 1 if config_manager.config.default_model == name else 0,
            }
            result.append(d)
        return JSONResponse(result)

    @require_auth
    async def handle_create_model(request: Request):
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "Invalid JSON"}, status_code=400)
        provider = (body.get("provider") or "").strip()
        name = (body.get("name") or "").strip()
        if not name or not provider:
            return JSONResponse({"error": "provider and name are required"}, status_code=400)
        try:
            await _run_in_thread(
                config_manager.add_or_update_llm_model,
                provider=provider,
                model_name=name,
                api_model_name=body.get("model", name),
                category=body.get("category", "text"),
                api_url=body.get("api_url"),
                options=body.get("options"),
                is_default=body.get("is_default"),
            )
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        return JSONResponse({"id": name, "name": name, "provider": provider})

    @require_auth
    async def handle_get_model(request: Request):
        model_id = request.path_params["model_id"]
        models = await _run_in_thread(config_manager.list_llm_models)
        mc = models.get(model_id)
        if mc is None:
            for key, m in models.items():
                if key.endswith(f"/{model_id}"):
                    mc = m
                    break
        if mc is None:
            return JSONResponse({"error": "Model not found"}, status_code=404)
        d = {
            "id": mc.name,
            "name": mc.name,
            "category": mc.category,
            "model": mc.model,
            "provider": mc.provider,
            "api_url": mc.api_url,
            "options": mc.options.model_dump() if mc.options else {},
            "is_default": 1 if config_manager.config.default_model == mc.name else 0,
        }
        return JSONResponse(d)

    @require_auth
    async def handle_update_model(request: Request):
        model_id = request.path_params["model_id"]
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "Invalid JSON"}, status_code=400)
        models = await _run_in_thread(config_manager.list_llm_models)
        mc = models.get(model_id)
        if mc is None:
            return JSONResponse({"error": "Model not found"}, status_code=404)
        try:
            await _run_in_thread(
                config_manager.add_or_update_llm_model,
                provider=mc.provider,
                model_name=mc.name,
                api_model_name=body.get("model", mc.model),
                category=body.get("category", mc.category),
                api_url=body.get("api_url", mc.api_url),
                options=body.get("options"),
                is_default=body.get("is_default"),
            )
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        return JSONResponse({"id": mc.name, "name": mc.name, "provider": mc.provider})

    @require_auth
    async def handle_delete_model(request: Request):
        model_id = request.path_params["model_id"]
        deleted = await _run_in_thread(config_manager.remove_llm_model, model_id)
        if not deleted:
            return JSONResponse({"error": "Model not found"}, status_code=404)
        return JSONResponse({"status": "deleted"})

    @require_auth
    async def handle_set_default_model(request: Request):
        model_id = request.path_params["model_id"]
        ok = await _run_in_thread(config_manager.set_default_model, model_id)
        if not ok:
            return JSONResponse({"error": "Model not found"}, status_code=404)
        return JSONResponse({"status": "ok"})

    # -- Providers ------------------------------------------------------------

    @require_auth
    async def handle_list_providers(request: Request):
        providers = await _run_in_thread(config_manager.list_providers)
        result = []
        for name, p in providers.items():
            result.append(
                {
                    "id": name,
                    "name": name,
                    "type": p.type,
                    "api_url": p.api_url,
                    "api_key": "***" if p.api_key else None,
                    "rate_limit": p.rate_limit.model_dump() if p.rate_limit else None,
                    "pricing": p.pricing.model_dump() if p.pricing else None,
                }
            )
        return JSONResponse(result)

    @require_auth
    async def handle_create_provider(request: Request):
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "Invalid JSON"}, status_code=400)
        name = (body.get("name") or "").strip()
        if not name:
            return JSONResponse({"error": "name is required"}, status_code=400)
        await _run_in_thread(
            config_manager.add_or_update_provider,
            name,
            body.get("type", "openai"),
            body.get("api_url", ""),
            body.get("api_key"),
            body.get("rate_limit"),
            body.get("pricing"),
        )
        return JSONResponse({"id": name, "name": name})

    @require_auth
    async def handle_update_provider(request: Request):
        provider_id = request.path_params["provider_id"]
        providers = await _run_in_thread(config_manager.list_providers)
        existing = providers.get(provider_id)
        if existing is None:
            return JSONResponse({"error": "Provider not found"}, status_code=404)
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "Invalid JSON"}, status_code=400)
        await _run_in_thread(
            config_manager.add_or_update_provider,
            provider_id,
            body.get("type", existing.type),
            body.get("api_url", existing.api_url),
            body.get("api_key") if body.get("api_key") is not None else existing.api_key,
            body.get("rate_limit"),
            body.get("pricing"),
        )
        return JSONResponse({"id": provider_id, "name": provider_id})

    @require_auth
    async def handle_delete_provider(request: Request):
        provider_id = request.path_params["provider_id"]
        try:
            deleted = await _run_in_thread(config_manager.remove_provider, provider_id)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        if not deleted:
            return JSONResponse({"error": "Provider not found"}, status_code=404)
        return JSONResponse({"status": "deleted"})

    return [
        # Models
        Route("/admin/api/models", handle_list_models),
        Route("/admin/api/models", handle_create_model, methods=["POST"]),
        Route("/admin/api/models/{model_id:path}/default", handle_set_default_model, methods=["POST"]),
        Route("/admin/api/models/{model_id:path}", handle_get_model),
        Route("/admin/api/models/{model_id:path}", handle_update_model, methods=["PATCH"]),
        Route("/admin/api/models/{model_id:path}", handle_delete_model, methods=["DELETE"]),
        # Providers
        Route("/admin/api/providers", handle_list_providers),
        Route("/admin/api/providers", handle_create_provider, methods=["POST"]),
        Route("/admin/api/providers/{provider_id:path}", handle_update_provider, methods=["PATCH"]),
        Route("/admin/api/providers/{provider_id:path}", handle_delete_provider, methods=["DELETE"]),
    ]
