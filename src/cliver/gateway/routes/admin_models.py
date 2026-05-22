"""Model, Provider, and Endpoint API routes for the admin portal."""

from __future__ import annotations

import logging
from typing import Callable

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

logger = logging.getLogger(__name__)


def get_model_routes(model_store, require_auth: Callable) -> list:
    """Return model, provider, and endpoint CRUD API routes."""
    from cliver.gateway.admin import _run_in_thread

    # -- Models ---------------------------------------------------------------

    @require_auth
    async def handle_list_models(request: Request):
        capability = request.query_params.get("capability")
        models = await _run_in_thread(model_store.list_models, capability or None)
        return JSONResponse([m.model_dump() for m in models])

    @require_auth
    async def handle_create_model(request: Request):
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "Invalid JSON"}, status_code=400)
        name = (body.get("name") or "").strip()
        provider_id = (body.get("provider_id") or "").strip()
        endpoint_id = (body.get("endpoint_id") or "").strip()
        if not name or not provider_id or not endpoint_id:
            return JSONResponse(
                {"error": "name, provider_id, and endpoint_id are required"},
                status_code=400,
            )
        model = await _run_in_thread(
            model_store.create_model,
            provider_id=provider_id,
            endpoint_id=endpoint_id,
            name=name,
            capabilities=body.get("capabilities"),
            options=body.get("options"),
            think_mode=body.get("think_mode"),
            context_window=body.get("context_window"),
            pricing=body.get("pricing"),
            is_default=body.get("is_default", 0),
        )
        return JSONResponse(model.model_dump())

    @require_auth
    async def handle_get_model(request: Request):
        model_id = request.path_params["model_id"]
        model = await _run_in_thread(model_store.get_model, model_id)
        if model is None:
            return JSONResponse({"error": "Model not found"}, status_code=404)
        return JSONResponse(model.model_dump())

    @require_auth
    async def handle_update_model(request: Request):
        model_id = request.path_params["model_id"]
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "Invalid JSON"}, status_code=400)
        model = await _run_in_thread(
            model_store.update_model,
            model_id,
            name=body.get("name"),
            endpoint_id=body.get("endpoint_id"),
            capabilities=body.get("capabilities"),
            options=body.get("options"),
            think_mode=body.get("think_mode"),
            context_window=body.get("context_window"),
            pricing=body.get("pricing"),
            is_default=body.get("is_default"),
        )
        if model is None:
            return JSONResponse({"error": "Model not found"}, status_code=404)
        return JSONResponse(model.model_dump())

    @require_auth
    async def handle_delete_model(request: Request):
        model_id = request.path_params["model_id"]
        deleted = await _run_in_thread(model_store.delete_model, model_id)
        if not deleted:
            return JSONResponse({"error": "Model not found"}, status_code=404)
        return JSONResponse({"status": "deleted"})

    @require_auth
    async def handle_set_default_model(request: Request):
        model_id = request.path_params["model_id"]
        ok = await _run_in_thread(model_store.set_default_model, model_id)
        if not ok:
            return JSONResponse({"error": "Model not found"}, status_code=404)
        return JSONResponse({"status": "ok"})

    # -- Providers ------------------------------------------------------------

    @require_auth
    async def handle_list_providers(request: Request):
        providers = await _run_in_thread(model_store.list_providers)
        return JSONResponse([p.model_dump() for p in providers])

    @require_auth
    async def handle_create_provider(request: Request):
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "Invalid JSON"}, status_code=400)
        name = (body.get("name") or "").strip()
        if not name:
            return JSONResponse({"error": "name is required"}, status_code=400)
        provider = await _run_in_thread(
            model_store.create_provider,
            name,
            body.get("type", "openai"),
            body.get("api_key"),
            body.get("rate_limit"),
        )
        return JSONResponse(provider.model_dump())

    @require_auth
    async def handle_update_provider(request: Request):
        provider_id = request.path_params["provider_id"]
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "Invalid JSON"}, status_code=400)
        provider = await _run_in_thread(
            model_store.update_provider,
            provider_id,
            name=body.get("name"),
            type=body.get("type"),
            api_key=body.get("api_key"),
            rate_limit=body.get("rate_limit"),
        )
        if provider is None:
            return JSONResponse({"error": "Provider not found"}, status_code=404)
        return JSONResponse(provider.model_dump())

    @require_auth
    async def handle_delete_provider(request: Request):
        provider_id = request.path_params["provider_id"]
        deleted = await _run_in_thread(model_store.delete_provider, provider_id)
        if not deleted:
            return JSONResponse({"error": "Provider not found"}, status_code=404)
        return JSONResponse({"status": "deleted"})

    # -- Endpoints ------------------------------------------------------------

    @require_auth
    async def handle_list_endpoints(request: Request):
        provider_id = request.path_params["provider_id"]
        endpoints = await _run_in_thread(model_store.list_endpoints, provider_id)
        return JSONResponse([e.model_dump() for e in endpoints])

    @require_auth
    async def handle_create_endpoint(request: Request):
        provider_id = request.path_params["provider_id"]
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "Invalid JSON"}, status_code=400)
        base_url = (body.get("base_url") or "").strip()
        if not base_url:
            return JSONResponse({"error": "base_url is required"}, status_code=400)
        endpoint = await _run_in_thread(model_store.create_endpoint, provider_id, base_url)
        return JSONResponse(endpoint.model_dump())

    @require_auth
    async def handle_update_endpoint(request: Request):
        endpoint_id = request.path_params["endpoint_id"]
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "Invalid JSON"}, status_code=400)
        base_url = body.get("base_url")
        endpoint = await _run_in_thread(model_store.update_endpoint, endpoint_id, base_url)
        if endpoint is None:
            return JSONResponse({"error": "Endpoint not found"}, status_code=404)
        return JSONResponse(endpoint.model_dump())

    @require_auth
    async def handle_delete_endpoint(request: Request):
        endpoint_id = request.path_params["endpoint_id"]
        deleted = await _run_in_thread(model_store.delete_endpoint, endpoint_id)
        if not deleted:
            return JSONResponse({"error": "Endpoint not found"}, status_code=404)
        return JSONResponse({"status": "deleted"})

    return [
        # Models
        Route("/admin/api/models", handle_list_models),
        Route("/admin/api/models", handle_create_model, methods=["POST"]),
        Route("/admin/api/models/{model_id}", handle_get_model),
        Route("/admin/api/models/{model_id}", handle_update_model, methods=["PATCH"]),
        Route("/admin/api/models/{model_id}", handle_delete_model, methods=["DELETE"]),
        Route("/admin/api/models/{model_id}/default", handle_set_default_model, methods=["POST"]),
        # Providers
        Route("/admin/api/providers", handle_list_providers),
        Route("/admin/api/providers", handle_create_provider, methods=["POST"]),
        Route("/admin/api/providers/{provider_id}", handle_update_provider, methods=["PATCH"]),
        Route("/admin/api/providers/{provider_id}", handle_delete_provider, methods=["DELETE"]),
        # Endpoints
        Route("/admin/api/providers/{provider_id}/endpoints", handle_list_endpoints),
        Route("/admin/api/providers/{provider_id}/endpoints", handle_create_endpoint, methods=["POST"]),
        Route("/admin/api/endpoints/{endpoint_id}", handle_update_endpoint, methods=["PATCH"]),
        Route("/admin/api/endpoints/{endpoint_id}", handle_delete_endpoint, methods=["DELETE"]),
    ]
