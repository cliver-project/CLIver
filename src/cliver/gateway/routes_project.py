"""Project, Issue, and Scenario API routes for the gateway."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

if TYPE_CHECKING:
    from cliver.notebook.store import NotebookStore
    from cliver.project.local_provider import LocalProvider
    from cliver.project.scenario_registry import ScenarioRegistry

logger = logging.getLogger(__name__)


def get_project_routes(
    provider: "LocalProvider",
    scenario_registry: "ScenarioRegistry",
    notebook_store: "NotebookStore",
    require_auth: Callable,
) -> list:
    """Return project/issue/scenario API routes."""

    # --- Projects ---

    @require_auth
    async def handle_projects_list(request: Request):
        projects = await provider.list_projects()
        return JSONResponse([p.model_dump() for p in projects])

    @require_auth
    async def handle_projects_create(request: Request):
        data = await request.json()
        name = data.get("name", "").strip()
        if not name:
            return JSONResponse({"error": "name is required"}, status_code=400)
        p = await provider.create_project(name, data.get("description", ""))
        return JSONResponse(p.model_dump())

    @require_auth
    async def handle_project_get(request: Request):
        project_id = request.path_params["id"]
        p = await provider.get_project(project_id)
        if not p:
            return JSONResponse({"error": "not found"}, status_code=404)
        return JSONResponse(p.model_dump())

    @require_auth
    async def handle_project_update(request: Request):
        project_id = request.path_params["id"]
        p = await provider.get_project(project_id)
        if not p:
            return JSONResponse({"error": "not found"}, status_code=404)
        data = await request.json()
        if "name" in data:
            p.name = data["name"]
        if "description" in data:
            p.description = data["description"]
        await provider.update_project(p)
        return JSONResponse({"status": "ok"})

    @require_auth
    async def handle_project_delete(request: Request):
        project_id = request.path_params["id"]
        if await provider.delete_project(project_id):
            return JSONResponse({"status": "ok"})
        return JSONResponse({"error": "not found"}, status_code=404)

    # --- Issues ---

    @require_auth
    async def handle_issues_list(request: Request):
        project_id = request.path_params["id"]
        status = request.query_params.get("status")
        labels_param = request.query_params.get("labels")
        labels = labels_param.split(",") if labels_param else None
        issues = await provider.list_issues(project_id, status=status, labels=labels)
        return JSONResponse([i.model_dump() for i in issues])

    @require_auth
    async def handle_issues_create(request: Request):
        project_id = request.path_params["id"]
        p = await provider.get_project(project_id)
        if not p:
            return JSONResponse({"error": "project not found"}, status_code=404)
        data = await request.json()
        title = data.get("title", "").strip()
        if not title:
            return JSONResponse({"error": "title is required"}, status_code=400)
        issue = await provider.create_issue(
            project_id=project_id,
            title=title,
            description=data.get("description", ""),
            priority=data.get("priority", "medium"),
            labels=data.get("labels"),
            assigned_agent=data.get("assigned_agent"),
            scenario_id=data.get("scenario_id"),
        )
        return JSONResponse(issue.model_dump())

    @require_auth
    async def handle_issue_get(request: Request):
        issue_id = request.path_params["id"]
        issue = await provider.get_issue(issue_id)
        if not issue:
            return JSONResponse({"error": "not found"}, status_code=404)
        return JSONResponse(issue.model_dump())

    @require_auth
    async def handle_issue_update(request: Request):
        issue_id = request.path_params["id"]
        issue = await provider.get_issue(issue_id)
        if not issue:
            return JSONResponse({"error": "not found"}, status_code=404)
        data = await request.json()
        for field in ("title", "description", "status", "priority", "assigned_agent", "scenario_id", "notebook_id"):
            if field in data:
                setattr(issue, field, data[field])
        if "labels" in data:
            issue.labels = data["labels"]
        await provider.update_issue(issue)
        return JSONResponse({"status": "ok"})

    @require_auth
    async def handle_issue_delete(request: Request):
        issue_id = request.path_params["id"]
        if await provider.delete_issue(issue_id):
            return JSONResponse({"status": "ok"})
        return JSONResponse({"error": "not found"}, status_code=404)

    @require_auth
    async def handle_generate_notebook(request: Request):
        issue_id = request.path_params["id"]
        issue = await provider.get_issue(issue_id)
        if not issue:
            return JSONResponse({"error": "issue not found"}, status_code=404)

        data = await request.json() if request.headers.get("content-type") == "application/json" else {}
        scenario_id = data.get("scenario_id") or issue.scenario_id
        if not scenario_id:
            return JSONResponse({"error": "scenario_id is required"}, status_code=400)

        nb = scenario_registry.generate_notebook(scenario_id, issue, notebook_store)
        if not nb:
            return JSONResponse({"error": f"scenario '{scenario_id}' not found"}, status_code=404)

        issue.notebook_id = nb.id
        issue.scenario_id = scenario_id
        issue.status = "in_progress"
        await provider.update_issue(issue)

        return JSONResponse({"status": "ok", "notebook_id": nb.id, "notebook_title": nb.title})

    # --- Scenarios ---

    @require_auth
    async def handle_scenarios_list(request: Request):
        scenarios = scenario_registry.list_scenarios()
        return JSONResponse([s.model_dump() for s in scenarios])

    @require_auth
    async def handle_scenario_get(request: Request):
        scenario_id = request.path_params["id"]
        s = scenario_registry.get_scenario(scenario_id)
        if not s:
            return JSONResponse({"error": "not found"}, status_code=404)
        result = s.model_dump()
        template = scenario_registry.get_template(scenario_id)
        if template:
            result["template"] = template
        return JSONResponse(result)

    return [
        Route("/admin/api/projects", handle_projects_list),
        Route("/admin/api/projects", handle_projects_create, methods=["POST"]),
        Route("/admin/api/projects/{id}", handle_project_get),
        Route("/admin/api/projects/{id}", handle_project_update, methods=["PUT"]),
        Route("/admin/api/projects/{id}", handle_project_delete, methods=["DELETE"]),
        Route("/admin/api/projects/{id}/issues", handle_issues_list),
        Route("/admin/api/projects/{id}/issues", handle_issues_create, methods=["POST"]),
        Route("/admin/api/issues/{id}", handle_issue_get),
        Route("/admin/api/issues/{id}", handle_issue_update, methods=["PUT"]),
        Route("/admin/api/issues/{id}", handle_issue_delete, methods=["DELETE"]),
        Route("/admin/api/issues/{id}/generate-notebook", handle_generate_notebook, methods=["POST"]),
        Route("/admin/api/scenarios", handle_scenarios_list),
        Route("/admin/api/scenarios/{id}", handle_scenario_get),
    ]
