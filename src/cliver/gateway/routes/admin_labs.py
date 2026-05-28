"""AI Lab routes for the admin portal."""

from __future__ import annotations

import json
import logging
from typing import Callable

from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

logger = logging.getLogger(__name__)


def get_lab_routes(lab_store, context: dict, require_auth: Callable) -> list:
    """Return AI Lab CRUD, chat, and golden test API routes."""
    from cliver.gateway.admin import _run_in_thread

    # -- Lab CRUD -----------------------------------------------------------

    @require_auth
    async def handle_list_labs(request: Request):
        labs = await _run_in_thread(lab_store.list_labs)
        return JSONResponse([lab.model_dump() for lab in labs])

    @require_auth
    async def handle_create_lab(request: Request):
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "Invalid JSON"}, status_code=400)
        title = (body.get("title") or "").strip()
        if not title:
            return JSONResponse({"error": "title is required"}, status_code=400)
        description = body.get("description", "")
        lab = await _run_in_thread(lab_store.create_lab, title, description)
        # Create a 1:1 session for this lab
        session_manager = context.get("cli_session_manager")
        if session_manager:
            await _run_in_thread(session_manager.create_lab_session, lab.id)
        return JSONResponse(lab.model_dump())

    @require_auth
    async def handle_get_lab(request: Request):
        lab_id = request.path_params["lab_id"]
        lab = await _run_in_thread(lab_store.get_lab, lab_id)
        if lab is None:
            return JSONResponse({"error": "Lab not found"}, status_code=404)
        session_manager = context.get("cli_session_manager")
        sessions = []
        session_id = None
        if session_manager:
            sessions = await _run_in_thread(session_manager.list_lab_sessions, lab_id)
            # Use the first session as the primary lab session (1:1 relationship)
            if sessions:
                session_id = sessions[0]["id"]
            else:
                # Create one if it doesn't exist yet
                session_id = await _run_in_thread(session_manager.create_lab_session, lab_id)
        return JSONResponse({"lab": lab.model_dump(), "sessions": sessions, "session_id": session_id})

    @require_auth
    async def handle_update_lab(request: Request):
        lab_id = request.path_params["lab_id"]
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "Invalid JSON"}, status_code=400)
        title = body.get("title", "")
        description = body.get("description", "")
        lab = await _run_in_thread(lab_store.update_lab, lab_id, title, description)
        if lab is None:
            return JSONResponse({"error": "Lab not found"}, status_code=404)
        return JSONResponse(lab.model_dump())

    @require_auth
    async def handle_delete_lab(request: Request):
        lab_id = request.path_params["lab_id"]
        deleted = await _run_in_thread(lab_store.delete_lab, lab_id)
        if not deleted:
            return JSONResponse({"error": "Lab not found"}, status_code=404)
        return JSONResponse({"status": "deleted"})

    # -- Golden Tests -------------------------------------------------------

    @require_auth
    async def handle_list_tests(request: Request):
        lab_id = request.path_params["lab_id"]
        tests = await _run_in_thread(lab_store.list_golden_tests, lab_id)
        return JSONResponse([t.model_dump() for t in tests])

    @require_auth
    async def handle_create_test(request: Request):
        lab_id = request.path_params["lab_id"]
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "Invalid JSON"}, status_code=400)
        name = (body.get("name") or "").strip()
        if not name:
            return JSONResponse({"error": "name is required"}, status_code=400)
        test = await _run_in_thread(
            lab_store.create_golden_test,
            lab_id,
            name,
            body.get("input", ""),
            body.get("expected_output", ""),
            body.get("expected_files", "[]"),
        )
        return JSONResponse(test.model_dump())

    @require_auth
    async def handle_update_test(request: Request):
        test_id = request.path_params["test_id"]
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "Invalid JSON"}, status_code=400)
        test = await _run_in_thread(
            lab_store.update_golden_test,
            test_id,
            body.get("name"),
            body.get("input"),
            body.get("expected_output"),
            body.get("expected_files"),
        )
        if test is None:
            return JSONResponse({"error": "Golden test not found"}, status_code=404)
        return JSONResponse(test.model_dump())

    @require_auth
    async def handle_delete_test(request: Request):
        test_id = request.path_params["test_id"]
        deleted = await _run_in_thread(lab_store.delete_golden_test, test_id)
        if not deleted:
            return JSONResponse({"error": "Golden test not found"}, status_code=404)
        return JSONResponse({"status": "deleted"})

    @require_auth
    async def handle_run_tests(request: Request):
        """Run all golden tests for a lab, return results with actual outputs."""
        lab_id = request.path_params["lab_id"]
        gateway = context.get("gateway")
        if not gateway:
            return JSONResponse({"error": "Agent not available"}, status_code=503)

        tests = await _run_in_thread(lab_store.list_golden_tests, lab_id)

        results = []
        for test in tests:
            try:
                agent = gateway._get_agent()
                response = await agent.chat(prompt=test.input)
                actual_text = response.message.text or ""
            except Exception as e:
                actual_text = f"Error: {e}"

            results.append(
                {
                    "test_id": test.id,
                    "name": test.name,
                    "input": test.input,
                    "expected_output": test.expected_output,
                    "actual_output": actual_text,
                    "expected_files": test.expected_files,
                }
            )

        return JSONResponse({"results": results})

    # -- Lab Sessions (config-only) ----------------------------------------

    @require_auth
    async def handle_create_lab_session(request: Request):
        """Create a lab session with options but without running the LLM."""
        session_manager = context.get("cli_session_manager")
        if not session_manager:
            return JSONResponse({"error": "Session manager not available"}, status_code=503)

        lab_id = request.path_params["lab_id"]
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "Invalid JSON"}, status_code=400)

        options = body.get("options") or {}
        session_id = await _run_in_thread(session_manager.create_lab_session, lab_id, "", options)
        return JSONResponse({"session_id": session_id, "options": options})

    # -- Lab Chat -----------------------------------------------------------

    @require_auth
    async def handle_lab_chat(request: Request):
        """Create or continue a lab chat session with SSE streaming."""
        gateway = context.get("gateway")
        if not gateway or not getattr(gateway, "_agent_core", None):
            return JSONResponse({"error": "Agent not available"}, status_code=503)

        session_manager = context.get("cli_session_manager")
        lab_id = request.path_params["lab_id"]
        session_id_param = request.path_params.get("session_id")

        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

        prompt = (body.get("message") or body.get("prompt", "")).strip()
        if not prompt:
            return JSONResponse({"error": "'message' is required"}, status_code=400)

        model = body.get("model") or gateway._get_default_model_name()
        system_message = body.get("system_message")
        agent_name = body.get("agent", "").strip()
        raw_history = body.get("conversation_history") or []
        tool_names = body.get("filter_tools")
        mcp_server_ids = body.get("mcp_server_ids") or []
        # Resolve MCP server IDs to names for tool prefix filtering.
        # Run in thread to avoid blocking the event loop.
        mcp_server_names: list[str] = []
        if mcp_server_ids:
            mcp_store = context.get("mcp_store")
            if mcp_store:
                all_servers = await _run_in_thread(mcp_store.list_servers)
                id_to_name = {s.id: s.name for s in all_servers}
                mcp_server_names = [id_to_name[sid] for sid in mcp_server_ids if sid in id_to_name]

        # Resolve agent config
        agent_role = None
        if agent_name:
            config = getattr(gateway, "_get_config_manager", None)
            if config:
                try:
                    cfg = config().config
                    agents = getattr(cfg, "agents", {}) or {}
                    agent_cfg = agents.get(agent_name)
                    if agent_cfg:
                        agent_model = getattr(agent_cfg, "model", None)
                        if agent_model:
                            model = agent_model
                        agent_role = getattr(agent_cfg, "role", None)
                except Exception:
                    pass

        # Session management
        session_id = None
        if session_id_param:
            existing = session_manager.get_session_info(session_id_param) if session_manager else None
            if not existing:
                return JSONResponse({"error": "Session not found"}, status_code=404)
            session_id = session_id_param
            if not raw_history and session_manager:
                loaded = session_manager.load_turns(session_id)
                raw_history = [{"role": t["role"], "content": t["content"]} for t in loaded]
        elif session_manager:
            session_options = {}
            if agent_name:
                session_options["agent"] = agent_name
            if model:
                session_options["model"] = model
            if system_message:
                session_options["system_prompt"] = system_message
            if tool_names:
                session_options["skills"] = tool_names
            if mcp_server_ids:
                session_options["mcp_servers"] = mcp_server_ids
            session_options = session_options if session_options else None
            session_id = session_manager.create_lab_session(lab_id, options=session_options)

        if session_manager and session_id:
            try:
                session_manager.append_turn(session_id, "user", prompt)
            except Exception as e:
                logger.warning("Failed to persist user turn: %s", e)

        # Build conversation history
        conversation_history = None
        if raw_history:
            from cliver.messages import CLIverMessage

            conversation_history = []
            for msg in raw_history:
                role = msg.get("role", "")
                content = msg.get("content", "")
                vendor = {}
                if "reasoning_content" in msg:
                    vendor["reasoning_content"] = msg["reasoning_content"]
                if role in ("user", "assistant"):
                    conversation_history.append(CLIverMessage(role=role, content=content, vendor_ext=vendor))

        def _system_appender():
            parts = []
            if agent_role:
                parts.append(agent_role)
            if system_message:
                parts.append(system_message)
            parts.append(
                "\n## Server Mode\n\n"
                "You are running as a backend API service via the admin portal. "
                "Make autonomous decisions. Be concise and direct. "
                "After a tool call completes, summarize the result briefly — "
                "do NOT repeat the tool input or arguments you already provided."
            )
            return "\n".join(parts)

        _tool_filter = None
        if tool_names or mcp_server_names:
            allowed_skills = set(tool_names) if tool_names else set()
            allowed_mcp_prefixes = {f"{n}#" for n in mcp_server_names} if mcp_server_names else set()

            async def _tool_filter(user_input, tools):
                result = []
                for t in tools:
                    name = t.name
                    if name in allowed_skills:
                        result.append(t)
                    elif allowed_mcp_prefixes and any(name.startswith(p) for p in allowed_mcp_prefixes):
                        result.append(t)
                    elif not allowed_skills and not allowed_mcp_prefixes:
                        result.append(t)
                return result

        uses_thinking = True

        async def generate():
            if session_id:
                yield f"data: {json.dumps({'type': 'session', 'session_id': session_id})}\n\n".encode()

            logger.info(
                "Lab chat start — lab=%s session=%s model=%s",
                lab_id,
                session_id or "new",
                model,
            )

            full_text = ""
            try:
                agent = gateway._get_agent(model)
                async for chunk in agent.stream(prompt=prompt):
                    if chunk.content:
                        full_text += chunk.content
                        data = json.dumps({"type": "content", "content": chunk.content})
                        yield f"data: {data}\n\n".encode()

                # Strip inline tool_calls JSON from the accumulated text
                # (some models emit tool calls as text rather than structured)
                from cliver.llm.llm_utils import strip_tool_calls_from_text

                clean_text = strip_tool_calls_from_text(full_text)

                if session_manager and session_id:
                    try:
                        session_manager.append_turn(session_id, "assistant", clean_text)
                    except Exception as e:
                        logger.warning("Failed to persist assistant turn: %s", e)

                done_data = {
                    "type": "done",
                    "content": clean_text,
                    "session_id": session_id,
                }
                if uses_thinking:
                    done_data["reasoning_content"] = ""
                yield f"data: {json.dumps(done_data)}\n\n".encode()

            except Exception as e:
                logger.error("Lab chat streaming error: %s", e, exc_info=True)
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n".encode()

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    @require_auth
    async def handle_lab_chat_save_config(request: Request):
        """Save session config options."""
        session_manager = context.get("cli_session_manager")
        if not session_manager:
            return JSONResponse({"error": "Session manager not available"}, status_code=503)

        session_id = request.path_params["session_id"]
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "Invalid JSON"}, status_code=400)

        options = body.get("options") or {}
        await _run_in_thread(session_manager.merge_options, session_id, options)
        return JSONResponse({"status": "updated"})

    return [
        # Lab CRUD
        Route("/admin/api/labs", handle_list_labs),
        Route("/admin/api/labs", handle_create_lab, methods=["POST"]),
        Route("/admin/api/labs/{lab_id}", handle_get_lab),
        Route("/admin/api/labs/{lab_id}", handle_update_lab, methods=["PATCH"]),
        Route("/admin/api/labs/{lab_id}", handle_delete_lab, methods=["DELETE"]),
        # Golden tests
        Route("/admin/api/labs/{lab_id}/golden-tests", handle_list_tests),
        Route("/admin/api/labs/{lab_id}/golden-tests", handle_create_test, methods=["POST"]),
        Route("/admin/api/labs/{lab_id}/golden-tests/{test_id}", handle_update_test, methods=["PATCH"]),
        Route("/admin/api/labs/{lab_id}/golden-tests/{test_id}", handle_delete_test, methods=["DELETE"]),
        Route("/admin/api/labs/{lab_id}/golden-tests/run", handle_run_tests, methods=["POST"]),
        # Lab sessions (create without running LLM)
        Route("/admin/api/labs/{lab_id}/sessions", handle_create_lab_session, methods=["POST"]),
        # Lab chat
        Route("/admin/api/labs/{lab_id}/chat", handle_lab_chat, methods=["POST"]),
        Route("/admin/api/labs/{lab_id}/chat/{session_id}", handle_lab_chat, methods=["POST"]),
        Route("/admin/api/labs/{lab_id}/chat/{session_id}", handle_lab_chat_save_config, methods=["PATCH"]),
    ]
