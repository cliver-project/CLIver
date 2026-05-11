"""Pure node runner functions for workflow steps.

Each runner receives extracted inputs and returns a JSON-serializable dict.
Neither runner touches LangGraph state directly.
"""

import importlib.util
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def run_python_node(
    inputs: Dict[str, Any],
    *,
    file_path: Optional[str] = None,
    code: Optional[str] = None,
    state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run a Python step — from a file or inline code snippet.

    The code must define ``run(inputs, state) -> dict``.
    For backward compat, ``run(inputs) -> dict`` is also accepted.
    """
    import inspect

    if file_path:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Python step file not found: {file_path}")
        spec = importlib.util.spec_from_file_location("workflow_step", str(path))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    elif code:
        import types

        module = types.ModuleType("workflow_step_inline")
        exec(code, module.__dict__)  # noqa: S102 — workflow code is user-authored
    else:
        raise ValueError("Python step requires either file_path or code")

    if not hasattr(module, "run"):
        raise AttributeError("Python step must define a run(inputs, state) -> dict function")

    sig = inspect.signature(module.run)
    if len(sig.parameters) >= 2:
        result = module.run(inputs, state or {})
    else:
        result = module.run(inputs)
    if not isinstance(result, dict):
        result = {"result": result}
    return result


async def run_llm_node(
    *,
    prompt: str,
    agent_name: Optional[str],
    tools: Optional[List[str]],
    output_format: str,
    outputs_dir: str,
    step_id: str,
    app_config,
    on_tool_event=None,
) -> Dict[str, Any]:
    """Run an LLM step via AgentCore. Returns result dict with text + file paths."""
    from cliver.llm import AgentCore
    from cliver.permissions import PermissionManager, PermissionMode

    agent_config = app_config.get_agent(agent_name)

    llm_models = app_config.models
    mcp_servers = {name: s.model_dump() for name, s in app_config.mcpServers.items()}

    permission_mgr = PermissionManager()
    permission_mgr.set_mode(PermissionMode.YOLO)

    enabled_toolsets = None
    if tools:
        enabled_toolsets = tools

    agent_core = AgentCore(
        llm_models=llm_models,
        mcp_servers=mcp_servers,
        default_model=agent_config.model or app_config.default_model,
        permission_manager=permission_mgr,
        on_tool_event=on_tool_event,
        enabled_toolsets=enabled_toolsets,
        model_auto_fallback=agent_config.auto_fallback if agent_config.auto_fallback is not None else True,
    )

    from cliver.agent import CliverAgent

    step_output_dir = Path(outputs_dir) / step_id
    step_output_dir.mkdir(parents=True, exist_ok=True)

    cliver_agent = CliverAgent(name=step_id, config=agent_config, agent_core=agent_core)
    response = await cliver_agent.run(
        prompt,
        outputs_dir=str(step_output_dir),
    )

    raw_content = response.content if hasattr(response, "content") else str(response)
    if isinstance(raw_content, list):
        result_text = "".join(
            b.get("text", "") for b in raw_content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    else:
        result_text = raw_content

    # Persist text result so the status endpoint can read it
    (step_output_dir / "result.txt").write_text(result_text or "", encoding="utf-8")

    files = []
    if step_output_dir.exists():
        for f in step_output_dir.iterdir():
            if f.is_file() and not f.name.endswith(".log"):
                suffix = f.suffix.lower()
                file_type = "file"
                if suffix in (".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"):
                    file_type = "image"
                elif suffix in (".mp3", ".wav", ".ogg", ".flac"):
                    file_type = "audio"
                elif suffix in (".mp4", ".webm", ".avi"):
                    file_type = "video"
                files.append(
                    {
                        "type": file_type,
                        "path": f"{step_id}/{f.name}",
                    }
                )

    result = {"result": result_text}
    if files:
        result["files"] = files

    if output_format == "json":
        try:
            parsed = json.loads(result_text)
            if isinstance(parsed, dict):
                result = parsed
                if files:
                    result["files"] = files
        except (json.JSONDecodeError, TypeError):
            pass

    return result
