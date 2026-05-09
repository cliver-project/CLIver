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


def run_python_node(file_path: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Load a .py file and call its run(inputs) -> dict function."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Python step file not found: {file_path}")

    spec = importlib.util.spec_from_file_location("workflow_step", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "run"):
        raise AttributeError(f"Python step file {file_path} must define a run(inputs: dict) -> dict function")

    result = module.run(inputs)
    if not isinstance(result, dict):
        result = {"result": result}
    return result


async def run_llm_node(
    *,
    prompt: str,
    model: Optional[str],
    role: Optional[str],
    tools: Optional[List[str]],
    output_format: str,
    outputs_dir: str,
    step_id: str,
    app_config,
    on_tool_event=None,
) -> Dict[str, Any]:
    """Run an LLM step via AgentCore. Returns result dict with text + file paths."""
    from cliver.llm import AgentCore
    from cliver.permissions import PermissionManager

    llm_models = app_config.get_llm_models()
    mcp_servers = app_config.get_mcp_servers_dict()

    permission_mgr = PermissionManager(mode="yolo")

    enabled_toolsets = None
    if tools:
        enabled_toolsets = tools

    agent = AgentCore(
        llm_models=llm_models,
        mcp_servers=mcp_servers,
        default_model=model or app_config.default_model,
        permission_manager=permission_mgr,
        on_tool_event=on_tool_event,
        enabled_toolsets=enabled_toolsets,
        model_auto_fallback=True,
    )

    system_appender = None
    if role:

        def system_appender():
            return f"\n\nYour role: {role}\n"

    step_output_dir = Path(outputs_dir) / step_id
    step_output_dir.mkdir(parents=True, exist_ok=True)

    response = await agent.process_user_input(
        user_input=prompt,
        model=model,
        system_message_appender=system_appender,
        outputs_dir=str(step_output_dir),
    )

    result_text = response.content if hasattr(response, "content") else str(response)

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
