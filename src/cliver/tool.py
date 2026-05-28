"""CLIverTool and the @tool decorator — no langchain dependency."""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Callable, Optional, get_args, get_origin

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ── JSON Schema helpers ────────────────────────────────────────


def _python_type_to_json_schema(annotation) -> dict:
    """Map a Python type hint to a JSON Schema type dict."""
    origin = get_origin(annotation)
    args = get_args(annotation)

    # Handle Optional[X] (Union[X, None]) and Union types
    if origin in (Optional, type(None) | str.__class__, ...):
        raise TypeError("Should not reach here")
    if hasattr(origin, "__name__") and origin.__name__ == "UnionType":
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _python_type_to_json_schema(non_none[0])
        return {"type": "string"}

    if annotation is str:
        return {"type": "string"}
    if annotation is int:
        return {"type": "integer"}
    if annotation is float:
        return {"type": "number"}
    if annotation is bool:
        return {"type": "boolean"}
    if annotation is Path:
        return {"type": "string", "format": "path"}

    if origin is list:
        item_schema = _python_type_to_json_schema(args[0]) if args else {}
        return {"type": "array", "items": item_schema}

    if origin is dict:
        return {"type": "object"}

    # Python 3.10+ UnionType
    if origin is type(int) | type(str):  # won't match, but safe
        return {"type": "string"}

    return {"type": "string"}  # fallback


def _is_optional(annotation) -> bool:
    """Check if a type hint is Optional[X] (i.e., allows None)."""
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is type(None) | str.__class__:
        return False
    if hasattr(origin, "__name__") and origin.__name__ == "UnionType":
        return type(None) in args
    return False


# ── CLIverTool ─────────────────────────────────────────────────


class CLIverTool(BaseModel):
    """A tool definition — what the LLM sees + how to execute it.

    ``execute`` is always a synchronous callable returning list[dict].
    Async functions are converted at registration time (via @tool decorator).
    AgentCore always calls execute via asyncio.to_thread() for event-loop safety.
    """

    name: str
    description: str  # one-liner for LLM tool schema
    long_description: str | None = None  # full docstring, for humans / help
    parameters: dict[str, Any]  # JSON Schema
    execute: Callable[..., list[dict]]

    model_config = {"arbitrary_types_allowed": True}

    def to_openai_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def to_anthropic_schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }


# ── Decorator ──────────────────────────────────────────────────


def tool(
    *,
    name: str | None = None,
    description: str = "",
) -> Callable:
    """Decorator: convert a sync or async function into a CLIverTool.

    JSON Schema for ``parameters`` is auto-generated from Python type hints.
    The function's docstring becomes ``long_description``.

    Sync functions are wrapped so ``execute`` is always synchronous.
    Async functions are wrapped via ``asyncio.run()`` — AgentCore calls
    ``execute`` via ``asyncio.to_thread()``, so each tool invocation
    gets its own event loop on a fresh thread.

    Example:

        @tool(description="Read a file from disk.")
        def read_file(path: str, offset: int = 0) -> list[dict]:
            \"\"\"Read a file from the local filesystem.

            Args:
                path: Path to read, relative to working directory.
                offset: Line number to start from (0-indexed).
            \"\"\"
            lines = Path(path).read_text().splitlines()
            if offset:
                lines = lines[offset:]
            return [{"text": "\\n".join(lines)}]
    """

    def decorator(fn):
        sig = inspect.signature(fn)

        # Build JSON Schema properties from type hints
        properties = {}
        required = []
        for param_name, param in sig.parameters.items():
            if param.annotation is inspect.Parameter.empty:
                properties[param_name] = {"type": "string"}
                required.append(param_name)
                continue

            properties[param_name] = _python_type_to_json_schema(
                param.annotation
            )
            if param.default is inspect.Parameter.empty and not _is_optional(
                param.annotation
            ):
                required.append(param_name)

        parameters: dict[str, Any] = {
            "type": "object",
            "properties": properties,
        }
        if required:
            parameters["required"] = required

        # Normalize execute to sync
        if inspect.iscoroutinefunction(fn):

            def _execute_sync(**kwargs):
                return asyncio.run(fn(**kwargs))

            execute = _execute_sync
        else:
            execute = fn

        return CLIverTool(
            name=name or fn.__name__,
            description=description or _first_line(inspect.getdoc(fn) or ""),
            long_description=inspect.getdoc(fn),
            parameters=parameters,
            execute=execute,
        )

    return decorator


def _first_line(text: str) -> str:
    """Extract the first non-empty line of a docstring."""
    for line in text.strip().splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


# ── Tool Registry ──────────────────────────────────────────────

# Toolset definitions: toolset_name → set of tool names
TOOLSETS: dict[str, set[str]] = {
    "core": {"Read", "Write", "LS", "Grep", "Bash", "Exec", "ImageGenerate"},
    "memory": {"MemoryRead", "MemoryWrite", "Identity", "SearchSessions"},
    "automation": {"Skill", "TodoRead", "TodoWrite", "CliverHelp"},
    "web": {"WebFetch", "WebSearch", "Browse"},
    "browser": {"Browser"},
    "container": {"Docker"},
}

_ALWAYS_ENABLED = {"core", "memory", "automation"}

_TOOLSET_CHECKS: dict[str, Callable[[], bool]] = {
    "web": lambda: True,
    "browser": lambda: False,  # opt-in via config
    "container": lambda: shutil.which("docker") is not None
    or shutil.which("podman") is not None,
}

_TOOL_CHECKS: dict[str, Callable[[], bool]] = {
    "Browse": lambda: bool(os.environ.get("FIRECRAWL_API_KEY")),
    "Browser": lambda: _playwright_ready(),
}


def _playwright_ready() -> bool:
    try:
        from playwright._impl._driver import compute_driver_executable

        return os.path.isfile(compute_driver_executable())
    except Exception:
        return False


class ToolRegistry:
    """Registry for builtin CLIverTools, filtered by toolset and environment."""

    def __init__(self, tools: list[CLIverTool] | None = None):
        self._by_name: dict[str, CLIverTool] = {}
        self._enabled_names: set[str] = set()
        if tools:
            for t in tools:
                self._by_name[t.name] = t

    def register(self, tool: CLIverTool) -> None:
        self._by_name[tool.name] = tool

    def register_all(self, tools: list[CLIverTool]) -> None:
        for t in tools:
            self._by_name[t.name] = t

    def _resolve_enabled(self, enabled_toolsets: list[str] | None = None) -> set[str]:
        """Determine which toolsets are enabled from config or environment."""
        if enabled_toolsets is not None:
            enabled = set(enabled_toolsets) | _ALWAYS_ENABLED
        else:
            enabled = set(_ALWAYS_ENABLED)
            for ts, check in _TOOLSET_CHECKS.items():
                try:
                    if check():
                        enabled.add(ts)
                except Exception:
                    pass

        names: set[str] = set()
        for ts in enabled:
            if ts in TOOLSETS:
                names.update(TOOLSETS[ts])

        # Apply per-tool environment checks
        for name in list(names):
            check = _TOOL_CHECKS.get(name)
            if check and not check():
                names.discard(name)

        return names

    def configure(self, enabled_toolsets: list[str] | None = None) -> None:
        """Re-compute which tools are enabled based on toolsets."""
        self._enabled_names = self._resolve_enabled(enabled_toolsets)

    @property
    def all_tools(self) -> list[CLIverTool]:
        """Get enabled tools (to send to LLM)."""
        if not self._enabled_names:
            self.configure()
        return [t for name, t in self._by_name.items() if name in self._enabled_names]

    def get(self, name: str) -> CLIverTool | None:
        return self._by_name.get(name)

    @property
    def tool_names(self) -> list[str]:
        return [t.name for t in self.all_tools]


# ── Auto-discovery ────────────────────────────────────────────


def discover_builtin_tools() -> list[CLIverTool]:
    """Scan ``cliver.tools`` for existing BaseTool instances and wrap as CLIverTool.

    This is a temporary bridge until all builtin tools are rewritten
    with the ``@tool`` decorator.
    """
    import inspect

    import cliver.tools as tools_module
    from langchain_core.tools import BaseTool

    wrapped: list[CLIverTool] = []
    for _name, obj in inspect.getmembers(tools_module):
        if not isinstance(obj, BaseTool):
            continue
        wrapped.append(_wrap_base_tool(obj))
    return wrapped


def _wrap_base_tool(bt) -> CLIverTool:
    """Wrap a langchain BaseTool as a CLIverTool."""
    from langchain_core.tools import BaseTool

    # Build JSON Schema from the tool's args_schema (pydantic model)
    parameters: dict[str, Any] = {"type": "object", "properties": {}}
    if hasattr(bt, "args_schema") and bt.args_schema:
        parameters = _pydantic_to_json_schema(bt.args_schema)

    def _execute(**kwargs) -> list[dict]:
        result = bt.invoke(input=kwargs)
        if isinstance(result, dict):
            return [result]
        if isinstance(result, list):
            return result
        if result is None:
            return [{}]
        return [{"tool_result": str(result)}]

    return CLIverTool(
        name=bt.name,
        description=bt.description or "",
        long_description=getattr(bt, "description", None),
        parameters=parameters,
        execute=_execute,
    )


def _pydantic_to_json_schema(model) -> dict:
    """Convert a Pydantic v2 model to JSON Schema dict."""
    import json

    schema = model.model_json_schema()
    # Remove pydantic-specific keys
    for key in ("title", "description", "$defs", "additionalProperties"):
        schema.pop(key, None)
    return schema
