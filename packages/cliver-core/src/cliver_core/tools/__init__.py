"""Builtin tools that ship with cliver-core.

Every non-underscore module in this package defines one or more ``CLIverTool``
objects via ``@tool``; :func:`discover_builtin_tools` imports them and collects
the tools. Modules prefixed with ``_`` are internal helpers and are skipped.

CLI-coupled tools (ask_user_question, cliver_help, create_task, memory,
search_sessions, web_search, image_generate, skill, browse_web, browser_action)
are re-derived in later phases.
"""

from __future__ import annotations

import importlib
import pkgutil

from cliver_core.tool import CLIverTool


def _discover(package_name: str, package_path: list[str]) -> list[CLIverTool]:
    """Import every public module in ``package_path`` and collect its tools."""
    found: dict[str, CLIverTool] = {}
    for info in pkgutil.iter_modules(package_path):
        if info.name.startswith("_"):
            continue
        module = importlib.import_module(f"{package_name}.{info.name}")
        for value in vars(module).values():
            if not isinstance(value, CLIverTool):
                continue
            existing = found.get(value.name)
            if existing is not None and existing is not value:
                raise RuntimeError(
                    f"duplicate builtin tool name {value.name!r} (second definition in {module.__name__})"
                )
            found[value.name] = value
    return [found[name] for name in sorted(found)]


def discover_builtin_tools() -> list[CLIverTool]:
    """Return the builtin tools defined in this package, sorted by name."""
    return _discover(__name__, __path__)


__all__ = ["discover_builtin_tools"]
