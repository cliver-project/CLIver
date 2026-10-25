"""ToolRegistry advertising: registration is enablement.

``builtin_tools=[...]`` / ``register()`` means "this tool is on" — the
registry advertises exactly what was registered, and AgentCore can only
execute what it advertised (pinned end-to-end in ``test_chat_roundtrip``).
"""

from __future__ import annotations

import importlib
import sys

import pytest

from cliver_core.tool import CLIverTool, ToolRegistry


def _tool(name: str) -> CLIverTool:
    return CLIverTool(
        name=name,
        description=f"{name} test tool",
        parameters={"type": "object", "properties": {}},
        execute=lambda: [],
    )


def _write_tool_module(package, module: str, tool_name: str) -> None:
    (package / f"{module}.py").write_text(
        "from cliver_core.tool import tool\n"
        "\n"
        "\n"
        f"@tool(name={tool_name!r}, description='test tool')\n"
        f"def {module.lstrip('_')}() -> list[dict]:\n"
        "    return []\n"
    )


def _forget_package(name: str) -> None:
    for module_name in list(sys.modules):
        if module_name == name or module_name.startswith(f"{name}."):
            sys.modules.pop(module_name, None)


def test_discovery_scans_every_public_module(tmp_path, monkeypatch) -> None:
    package = tmp_path / "scanpkg"
    package.mkdir()
    (package / "__init__.py").write_text("")
    _write_tool_module(package, "alpha", "Alpha")
    _write_tool_module(package, "_helper", "Hidden")

    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    try:
        from cliver_core.tools import _discover

        found = _discover("scanpkg", [str(package)])
    finally:
        _forget_package("scanpkg")

    assert [tool.name for tool in found] == ["Alpha"]


def test_discovery_rejects_duplicate_tool_names(tmp_path, monkeypatch) -> None:
    package = tmp_path / "duppkg"
    package.mkdir()
    (package / "__init__.py").write_text("")
    _write_tool_module(package, "one", "Dup")
    _write_tool_module(package, "two", "Dup")

    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    try:
        from cliver_core.tools import _discover

        with pytest.raises(RuntimeError, match="duplicate builtin tool name 'Dup'"):
            _discover("duppkg", [str(package)])
    finally:
        _forget_package("duppkg")


def test_registry_advertises_every_registered_tool() -> None:
    registry = ToolRegistry([_tool("Read"), _tool("Browser"), _tool("sum")])

    assert [t.name for t in registry.all_tools] == ["Browser", "Read", "sum"]


def test_register_adds_a_tool_to_the_advertised_set() -> None:
    registry = ToolRegistry()
    registry.register(_tool("sum"))
    registry.register_all([_tool("Read")])

    assert [t.name for t in registry.all_tools] == ["Read", "sum"]


def test_discover_builtin_tools_delegates_to_the_tools_package() -> None:
    from cliver_core.tool import discover_builtin_tools
    from cliver_core.tools import discover_builtin_tools as tools_discover

    discovered = discover_builtin_tools()
    assert [tool.name for tool in discovered] == [tool.name for tool in tools_discover()]
    assert len(discovered) == 9
