"""ToolRegistry advertising: explicit registrations vs the toolset gate.

``builtin_tools=[custom]`` / ``register()`` means "this tool is on" — the
toolset enablement only governs the builtin toolsets, so a directly
registered tool is always advertised (and AgentCore can only execute what it
advertised; pinned end-to-end in ``test_chat_roundtrip``).
"""

from __future__ import annotations

from cliver_core.tool import TOOLSETS, CLIverTool, ToolRegistry


def _tool(name: str) -> CLIverTool:
    return CLIverTool(
        name=name,
        description=f"{name} test tool",
        parameters={"type": "object", "properties": {}},
        execute=lambda: [],
    )


def test_directly_registered_tools_bypass_the_toolset_gate() -> None:
    # "sum" belongs to no toolset — an explicit registration is the enablement
    registry = ToolRegistry([_tool("sum")])
    assert [t.name for t in registry.all_tools] == ["sum"]

    registry = ToolRegistry()
    registry.register(_tool("sum"))
    registry.configure(enabled_toolsets=["core"])
    assert [t.name for t in registry.all_tools] == ["sum"]


def test_toolset_tools_still_follow_their_gate() -> None:
    assert "Read" in TOOLSETS["core"]
    registry = ToolRegistry([_tool("Read"), _tool("Browser"), _tool("sum")])

    # default configure: core ✓ (always on), browser ✗ (opt-in toolset),
    # custom ✓ — the fix must not degrade into "advertise everything"
    assert [t.name for t in registry.all_tools] == ["Read", "sum"]
