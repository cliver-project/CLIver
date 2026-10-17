"""cliver_core.mcp stays importable — and discovery usable — without the MCP SDK.

Subprocess + ``sys.meta_path`` blocker, mirroring ``test_core_closure.py``:
the SDK extra is absent in the test process, so ``MCPClient`` must demand it
with the pinned message while the module import and discovery keep working.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

BLOCKED = frozenset(
    {
        "mcp",
        "mcp_types",
        "httpx2",
        "httpcore2",
        "starlette",
        "sse_starlette",
        "uvicorn",
        "jsonschema",
        "pyjwt",
        "cryptography",
    }
)

_SCRIPT = f"""
import importlib.abc
import sys

class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        top = fullname.split(".", 1)[0]
        if top in {set(BLOCKED)!r}:
            raise ModuleNotFoundError(f"No module named '{{fullname}}'", name=fullname)
        return None

sys.meta_path.insert(0, Blocker())

import cliver_core.mcp  # noqa: F401

from cliver_core.mcp import MCPClient, discover_mcp_servers

leaked = [name for name in sys.modules if name.split(".", 1)[0] in {set(BLOCKED)!r}]
assert not leaked, f"MCP SDK modules were imported: {{leaked}}"

result = discover_mcp_servers(cwd="/nonexistent-cwd", home="/nonexistent-home", env={{}})
assert result.servers == {{}}, result.servers
assert result.summary() == "0 servers from 0 sources"

MCPClient({{}})  # an empty config never touches the SDK

try:
    MCPClient({{"filesystem": {{"transport": "stdio", "command": "npx"}}}})
except ImportError as exc:
    assert "cliver-core[mcp]" in str(exc), str(exc)
else:
    raise AssertionError("a real config must require the mcp extra")

print("OK")
"""


def test_mcp_module_is_importable_without_the_sdk() -> None:
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(_SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
    assert "OK" in result.stdout
