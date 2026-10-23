"""
Test that cliver_core's import closure sticks to its declared hard deps.

Hard deps (per pyproject): pydantic, openai, anthropic, httpx, mcp.  The MCP
SDK is still blocked below — its import is deferred to ``cliver_core.mcp``
adapters, so plain ``import cliver_core`` stays light.  Everything else — CLI,
gateway, tooling libraries — must not be pulled in by an import.

CraftForge pattern: spawn a subprocess with a sys.meta_path finder that raises
ImportError for any module on the blocklist, then import cliver_core and assert
no exception. Runs in a subprocess because sys.meta_path blockers installed in
the test process cannot retroactively un-import already-loaded modules.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

BLOCKED = frozenset(
    {
        "click",
        "rich",
        "prompt_toolkit",
        "dotenv",
        "mcp",
        "yaml",
        "jinja2",
        "requests",
        "sqlalchemy",
        "apscheduler",
        "starlette",
        "uvicorn",
        "firecrawl",
        "playwright",
    }
)


_INSTALL_AND_IMPORT = f"""
import sys, importlib.abc

class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        top = fullname.split(".", 1)[0]
        if top in {set(BLOCKED)!r}:
            raise ImportError(f"blocked: {{fullname}}")
        return None

sys.meta_path.insert(0, Blocker())

import cliver_core
import cliver_core.agent
import cliver_core.bindings
import cliver_core.capabilities
import cliver_core.capabilities.asr
import cliver_core.capabilities.embed
import cliver_core.capabilities.image
import cliver_core.capabilities.rerank
import cliver_core.capabilities.tts
import cliver_core.capabilities.video
import cliver_core.media_jobs
import cliver_core.messages
import cliver_core.presets
import cliver_core.tool
import cliver_core.media
import cliver_core.events
import cliver_core.util
import cliver_core.constants
import cliver_core.provider
import cliver_core.provider.transport
import cliver_core.provider.transports
import cliver_core.provider.providers
import cliver_core.tools
# NOTE: cliver_core.mcp is intentionally NOT imported here — the MCP SDK is
# a hard dependency but imported lazily, so `import cliver_core` never loads it.
print("OK")
"""


def test_cliver_core_imports_with_only_declared_deps() -> None:
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(_INSTALL_AND_IMPORT)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"cliver_core import closure leaked blocked deps.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "OK" in result.stdout
