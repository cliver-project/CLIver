"""Live check — needs CLIVER_LIVE_VERIFICATION=1 and DEEPSEEK_API_KEY."""

from __future__ import annotations

import tempfile
from pathlib import Path

from _shared.testing import skip_unless_live
from safe_files import build_agent, run, seed_workspace


async def test_safe_files_live() -> None:
    skip_unless_live("DEEPSEEK_API_KEY")

    with tempfile.TemporaryDirectory(prefix="cliver-safe-files-") as tmp:
        workspace = seed_workspace(Path(tmp))
        agent = build_agent(workspace)
        try:
            response = await run(agent)
        finally:
            await agent.aclose()

        assert response.message.content
        assert not (workspace / "summary.md").exists()
