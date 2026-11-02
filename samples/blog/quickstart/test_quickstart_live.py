"""Live check — needs CLIVER_LIVE_VERIFICATION=1 and DEEPSEEK_API_KEY."""

from __future__ import annotations

from _shared.testing import skip_unless_live
from quickstart import run


async def test_quickstart_live() -> None:
    skip_unless_live("DEEPSEEK_API_KEY")

    response = await run()

    assert response.message.content
