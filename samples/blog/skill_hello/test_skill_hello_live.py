"""Live check — needs CLIVER_LIVE_VERIFICATION=1 and DEEPSEEK_API_KEY."""

from __future__ import annotations

from _shared.testing import skip_unless_live
from skill_hello import build_agent, run


async def test_skill_hello_live() -> None:
    skip_unless_live("DEEPSEEK_API_KEY")

    agent = build_agent()
    try:
        response = await run(agent)
    finally:
        await agent.aclose()

    assert "こんにちは" in (response.message.content or "")
