"""Live check — needs CLIVER_LIVE_VERIFICATION=1 and DEEPSEEK_API_KEY."""

from __future__ import annotations

from _shared.testing import skip_unless_live
from code_review import ReviewReport, build_agent, review_repo


async def test_code_review_live() -> None:
    skip_unless_live("DEEPSEEK_API_KEY")

    agent = build_agent()
    try:
        result = await review_repo(agent)
    finally:
        await agent.aclose()

    report = result.value
    assert isinstance(report, ReviewReport)
    assert report.findings
    assert result.attempts >= 1
    for finding in report.findings:
        assert finding.line >= 1
        assert finding.file.endswith(("aggregate.py", "loaders.py"))
