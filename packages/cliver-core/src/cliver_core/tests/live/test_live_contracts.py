"""Live: structured output, the run deadline, and credential rotation.

These are the harness-facing contracts an application builds on; each check
runs against DeepSeek with real wire traffic.
"""

from __future__ import annotations

import os
from typing import Literal

import pytest
from pydantic import BaseModel

from cliver_core import AgentCore, ModelBindings, ProviderError, RunTimeoutError, presets
from cliver_core.tests.live.support import LIVE_OFF_REASON, live_enabled, skip_unless_keys

pytestmark = pytest.mark.skipif(not live_enabled(), reason=LIVE_OFF_REASON)


class PartVerdict(BaseModel):
    part_id: str
    verdict: Literal["pass", "fail"]
    confidence: float


async def test_chat_structured_returns_validated_value() -> None:
    skip_unless_keys("DEEPSEEK_API_KEY")
    agent = AgentCore(ModelBindings(llm=presets.deepseek(temperature=0)), run_timeout=120)

    try:
        result = await agent.chat_structured(
            "Judge part P-1 from the inspection log: it passed. "
            "Return part_id, verdict ('pass' or 'fail'), and confidence.",
            schema=PartVerdict,
            run_id="live-structured",
        )
        assert isinstance(result.value, PartVerdict)
        assert result.value.part_id == "P-1"
        assert result.value.verdict == "pass"
        assert result.attempts == 1
        assert result.usage_total is not None
        assert result.run_id == "live-structured"
        assert result.response.run_id == "live-structured"
        # auto mode on the OpenAI protocol sends a json_object wire option.
        options = result.response.message.trace[0]["request"]["options"]
        assert options.get("response_format") == {"type": "json_object"}
    finally:
        await agent.aclose()


async def test_run_timeout_raises_before_the_model_answers() -> None:
    skip_unless_keys("DEEPSEEK_API_KEY")
    agent = AgentCore(ModelBindings(llm=presets.deepseek(temperature=0)))

    try:
        with pytest.raises(RunTimeoutError) as excinfo:
            await agent.chat("Say OK.", run_timeout=0.001)
        assert excinfo.value.timeout == 0.001
        assert excinfo.value.run_id
    finally:
        await agent.aclose()


async def test_credential_rotation_recovers_from_a_bad_key() -> None:
    skip_unless_keys("DEEPSEEK_API_KEY")
    agent = AgentCore(
        ModelBindings(llm=presets.deepseek(api_key="sk-live-verification-invalid")),
        run_timeout=120,
    )

    try:
        with pytest.raises(ProviderError):
            await agent.chat("Reply with exactly: OK")

        agent.llm.api_key = os.environ["DEEPSEEK_API_KEY"]
        await agent.arefresh_credentials()

        response = await agent.chat("Reply with exactly: OK", options={"temperature": 0})
        assert "ok" in (response.message.content or "").lower()
    finally:
        await agent.aclose()
