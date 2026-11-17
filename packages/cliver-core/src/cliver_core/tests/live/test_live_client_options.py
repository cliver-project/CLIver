"""Live: one ``ClientOptions`` object configures both real wire paths.

The unit suite pins the plumbing with fake clients; these checks prove the
injected ``httpx2`` client (pool limits) and the SDK's custom User-Agent
survive real traffic on both protocols.
"""

from __future__ import annotations

import pytest

from cliver_core import AgentCore, ClientOptions, ModelBindings, presets
from cliver_core.tests.live.support import LIVE_OFF_REASON, live_enabled, skip_unless_keys

pytestmark = pytest.mark.skipif(not live_enabled(), reason=LIVE_OFF_REASON)


async def test_openai_protocol_chat_runs_through_a_limited_pool() -> None:
    skip_unless_keys("DEEPSEEK_API_KEY")
    agent = AgentCore(
        ModelBindings(
            llm=presets.deepseek(
                temperature=0,
                client=ClientOptions(
                    user_agent="cliver-live-verification/1.0",
                    timeout=60,
                    max_connections=4,
                    max_keepalive_connections=2,
                    keepalive_expiry=10.0,
                ),
            )
        ),
        run_timeout=120,
    )

    try:
        response = await agent.chat("Reply with exactly: POOL-OK")
        assert "POOL-OK" in response.message.content

        client = agent.bindings.llm.build_provider().transport._client
        pool = client._client._transport._pool
        assert pool._max_connections == 4
        assert pool._max_keepalive_connections == 2
        assert pool._keepalive_expiry == 10.0
        assert client._custom_headers["User-Agent"] == "cliver-live-verification/1.0"
    finally:
        await agent.aclose()


async def test_anthropic_protocol_chat_runs_through_a_limited_pool() -> None:
    skip_unless_keys("MINIMAX_API_KEY")
    agent = AgentCore(
        ModelBindings(
            llm=presets.minimax_anthropic(
                temperature=0,
                client=ClientOptions(
                    max_connections=3,
                    max_keepalive_connections=1,
                    keepalive_expiry=10.0,
                ),
            )
        ),
        run_timeout=120,
    )

    try:
        response = await agent.chat("Reply with exactly: POOL-OK")
        assert "POOL-OK" in response.message.content

        pool = agent.bindings.llm.build_provider().transport._client._client._transport._pool
        assert pool._max_connections == 3
        assert pool._max_keepalive_connections == 1
    finally:
        await agent.aclose()
