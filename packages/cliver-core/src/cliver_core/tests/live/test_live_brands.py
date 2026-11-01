"""Live: every chat preset still completes a real request.

One short, temperature-0 request per brand/protocol leaf.  A brand whose key
is absent warns and skips; the rest still run.  A failure names the key
variable, protocol, model, and resolved endpoint — never the key value.

An account may bill one endpoint differently from another (a GLM Coding Plan,
for example, covers ``/api/coding/paas/v4`` but not the pay-as-you-go
``/api/paas/v4``).  Override a leaf's endpoint with
``CLIVER_LIVE_BASE_URL_<BRAND>`` (upper-case, ``-`` becomes ``_``)::

    CLIVER_LIVE_BASE_URL_GLM=https://open.bigmodel.cn/api/coding/paas/v4

The Bailian workspace-dedicated host is exercised separately; set
``CLIVER_LIVE_WORKSPACE_ID`` (and optionally ``CLIVER_LIVE_REGION``) to a
workspace the ``DASHSCOPE_API_KEY`` belongs to.
"""

from __future__ import annotations

import os

import pytest

from cliver_core import AgentCore, ModelBindings, ProviderError, presets
from cliver_core.tests.live.support import (
    LIVE_OFF_REASON,
    live_enabled,
    skip_unless_keys,
    skip_with_warning,
)

pytestmark = pytest.mark.skipif(not live_enabled(), reason=LIVE_OFF_REASON)

CASES = [
    pytest.param("deepseek", "DEEPSEEK_API_KEY", presets.deepseek, id="deepseek"),
    pytest.param("deepseek-anthropic", "DEEPSEEK_API_KEY", presets.deepseek_anthropic, id="deepseek-anthropic"),
    pytest.param("glm", "GLM_API_KEY", presets.glm, id="glm"),
    pytest.param("glm-anthropic", "GLM_API_KEY", presets.glm_anthropic, id="glm-anthropic"),
    pytest.param("minimax-anthropic", "MINIMAX_API_KEY", presets.minimax_anthropic, id="minimax-anthropic"),
    pytest.param("qwen", "DASHSCOPE_API_KEY", presets.qwen, id="qwen"),
]


def _base_url_override(brand: str) -> str:
    name = "CLIVER_LIVE_BASE_URL_" + brand.upper().replace("-", "_")
    return os.environ.get(name, "").strip()


@pytest.mark.parametrize(("brand", "env_var", "factory"), CASES)
async def test_brand_chat_smoke(brand: str, env_var: str, factory) -> None:
    skip_unless_keys(env_var)
    override = _base_url_override(brand)
    binding = factory(temperature=0, **({"base_url": override} if override else {}))
    endpoint = binding.get_provider().base_url or "(no base_url)"
    agent = AgentCore(ModelBindings(llm=binding), run_timeout=120)

    try:
        response = await agent.chat("Reply with exactly the word OK.")
        text = (response.message.content or "").strip().lower()
        assert "ok" in text, f"{brand} answered unexpectedly: {text[:200]!r}"
        assert response.usage_total is not None
    except ProviderError as exc:
        pytest.fail(
            f"{brand} chat failed — key {env_var}, protocol {binding.protocol}, "
            f"model {binding.model}, endpoint {endpoint}: {exc}",
            pytrace=False,
        )
    finally:
        await agent.aclose()


async def test_bailian_workspace_host_chat_smoke() -> None:
    """A workspace-scoped key reaches Bailian's dedicated maas host.

    The shared ``dashscope.aliyuncs.com`` host is covered by the ``qwen``
    case above.  This proves the preset's workspace composition against the
    real service.  Set ``CLIVER_LIVE_WORKSPACE_ID`` (or
    ``CLIVER_SYSTEM_ONE_WORKSPACE_ID``) to the workspace the key belongs to.
    """
    skip_unless_keys("DASHSCOPE_API_KEY")
    workspace_id = (
        os.environ.get("CLIVER_LIVE_WORKSPACE_ID", "").strip()
        or os.environ.get("CLIVER_SYSTEM_ONE_WORKSPACE_ID", "").strip()
    )
    if not workspace_id:
        skip_with_warning(
            "live verification skipped: set CLIVER_LIVE_WORKSPACE_ID to exercise the Bailian workspace-dedicated host"
        )

    region = os.environ.get("CLIVER_LIVE_REGION", "").strip() or "cn-beijing"
    binding = presets.qwen(workspace_id=workspace_id, region=region, temperature=0)
    endpoint = binding.get_provider().base_url
    assert endpoint.startswith(f"https://{workspace_id}."), endpoint
    agent = AgentCore(ModelBindings(llm=binding), run_timeout=120)

    try:
        response = await agent.chat("Reply with exactly the word OK.")
        text = (response.message.content or "").strip().lower()
        assert "ok" in text, f"bailian workspace host answered unexpectedly: {text[:200]!r}"
    except ProviderError as exc:
        pytest.fail(
            f"bailian workspace chat failed — key DASHSCOPE_API_KEY, model {binding.model}, endpoint {endpoint}: {exc}",
            pytrace=False,
        )
    finally:
        await agent.aclose()
