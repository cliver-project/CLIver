"""Per-call snapshots: a call advertises the tools registered when it starts.

AgentCore reads the registry once per ``chat`` / ``stream`` / ``run_skill``
call.  Registering a tool while a call is in flight does not change that
call's advertised set — it appears on the next call.  Hosts should configure
the agent (registry, bindings, overrides) before serving traffic; mutating
shared state mid-call is not part of the contract.
"""

from __future__ import annotations

import pytest

from cliver_core.agent import AgentCore
from cliver_core.bindings import ModelBinding, ModelBindings
from cliver_core.messages import CLIverMessage
from cliver_core.provider import CLIverRequest, CLIverResponse, Provider
from cliver_core.tool import CLIverTool


def _tool(name: str) -> CLIverTool:
    return CLIverTool(
        name=name,
        description=name,
        parameters={"type": "object", "properties": {}},
        execute=lambda **_kwargs: [{"text": name}],
    )


class _MutatingProvider(Provider):
    protocol = "openai"
    brand = "fake-call-snapshot"

    def __init__(self) -> None:
        super().__init__(api_key="", base_url="")
        self.requests: list[CLIverRequest] = []
        self.agent: AgentCore | None = None

    async def chat(self, request: CLIverRequest) -> CLIverResponse:
        self.requests.append(request)
        if len(self.requests) == 1:
            assert self.agent is not None
            self.agent.tool_registry.register(_tool("late"))
        return CLIverResponse(message=CLIverMessage(role="assistant", content="ok"))

    async def stream(self, request):  # pragma: no cover
        raise AssertionError("stream() is not used here")


@pytest.mark.asyncio
async def test_a_tool_registered_mid_call_is_advertised_on_the_next_call() -> None:
    provider = _MutatingProvider()
    agent = AgentCore(
        ModelBindings(llm=ModelBinding.llm(model="mock-model", provider=provider)),
        builtin_tools=[_tool("early")],
    )
    provider.agent = agent

    await agent.chat("one")
    await agent.chat("two")

    first = [tool.name for tool in provider.requests[0].tools or []]
    second = [tool.name for tool in provider.requests[1].tools or []]
    assert first == ["early"]
    assert second == ["early", "late"]
