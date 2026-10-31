"""A ModelBinding belongs to one event loop.

A cached provider holds loop-bound clients (the SDK and its ``httpx`` pool,
stream tasks).  Using the same binding from a second loop — in practice, a
second thread — fails deep inside the client or hangs.  The binding records
the loop on first use and rejects a different one with a message that says
what to do: one binding (and agent) per event loop.  ``aclose`` releases the
claim, so a binding can be reused after its agent shuts down.
"""

from __future__ import annotations

import asyncio

import pytest

from cliver_core.agent import AgentCore
from cliver_core.bindings import ModelBinding, ModelBindings
from cliver_core.messages import CLIverMessage
from cliver_core.provider import CLIverRequest, CLIverResponse, Provider


class _FakeProvider(Provider):
    protocol = "openai"
    brand = "fake-loop-affinity"

    def __init__(self) -> None:
        super().__init__(api_key="", base_url="")
        self.calls = 0

    async def chat(self, request: CLIverRequest) -> CLIverResponse:
        self.calls += 1
        return CLIverResponse(message=CLIverMessage(role="assistant", content="ok"))

    async def stream(self, request):  # pragma: no cover
        raise AssertionError("stream() is not used here")


def _agent(binding: ModelBinding) -> AgentCore:
    return AgentCore(ModelBindings(llm=binding))


def test_binding_rejects_a_second_event_loop() -> None:
    binding = ModelBinding.llm(model="mock-model", provider=_FakeProvider())

    async def run() -> str:
        return (await _agent(binding).chat("hi")).message.content or ""

    assert asyncio.run(run()) == "ok"
    with pytest.raises(RuntimeError, match="another event loop"):
        asyncio.run(run())


def test_aclose_releases_loop_affinity() -> None:
    binding = ModelBinding.llm(model="mock-model", provider=_FakeProvider())

    async def run() -> None:
        agent = _agent(binding)
        await agent.chat("hi")
        await agent.aclose()

    asyncio.run(run())
    asyncio.run(run())  # the binding is free again after its agent closes


def test_aclose_from_another_loop_raises() -> None:
    binding = ModelBinding.llm(model="mock-model", provider=_FakeProvider())

    async def use() -> None:
        await _agent(binding).chat("hi")

    asyncio.run(use())
    with pytest.raises(RuntimeError, match="another event loop"):
        asyncio.run(binding.aclose())


@pytest.mark.asyncio
async def test_concurrent_calls_in_one_loop_share_the_binding() -> None:
    provider = _FakeProvider()
    binding = ModelBinding.llm(model="mock-model", provider=provider)
    agent = _agent(binding)

    await asyncio.gather(agent.chat("hi"), agent.chat("hi"))

    assert binding.get_provider() is provider
    assert provider.calls == 2
