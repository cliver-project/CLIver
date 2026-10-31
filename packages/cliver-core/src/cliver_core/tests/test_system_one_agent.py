"""AgentCore.system_one — binding defaults, failover, and the unbound error."""

from __future__ import annotations

from typing import Any

import pytest

from cliver_core.agent import AgentCore
from cliver_core.bindings import ModelBinding, ModelBindings
from cliver_core.errors import CapabilityNotSupported, ProviderError
from cliver_core.system_one import Noul, NoulAnswer, SystemOneResult


class _SystemOneStub:
    """Records every call; returns a canned result or raises a canned error."""

    def __init__(self, result: SystemOneResult | None = None, error: Exception | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.result = result
        self.error = error

    async def system_one(self, **kwargs) -> SystemOneResult:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        assert self.result is not None
        return self.result


def _result(value: float = 0.9) -> SystemOneResult:
    return SystemOneResult(answers={"q": NoulAnswer(noul=value)})


def _agent(*bindings: ModelBinding) -> AgentCore:
    mounts = {"llm": ModelBinding.llm(model="mock"), "system_one": list(bindings) if len(bindings) > 1 else bindings[0]}
    return AgentCore(ModelBindings(**mounts))


def _binding(provider: Any, **params: Any) -> ModelBinding:
    return ModelBinding.system_one(model="decision-model-preview", provider=provider, **params)


def test_agent_has_system_one_api() -> None:
    assert hasattr(AgentCore, "system_one")


@pytest.mark.asyncio
async def test_system_one_routes_through_the_binding() -> None:
    stub = _SystemOneStub(_result(0.75))
    agent = _agent(_binding(stub, timeout=5.0))

    result = await agent.system_one(
        "I was charged twice",
        {"billing": Noul(instructions="Is this about billing?")},
    )

    assert result.answers["q"].noul == pytest.approx(0.75)
    call = stub.calls[-1]
    assert call["state"] == "I was charged twice"
    assert call["model"] == "decision-model-preview"
    assert call["timeout"] == 5.0  # binding default
    assert isinstance(call["questions"]["billing"], Noul)
    await agent.aclose()


@pytest.mark.asyncio
async def test_call_site_options_override_binding_defaults() -> None:
    stub = _SystemOneStub(_result())
    agent = _agent(_binding(stub, timeout=5.0))

    await agent.system_one("x", {"q": Noul(instructions="Yes?")}, timeout=1.5)

    assert stub.calls[-1]["timeout"] == 1.5
    await agent.aclose()


@pytest.mark.asyncio
async def test_unbound_system_one_reports_the_capability() -> None:
    agent = AgentCore(ModelBindings(llm=ModelBinding.llm(model="mock")))

    with pytest.raises(CapabilityNotSupported, match="system_one"):
        await agent.system_one("x", {"q": Noul(instructions="Yes?")})
    await agent.aclose()


@pytest.mark.asyncio
async def test_a_retryable_failure_fails_over_to_the_next_binding() -> None:
    primary = _SystemOneStub(error=ProviderError("rate limited", retryable=True))
    backup = _SystemOneStub(_result(0.4))
    agent = _agent(_binding(primary), _binding(backup))

    result = await agent.system_one("x", {"q": Noul(instructions="Yes?")})

    assert result.answers["q"].noul == pytest.approx(0.4)
    assert len(primary.calls) == 1
    assert len(backup.calls) == 1
    await agent.aclose()


@pytest.mark.asyncio
async def test_a_non_retryable_failure_propagates_without_failover() -> None:
    primary = _SystemOneStub(error=ProviderError("bad request", retryable=False))
    backup = _SystemOneStub(_result())
    agent = _agent(_binding(primary), _binding(backup))

    with pytest.raises(ProviderError, match="bad request"):
        await agent.system_one("x", {"q": Noul(instructions="Yes?")})
    assert backup.calls == []
    await agent.aclose()


@pytest.mark.asyncio
async def test_a_closed_agent_rejects_system_one() -> None:
    agent = _agent(_binding(_SystemOneStub(_result())))
    await agent.aclose()

    with pytest.raises(RuntimeError, match="closed"):
        await agent.system_one("x", {"q": Noul(instructions="Yes?")})
