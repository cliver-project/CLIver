"""Token usage: ``usage`` is the final call; ``usage_total`` sums the run.

A run's cost is the sum of every model call it makes — including the
intermediate calls that requested tools.  ``usage`` keeps its historical
meaning (the final response's own usage); ``usage_total`` carries the
aggregate, so existing callers see no change.
"""

from __future__ import annotations

import pytest

from cliver_core.agent import AgentCore
from cliver_core.bindings import ModelBinding, ModelBindings
from cliver_core.messages import CLIverMessage, ToolCall, UsageInfo
from cliver_core.provider import CLIverRequest, CLIverResponse, Provider
from cliver_core.tool import CLIverTool, tool


@tool(name="echo", description="Echoes the input back.")
def echo(value: str) -> str:
    return f"echo: {value}"


@tool(name="boom", description="Always fails.")
def boom(value: str) -> str:
    raise RuntimeError("boom")


def _agent(
    provider: Provider,
    *,
    tools: tuple[CLIverTool, ...] = (echo,),
    max_consecutive_errors: int = 5,
) -> AgentCore:
    return AgentCore(
        ModelBindings(llm=ModelBinding.llm(model="mock-model", provider=provider)),
        builtin_tools=list(tools),
        max_consecutive_errors=max_consecutive_errors,
    )


class _SingleTurnChatProvider(Provider):
    protocol = "openai"
    brand = "mock-usage-single"

    def __init__(self, usage: UsageInfo | None) -> None:
        super().__init__(api_key="", base_url="")
        self.usage = usage
        self.calls: list[CLIverRequest] = []

    async def chat(self, request: CLIverRequest) -> CLIverResponse:
        self.calls.append(request)
        return CLIverResponse(
            message=CLIverMessage(role="assistant", content="done"),
            usage=self.usage,
        )

    async def stream(self, request):
        raise AssertionError("stream() is not used here")


class _TwoTurnChatProvider(Provider):
    """Turn 1 asks for a tool; turn 2 answers with text."""

    protocol = "openai"
    brand = "mock-usage-two-turn"

    def __init__(self, first: UsageInfo | None, second: UsageInfo | None) -> None:
        super().__init__(api_key="", base_url="")
        self.first = first
        self.second = second
        self.calls: list[CLIverRequest] = []

    async def chat(self, request: CLIverRequest) -> CLIverResponse:
        self.calls.append(request)
        if len(self.calls) == 1:
            return CLIverResponse(
                message=CLIverMessage(
                    role="assistant",
                    content=None,
                    tool_calls=[ToolCall(id="call_1", name="echo", args={"value": "hi"})],
                ),
                usage=self.first,
            )
        return CLIverResponse(
            message=CLIverMessage(role="assistant", content="done"),
            usage=self.second,
        )

    async def stream(self, request):
        raise AssertionError("stream() is not used here")


class _AlwaysToolChatProvider(Provider):
    """Every turn asks for the same tool — drives stop/max-iteration paths."""

    protocol = "openai"
    brand = "mock-usage-always-tool"

    def __init__(self, tool_name: str, usage: UsageInfo | None) -> None:
        super().__init__(api_key="", base_url="")
        self.tool_name = tool_name
        self.usage = usage
        self.calls: list[CLIverRequest] = []

    async def chat(self, request: CLIverRequest) -> CLIverResponse:
        self.calls.append(request)
        return CLIverResponse(
            message=CLIverMessage(
                role="assistant",
                content=None,
                tool_calls=[
                    ToolCall(
                        id=f"call_{len(self.calls)}",
                        name=self.tool_name,
                        args={"value": "hi"},
                    )
                ],
            ),
            usage=self.usage,
        )

    async def stream(self, request):
        raise AssertionError("stream() is not used here")


def test_usage_info_total_tokens_is_input_plus_output() -> None:
    usage = UsageInfo(input_tokens=10, output_tokens=4, cache_read_input_tokens=7)
    assert usage.total_tokens == 14


@pytest.mark.asyncio
async def test_chat_keeps_final_usage_and_sums_usage_total_across_iterations() -> None:
    final = UsageInfo(input_tokens=20, output_tokens=5)
    provider = _TwoTurnChatProvider(
        UsageInfo(input_tokens=10, output_tokens=2, cache_read_input_tokens=3),
        final,
    )

    response = await _agent(provider).chat("hi")

    assert provider.calls and len(provider.calls) == 2
    # ``usage`` is unchanged: the final response's own usage.
    assert response.usage == final
    # ``usage_total`` is the sum over every Re-Act iteration.
    assert response.usage_total == UsageInfo(
        input_tokens=30,
        output_tokens=7,
        cache_read_input_tokens=3,
    )
    assert response.usage_total.total_tokens == 37


@pytest.mark.asyncio
async def test_chat_single_call_usage_total_equals_usage() -> None:
    usage = UsageInfo(input_tokens=7, output_tokens=3)

    response = await _agent(_SingleTurnChatProvider(usage)).chat("hi")

    assert response.usage == usage
    assert response.usage_total == usage


@pytest.mark.asyncio
async def test_chat_usage_total_stays_none_when_the_provider_reports_none() -> None:
    response = await _agent(_SingleTurnChatProvider(None)).chat("hi")

    assert response.usage is None
    assert response.usage_total is None


@pytest.mark.asyncio
async def test_chat_usage_total_ignores_missing_usage_from_one_iteration() -> None:
    final = UsageInfo(input_tokens=5, output_tokens=1)
    provider = _TwoTurnChatProvider(None, final)

    response = await _agent(provider).chat("hi")

    assert response.usage == final
    assert response.usage_total == final


@pytest.mark.asyncio
async def test_chat_usage_total_keeps_known_cache_reads_when_one_iteration_omits_them() -> None:
    provider = _TwoTurnChatProvider(
        UsageInfo(input_tokens=1, output_tokens=1),
        UsageInfo(input_tokens=2, output_tokens=2, cache_read_input_tokens=4),
    )

    response = await _agent(provider).chat("hi")

    assert response.usage_total == UsageInfo(
        input_tokens=3,
        output_tokens=3,
        cache_read_input_tokens=4,
    )


@pytest.mark.asyncio
async def test_chat_sums_usage_total_when_the_loop_stops_on_tool_errors() -> None:
    provider = _AlwaysToolChatProvider("boom", UsageInfo(input_tokens=10, output_tokens=2))

    response = await _agent(
        provider,
        tools=(boom,),
        max_consecutive_errors=2,
    ).chat("hi")

    assert "failed repeatedly" in response.message.content
    assert len(provider.calls) == 2
    # ``usage`` keeps its meaning: the last iteration's model call.
    assert response.usage == UsageInfo(input_tokens=10, output_tokens=2)
    assert response.usage_total == UsageInfo(input_tokens=20, output_tokens=4)


@pytest.mark.asyncio
async def test_chat_sums_usage_total_when_max_iterations_is_reached() -> None:
    provider = _AlwaysToolChatProvider("echo", UsageInfo(input_tokens=10, output_tokens=2))

    response = await _agent(provider).chat("hi", max_iterations=2)

    assert "maximum iterations" in response.message.content
    assert len(provider.calls) == 2
    assert response.usage is None  # no final model response
    assert response.usage_total == UsageInfo(input_tokens=20, output_tokens=4)
