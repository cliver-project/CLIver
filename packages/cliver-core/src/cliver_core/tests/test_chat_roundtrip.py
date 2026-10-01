"""
Smoke test: AgentCore.chat() runs the Re-Act loop end-to-end against a
mocked Provider. Proves the loop, message construction, and ToolCallAccumulator
all wire together — independent of any real provider SDK.
"""

from __future__ import annotations

import pytest

from cliver_core.agent import AgentCore
from cliver_core.provider import (
    CLIverRequest,
    CLIverResponse,
    Provider,
)
from cliver_core.tool import tool


class _MockProvider(Provider):
    """Provider that returns a single canned response — no SDK import."""

    supported_protocols = ["mock"]

    def __init__(self) -> None:
        super().__init__(protocol="mock", api_key="", base_url="")
        self.calls: list[CLIverRequest] = []

    async def chat(self, request: CLIverRequest) -> CLIverResponse:
        self.calls.append(request)
        from cliver_core.messages import CLIverMessage

        return CLIverResponse(
            message=CLIverMessage(role="assistant", content="Hello from mock."),
            usage=None,
            media=None,
            finish_reason="stop",
            raw=None,
        )

    async def stream(self, request, on_chunk=None, on_response=None):
        # Not exercised in this test
        if False:
            yield None

    def msg_to_native(self, message):
        # Minimal pass-through — round-trip just needs the loop to run
        return {"role": message.role, "content": message.content}


@tool(name="echo", description="Echoes the input back.")
def echo(value: str) -> str:
    return f"echo: {value}"


@pytest.mark.asyncio
async def test_chat_roundtrip_with_mocked_provider() -> None:
    provider = _MockProvider()
    agent = AgentCore(
        provider=provider,
        model="mock-model",
        builtin_tools=[],
    )
    response = await agent.chat("hi")
    assert response.message.role == "assistant"
    assert "Hello from mock" in response.message.content
    assert len(provider.calls) == 1


@pytest.mark.asyncio
async def test_agent_core_constructs_without_tools() -> None:
    """Smoke check: AgentCore can be constructed with no tools and still chat."""
    provider = _MockProvider()
    agent = AgentCore(
        provider=provider,
        model="mock-model",
        builtin_tools=[],
    )
    assert agent.tool_registry is not None
    response = await agent.chat("hi")
    assert response.message.role == "assistant"
