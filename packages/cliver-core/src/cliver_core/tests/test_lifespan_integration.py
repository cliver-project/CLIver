"""The generic ASGI lifespan integration — no FastAPI dependency.

``agent_lifespan(bindings)`` returns a lifespan callable in the plain ASGI
shape (``app -> async context manager``): it builds the agent, preflights the
bindings on the serving loop, publishes the agent on ``app.state`` for
``Depends(get_agent)``, and closes on every exit path.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from cliver_core.agent import AgentCore
from cliver_core.bindings import ModelBinding, ModelBindings
from cliver_core.errors import LoopAffinityError
from cliver_core.integrations.lifespan import agent_lifespan, get_agent
from cliver_core.messages import CLIverMessage
from cliver_core.provider import CLIverRequest, CLIverResponse, Provider


class _FakeProvider(Provider):
    protocol = "openai"
    brand = "fake-lifespan"

    def __init__(self, close_error: Exception | None = None) -> None:
        super().__init__(api_key="", base_url="")
        self.closed = 0
        self.close_error = close_error

    async def chat(self, request: CLIverRequest) -> CLIverResponse:
        return CLIverResponse(message=CLIverMessage(role="assistant", content="ok"))

    async def stream(self, request):  # pragma: no cover
        raise AssertionError("stream() is not used here")

    async def close(self) -> None:
        self.closed += 1
        if self.close_error is not None:
            raise self.close_error


def _built_bindings(monkeypatch) -> tuple[ModelBindings, _FakeProvider]:
    provider = _FakeProvider()
    monkeypatch.setattr("cliver_core.bindings.create_provider", lambda **kwargs: provider)
    bindings = ModelBindings(llm=ModelBinding.llm(model="mock-model", base_url="https://example.test"))
    return bindings, provider


def _app() -> SimpleNamespace:
    return SimpleNamespace(state=SimpleNamespace())


@pytest.mark.asyncio
async def test_lifespan_preflights_before_yield_and_publishes_the_agent(monkeypatch) -> None:
    bindings, provider = _built_bindings(monkeypatch)
    app = _app()

    async with agent_lifespan(bindings)(app) as yielded:
        assert yielded is None  # the ASGI lifespan yield is reserved for a state mapping
        agent = app.state.agent
        assert isinstance(agent, AgentCore)
        # The provider exists inside the context: preflight ran before yield.
        assert bindings.llm.build_provider() is provider
        assert (await agent.chat("hi")).message.content == "ok"

    assert provider.closed == 1


@pytest.mark.asyncio
async def test_lifespan_yields_none_for_the_asgi_state_protocol(monkeypatch) -> None:
    """Starlette merges a non-None lifespan yield into scope["state"]; an agent is not a mapping."""
    bindings, _ = _built_bindings(monkeypatch)
    app = _app()
    scope_state: dict[str, object] = {}

    async with agent_lifespan(bindings)(app) as maybe_state:
        if maybe_state is not None:
            # What starlette.routing.Router.lifespan does with the yielded value.
            scope_state.update(maybe_state)

    assert scope_state == {}
    assert isinstance(app.state.agent, AgentCore)


@pytest.mark.asyncio
async def test_lifespan_closes_on_exceptional_exit(monkeypatch) -> None:
    bindings, provider = _built_bindings(monkeypatch)
    app = _app()

    with pytest.raises(RuntimeError, match="boom"):
        async with agent_lifespan(bindings)(app):
            await app.state.agent.chat("hi")
            raise RuntimeError("boom")

    assert provider.closed == 1


@pytest.mark.asyncio
async def test_lifespan_propagates_a_close_failure_on_normal_exit(monkeypatch) -> None:
    provider = _FakeProvider(close_error=RuntimeError("close failed"))
    monkeypatch.setattr("cliver_core.bindings.create_provider", lambda **kwargs: provider)
    bindings = ModelBindings(llm=ModelBinding.llm(model="mock-model", base_url="https://example.test"))
    app = _app()

    with pytest.raises(RuntimeError, match="close failed"):
        async with agent_lifespan(bindings)(app):
            pass

    assert provider.closed == 1


def test_lifespan_keeps_the_preflight_error_when_close_also_fails(monkeypatch) -> None:
    """A close on the wrong loop must not replace the labeled preflight error."""
    bindings, _ = _built_bindings(monkeypatch)

    async def first_loop() -> None:
        await bindings.preflight()  # claims the first loop and builds the provider

    asyncio.run(first_loop())
    app = _app()

    async def second_loop() -> None:
        with pytest.raises(LoopAffinityError, match=r"bindings\.llm"):
            async with agent_lifespan(bindings)(app):
                pass  # pragma: no cover - preflight must fail first

    asyncio.run(second_loop())


@pytest.mark.asyncio
async def test_lifespan_fails_startup_before_publishing_a_broken_agent() -> None:
    bindings = ModelBindings(llm=ModelBinding.llm(model="bad-model", api_key="k"))
    app = _app()

    with pytest.raises(ValueError, match=r"bindings\.llm.*bad-model"):
        async with agent_lifespan(bindings)(app):
            pass  # pragma: no cover - preflight must fail first

    assert not hasattr(app.state, "agent")


@pytest.mark.asyncio
async def test_lifespan_yields_without_app_state() -> None:
    bindings = ModelBindings(
        llm=ModelBinding.llm(model="mock-model", provider=_FakeProvider()),
    )
    app = SimpleNamespace()  # no .state, e.g. a minimal ASGI test double

    async with agent_lifespan(bindings)(app) as yielded:
        assert yielded is None


def test_get_agent_reads_the_app_state() -> None:
    agent = AgentCore(ModelBindings(llm=ModelBinding.llm(model="mock-model", provider=_FakeProvider())))
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(agent=agent)))

    assert get_agent(request) is agent


def test_get_agent_explains_a_missing_agent() -> None:
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))

    with pytest.raises(RuntimeError, match="agent_lifespan"):
        get_agent(request)
