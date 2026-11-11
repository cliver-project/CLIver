"""Generic ASGI lifespan integration for :class:`AgentCore`.

No FastAPI or Starlette import: this module uses the plain ASGI lifespan
shape — a callable that takes the app and returns an async context manager —
so it works with FastAPI, Starlette, or a bare ASGI harness.

The context manager yields **nothing**: the ASGI lifespan protocol reserves a
non-None yield for a state mapping, which the server merges into
``scope["state"]``.  The agent is published on ``app.state.agent`` instead::

    from fastapi import Depends, FastAPI, Request

    from cliver_core.integrations.lifespan import agent_lifespan, get_agent

    app = FastAPI(lifespan=agent_lifespan(bindings))


    def current_agent(request: Request) -> AgentCore:
        return get_agent(request)


    @app.post("/chat")
    async def chat(agent: AgentCore = Depends(current_agent)):
        ...

The one-line wrapper is needed because FastAPI resolves a dependency by
parameter annotation; ``get_agent`` takes ``request`` as ``Any`` so this
module never imports ``Request``, and annotating the parameter is the
application's job.

The lifespan builds the agent on the serving loop, runs
:meth:`ModelBindings.preflight` (so a missing endpoint or credential fails at
startup, not on the first request), publishes the agent on ``app.state``, and
closes it on every exit path.  When an exception is already propagating, a
close failure is suppressed so it cannot replace the original error; on a
normal exit a close failure propagates.  Provider ownership is unchanged: the
agent closes providers its bindings built, never a caller-supplied provider or
the MCP client.

Endpoints using a loop-bound agent must be ``async def``.  A sync ``def``
endpoint runs in a worker thread on a different event loop, and a binding
already claimed by the serving loop raises
:class:`~cliver_core.errors.LoopAffinityError`.
"""

from __future__ import annotations

from contextlib import AbstractAsyncContextManager, asynccontextmanager, suppress
from typing import Any, AsyncIterator, Callable

from cliver_core.agent import AgentCore
from cliver_core.bindings import ModelBindings

__all__ = ["agent_lifespan", "get_agent"]


def agent_lifespan(
    bindings: ModelBindings,
    **agent_kwargs: Any,
) -> Callable[[Any], AbstractAsyncContextManager[None]]:
    """Return an ASGI lifespan callable that owns one :class:`AgentCore`.

    Pass it straight to the framework — ``FastAPI(lifespan=agent_lifespan(bindings))``
    or ``Starlette(lifespan=...)`` — or compose it inside your own lifespan.
    *agent_kwargs* are forwarded to :meth:`AgentCore.open`.

    The returned context manager yields ``None`` (the ASGI lifespan yield is
    reserved for a state mapping), sets ``app.state.agent`` when the app
    exposes a ``state`` attribute (Starlette/FastAPI do), and awaits
    ``aclose()`` — on normal exit, on an exception from the served
    application, and when preflight itself fails.  If an exception is already
    propagating, a close failure is suppressed rather than replacing it.  The
    MCP client, if one is passed, stays caller-owned: this helper does not
    close it.
    """

    @asynccontextmanager
    async def lifespan(app: Any) -> AsyncIterator[None]:
        agent = AgentCore.open(bindings, **agent_kwargs)
        try:
            await bindings.preflight()
            if hasattr(app, "state"):
                app.state.agent = agent
            yield
        except BaseException:
            with suppress(BaseException):
                await agent.aclose()
            raise
        else:
            await agent.aclose()

    return lifespan


def get_agent(request: Any) -> AgentCore:
    """The agent published by :func:`agent_lifespan` on ``request.app.state``.

    Call it from an application-side dependency wrapper — FastAPI resolves
    ``Depends`` by parameter annotation, and this module cannot annotate
    ``Request`` without importing Starlette::

        from fastapi import Request

        def current_agent(request: Request) -> AgentCore:
            return get_agent(request)

    ``get_agent`` itself takes ``request`` as ``Any`` and only reads
    ``request.app.state``.
    """
    try:
        return request.app.state.agent
    except AttributeError:
        raise RuntimeError(
            "no agent on request.app.state — install agent_lifespan(bindings) as the app's lifespan"
        ) from None
