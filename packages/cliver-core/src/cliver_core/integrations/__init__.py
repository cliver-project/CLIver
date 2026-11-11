"""Framework integrations built on the core API only.

Nothing here imports FastAPI, Starlette, or another web framework: the
lifespan helper uses the generic ASGI lifespan shape (a callable taking the
app and returning an async context manager) and touches only ``app.state``.
Framework-specific adapters, if ever needed, belong behind an optional extra
or a separate package — never in the core install.
"""

from cliver_core.integrations.lifespan import agent_lifespan, get_agent

__all__ = ["agent_lifespan", "get_agent"]
