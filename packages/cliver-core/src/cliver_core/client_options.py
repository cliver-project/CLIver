"""One data object for every HTTP-client setting a binding's provider owns.

A :class:`~cliver_core.bindings.ModelBinding` carries a :class:`ClientOptions`;
the provider it builds derives both of its HTTP stacks from that one object:

* the official SDK client (``httpx2``, used for chat/stream and transport
  posts) — User-Agent, request timeout, transport retries, and connection-pool
  limits;
* the shared raw-HTTP client (``httpx``, used for embed / rerank / System One
  and the media polling seams) — User-Agent and the same pool limits.

``None`` on a field means "keep the underlying client's default": 600 s read
timeout and 2 retries on both builtin SDKs, and httpx's 100 connections / 20
keep-alive connections / 5 s keep-alive expiry for pools.

The legacy flat binding arguments ``client_timeout`` / ``client_max_retries``
merge into the same object; setting a value in both places is allowed only
when the values agree, so a configuration can never be silently overridden.::

    from cliver_core import ClientOptions

    binding = ModelBinding.llm(
        model="deepseek-chat",
        api_key=key,
        client=ClientOptions(
            user_agent="my-app/1.2",
            timeout=30,
            max_retries=0,
            max_connections=32,
            max_keepalive_connections=8,
            keepalive_expiry=30.0,
        ),
    )
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

# The pool defaults of the ``httpx`` client the raw seams use.  Mirroring them
# lets validation reject a keep-alive pool larger than the total pool before
# the client is built (httpx itself would raise later, on first use).
DEFAULT_MAX_CONNECTIONS = 100
DEFAULT_MAX_KEEPALIVE_CONNECTIONS = 20
DEFAULT_KEEPALIVE_EXPIRY = 5.0


def validate_timeout(value: Any, name: str = "timeout") -> float | None:
    """Normalize a positive, finite number of seconds; ``None`` passes through."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive number of seconds or None")
    return float(value)


def validate_max_retries(value: Any, name: str = "max_retries") -> int | None:
    """Normalize a non-negative integer retry count; ``None`` passes through."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer or None")
    return value


def _validate_positive_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer or None")
    return value


def _validate_non_negative_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer or None")
    return value


def _validate_user_agent(value: Any, name: str = "user_agent") -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string or None")
    return value.strip()


@dataclass(frozen=True, kw_only=True)
class ClientOptions:
    """HTTP-client settings shared by the SDK client and the raw-HTTP client.

    Frozen so one object can be shared by a binding spec and its
    ``for_current_loop()`` clones.  Fields are validated and normalized on
    construction; an invalid combination fails here, before any client is
    built.
    """

    #: ``User-Agent`` header for every request the provider makes.  ``None``
    #: keeps the SDK's own header (``OpenAI/Python x.y.z``) and leaves the raw
    #: client on httpx's default.
    user_agent: str | None = None
    #: SDK-client request timeout in seconds — every HTTP phase.  ``None``
    #: keeps the SDK default (600 s read on both builtin SDKs).  This is a
    #: client-construction setting, not a per-request ``timeout`` in ``params``.
    timeout: float | None = None
    #: SDK transport retry count.  ``None`` keeps the SDK default (2 on both
    #: builtin SDKs); ``0`` lets binding failover react on the first error.
    max_retries: int | None = None
    #: Total concurrent connections one provider may open.  ``None`` keeps
    #: httpx's default (100).  Applies to both the SDK client's pool and the
    #: shared raw-HTTP client's pool.
    max_connections: int | None = None
    #: Idle connections kept alive for reuse.  ``None`` keeps httpx's default
    #: (20); ``0`` disables keep-alive.  Never larger than ``max_connections``
    #: (httpx's default 100 when that is unset).
    max_keepalive_connections: int | None = None
    #: Seconds an idle keep-alive connection stays in the pool.  ``None``
    #: keeps httpx's default (5 s).
    keepalive_expiry: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "user_agent", _validate_user_agent(self.user_agent))
        object.__setattr__(self, "timeout", validate_timeout(self.timeout))
        object.__setattr__(self, "max_retries", validate_max_retries(self.max_retries))
        object.__setattr__(self, "max_connections", _validate_positive_int(self.max_connections, "max_connections"))
        object.__setattr__(
            self,
            "max_keepalive_connections",
            _validate_non_negative_int(self.max_keepalive_connections, "max_keepalive_connections"),
        )
        object.__setattr__(self, "keepalive_expiry", validate_timeout(self.keepalive_expiry, "keepalive_expiry"))
        if self.max_keepalive_connections is not None:
            limit = self.max_connections if self.max_connections is not None else DEFAULT_MAX_CONNECTIONS
            if self.max_keepalive_connections > limit:
                raise ValueError(
                    "max_keepalive_connections cannot exceed max_connections "
                    f"({self.max_keepalive_connections} > {limit})"
                )


def resolve_client_options(
    client: ClientOptions | None,
    *,
    user_agent: Any = None,
    timeout: Any = None,
    max_retries: Any = None,
) -> ClientOptions:
    """Merge legacy flat arguments into one :class:`ClientOptions`.

    The flat ``client_timeout`` / ``client_max_retries`` constructor arguments
    and a ``client=ClientOptions(...)`` object are two spellings for the same
    settings.  A value present in both places must agree — otherwise
    :class:`ValueError` names the conflicting field instead of picking one
    silently.  ``None`` flat arguments keep the object's value.
    """
    if client is not None and not isinstance(client, ClientOptions):
        raise TypeError(f"client must be a ClientOptions instance or None (got {type(client).__name__})")
    options = client or ClientOptions()

    updates: dict[str, Any] = {}
    flat_user_agent = _validate_user_agent(user_agent, "user_agent")
    if flat_user_agent is not None:
        if options.user_agent is not None and options.user_agent != flat_user_agent:
            raise ValueError("user_agent=... conflicts with client=ClientOptions(user_agent=...); set one or the other")
        updates["user_agent"] = flat_user_agent

    flat_timeout = validate_timeout(timeout, "client_timeout")
    if flat_timeout is not None:
        if options.timeout is not None and options.timeout != flat_timeout:
            raise ValueError(
                "client_timeout=... conflicts with client=ClientOptions(timeout=...); set one or the other"
            )
        updates["timeout"] = flat_timeout

    flat_retries = validate_max_retries(max_retries, "client_max_retries")
    if flat_retries is not None:
        if options.max_retries is not None and options.max_retries != flat_retries:
            raise ValueError(
                "client_max_retries=... conflicts with client=ClientOptions(max_retries=...); set one or the other"
            )
        updates["max_retries"] = flat_retries

    return replace(options, **updates) if updates else options
