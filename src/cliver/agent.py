"""Agent abstraction layer for CLIver.

Defines the Agent ABC and unified result types used by all agent backends.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, AsyncIterator, List, Optional

if TYPE_CHECKING:
    from cliver.config import AgentConfig
    from cliver.llm.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)


@dataclass
class Artifact:
    """A file produced by an agent execution."""

    path: str
    media_type: str
    size: Optional[int] = None
    description: str = ""


@dataclass
class AgentResult:
    """Unified result returned by all agent types."""

    text: str
    status: str  # "completed" | "error" | "timeout"
    artifacts: List[Artifact] = field(default_factory=list)
    duration_ms: int = 0
    model: Optional[str] = None
    token_usage: Optional[dict] = None
    error: Optional[str] = None
    raw: Optional[dict] = None


@dataclass
class AgentChunk:
    """Incremental chunk for streaming agent output."""

    text: str = ""
    chunk_type: str = "text"  # "text" | "status" | "artifact" | "done"
    artifact: Optional[Artifact] = None
    final_result: Optional[AgentResult] = None


class Agent(ABC):
    """Abstract base for all agent types."""

    def __init__(
        self,
        name: str,
        config: "AgentConfig",
        rate_limiter: Optional["RateLimiter"] = None,
        rate_limit_key: Optional[str] = None,
        **kwargs,
    ):
        self.name = name
        self.config = config
        self._rate_limiter = rate_limiter
        self._rate_limit_key = rate_limit_key

    async def initialize(self, context: dict = None) -> None:  # noqa: B027
        pass

    async def run(
        self,
        prompt: str,
        *,
        images: List[str] = None,
        files: List[str] = None,
        timeout_s: int = None,
        **kwargs,
    ) -> AgentResult:
        if self._rate_limiter and self._rate_limit_key:
            await self._rate_limiter.wait(self._rate_limit_key)

        effective_timeout = timeout_s or self.config.timeout_s
        last_error = None

        for attempt in range(1 + self.config.max_retries):
            try:
                return await asyncio.wait_for(
                    self._do_run(prompt, images=images, files=files, **kwargs),
                    timeout=effective_timeout,
                )
            except asyncio.TimeoutError:
                last_error = f"Timeout after {effective_timeout}s (attempt {attempt + 1})"
            except Exception as e:
                last_error = str(e)
                if attempt < self.config.max_retries:
                    continue
                break

        is_timeout = last_error and "Timeout" in last_error
        return AgentResult(
            text="",
            status="timeout" if is_timeout else "error",
            error=last_error,
            duration_ms=int(effective_timeout * 1000) if is_timeout else 0,
        )

    @abstractmethod
    async def _do_run(self, prompt: str, **kwargs) -> AgentResult: ...

    async def stream(
        self,
        prompt: str,
        *,
        images: List[str] = None,
        files: List[str] = None,
        timeout_s: int = None,
        **kwargs,
    ) -> AsyncIterator[AgentChunk]:
        result = await self.run(prompt, images=images, files=files, timeout_s=timeout_s, **kwargs)
        yield AgentChunk(text=result.text, chunk_type="done", final_result=result)

    async def cleanup(self) -> None:  # noqa: B027
        pass
