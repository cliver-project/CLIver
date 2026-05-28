"""Agent — a configured AgentCore with retry/timeout logic.

An Agent wraps an AgentCore instance with a persona (system prompt)
and operational concerns (retries, timeouts) from AgentConfig.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, AsyncIterator

from cliver.media import MediaContent
from cliver.messages import CLIverMessageChunk
from cliver.provider import CLIverResponse

if TYPE_CHECKING:
    from cliver.config import AgentConfig
    from cliver.llm.new_agent import AgentCore

logger = logging.getLogger(__name__)


class Agent:
    """A configured AgentCore with retry/timeout logic.

    One Agent per persona — wraps an AgentCore and adds:
    - System prompt from config (role, system_prompt, skills)
    - Retry on failure (config.max_retries)
    - Timeout (config.timeout_s)

    Return types are the same as AgentCore — no custom result wrappers.
    """

    def __init__(
        self,
        name: str,
        config: "AgentConfig",
        agent_core: "AgentCore",
    ):
        self.name = name
        self.config = config
        self._core = agent_core

    # ── Public API ────────────────────────────────────────────

    async def chat(
        self,
        prompt: str,
        *,
        media: list[MediaContent] | None = None,
        **kwargs,
    ) -> CLIverResponse:
        """Run with retry/timeout.  Returns CLIverResponse (same as AgentCore)."""
        system_prompt = self._build_system_prompt()
        timeout = self.config.timeout_s or 300
        last_error = None

        for attempt in range(1 + self.config.max_retries):
            try:
                return await asyncio.wait_for(
                    self._core.chat(
                        user_input=prompt,
                        system_prompt=system_prompt,
                        media=media,
                        **kwargs,
                    ),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                last_error = f"Timeout after {timeout}s (attempt {attempt + 1})"
                logger.warning("%s: %s", self.name, last_error)
            except Exception as e:
                last_error = str(e)
                if attempt < self.config.max_retries:
                    logger.warning("%s: retry %d/%d: %s", self.name, attempt + 1, self.config.max_retries, e)
                    continue
                break

        return CLIverResponse(
            message=__import__("cliver.messages").messages.CLIverMessage(
                role="assistant",
                content=f"Error: {last_error}",
            ),
        )

    async def stream(
        self,
        prompt: str,
        *,
        media: list[MediaContent] | None = None,
        **kwargs,
    ) -> AsyncIterator[CLIverMessageChunk]:
        """Streaming — no retry for streams.  Yields CLIverMessageChunk."""
        system_prompt = self._build_system_prompt()
        async for chunk in self._core.stream(
            user_input=prompt,
            system_prompt=system_prompt,
            media=media,
            **kwargs,
        ):
            yield chunk

    async def generate(
        self,
        prompt: str,
        *,
        media_type: str = "image",
        media: list[MediaContent] | None = None,
        output_dir: str | None = None,
        **options,
    ) -> CLIverResponse:
        """Generate media.  Returns CLIverResponse with media field."""
        return await self._core.generate(
            prompt=prompt,
            media_type=media_type,
            media=media,
            output_dir=output_dir,
            **options,
        )

    # ── Helpers ────────────────────────────────────────────────

    def _build_system_prompt(self) -> str | None:
        role = self.config.system_prompt
        if not role:
            return None
        return f"Your role: {role}"

    async def cleanup(self) -> None:
        pass
