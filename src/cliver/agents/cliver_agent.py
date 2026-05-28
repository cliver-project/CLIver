"""CliverAgent — wraps new AgentCore for CLIver's built-in LLM execution."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, AsyncIterator

from cliver.agent import Agent, AgentChunk, AgentResult

if TYPE_CHECKING:
    from cliver.config import AgentConfig
    from cliver.llm.new_agent import AgentCore as NewAgentCore

logger = logging.getLogger(__name__)


class CliverAgent(Agent):
    """Agent backed by CLIver's new AgentCore (langchain-free)."""

    def __init__(
        self,
        name: str,
        config: "AgentConfig",
        agent_core: "NewAgentCore",
        **kwargs,
    ):
        super().__init__(
            name=name,
            config=config,
        )
        self._agent_core = agent_core

    async def _do_run(self, prompt: str, *, images=None, files=None, **kwargs) -> AgentResult:
        start = time.monotonic()
        try:
            response = await self._agent_core.chat(
                user_input=prompt,
                system_prompt=self._build_system_prompt(),
            )
            duration = int((time.monotonic() - start) * 1000)
            text = response.message.text or ""

            return AgentResult(
                text=text,
                status="completed",
                duration_ms=duration,
                model=self.config.model,
            )
        except Exception as e:
            duration = int((time.monotonic() - start) * 1000)
            logger.warning("CliverAgent run failed: %s", e)
            return AgentResult(
                text="",
                status="error",
                error=str(e),
                duration_ms=duration,
            )

    async def stream(
        self, prompt: str, *, images=None, files=None, timeout_s=None, **kwargs
    ) -> AsyncIterator[AgentChunk]:
        start = time.monotonic()
        full_text: list[str] = []
        try:
            async for chunk in self._agent_core.stream(
                user_input=prompt,
                system_prompt=self._build_system_prompt(),
            ):
                if chunk.content:
                    full_text.append(chunk.content)
                    yield AgentChunk(text=chunk.content, chunk_type="text")
        except Exception as e:
            duration = int((time.monotonic() - start) * 1000)
            yield AgentChunk(
                chunk_type="done",
                final_result=AgentResult(
                    text="".join(full_text),
                    status="error",
                    error=str(e),
                    duration_ms=duration,
                ),
            )
            return

        duration = int((time.monotonic() - start) * 1000)
        yield AgentChunk(
            chunk_type="done",
            final_result=AgentResult(
                text="".join(full_text),
                status="completed",
                duration_ms=duration,
                model=self.config.model,
            ),
        )

    def _build_system_prompt(self) -> str | None:
        role = self.config.system_prompt
        if not role:
            return None
        return f"Your role: {role}"
