"""CliverAgent — wraps AgentCore for CLIver's built-in LLM execution."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, AsyncIterator, List, Optional

from cliver.agent import Agent, AgentChunk, AgentResult, Artifact

if TYPE_CHECKING:
    from cliver.config import AgentConfig, ModelConfig, ProviderConfig
    from cliver.llm.llm import AgentCore
    from cliver.llm.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)


class CliverAgent(Agent):
    """Agent backed by CLIver's AgentCore (internal LLM execution)."""

    def __init__(
        self,
        name: str,
        config: "AgentConfig",
        agent_core: "AgentCore",
        model_config: Optional["ModelConfig"] = None,
        provider_config: Optional["ProviderConfig"] = None,
        rate_limiter: Optional["RateLimiter"] = None,
        **kwargs,
    ):
        rate_limit_key = None
        if provider_config and rate_limiter:
            rate_limit_key = f"{provider_config.api_url}|{provider_config.api_key or ''}"
        super().__init__(
            name=name,
            config=config,
            rate_limiter=rate_limiter,
            rate_limit_key=rate_limit_key,
        )
        self._agent_core = agent_core
        self._model_config = model_config
        self._provider_config = provider_config

    async def _do_run(self, prompt: str, *, images=None, files=None, **kwargs) -> AgentResult:
        start = time.monotonic()
        try:
            message = await asyncio.to_thread(
                self._agent_core.process_user_input,
                user_input=prompt,
                images=images or [],
                files=files or [],
                model=self.config.model,
                system_message_appender=self._build_system_appender(),
                auto_fallback=self.config.auto_fallback,
                **kwargs,
            )
            duration = int((time.monotonic() - start) * 1000)
            text = message.content if isinstance(message.content, str) else str(message.content)
            artifacts = self._extract_artifacts_from_message(message)

            return AgentResult(
                text=text,
                status="completed",
                artifacts=artifacts,
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
        if self._rate_limiter and self._rate_limit_key:
            await self._rate_limiter.wait(self._rate_limit_key)

        start = time.monotonic()
        full_text = []
        try:
            async for chunk in self._agent_core._stream_user_input_async(
                user_input=prompt,
                model=self.config.model,
                system_message_appender=self._build_system_appender(),
                auto_fallback=self.config.auto_fallback,
                images=images or [],
                files=files or [],
                **kwargs,
            ):
                text = chunk.content if hasattr(chunk, "content") else str(chunk)
                if text:
                    full_text.append(text)
                    yield AgentChunk(text=text, chunk_type="text")
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

    def _build_system_appender(self):
        if not self.config.system_prompt:
            return None
        role = self.config.system_prompt
        return lambda: f"\n\nYour role: {role}"

    def _extract_artifacts_from_message(self, message) -> List[Artifact]:
        artifacts = []
        if hasattr(message, "additional_kwargs"):
            for tc in message.additional_kwargs.get("tool_calls", []):
                func = tc.get("function", {})
                if func.get("name") in ("write_file", "save_file"):
                    import json as _json

                    try:
                        args = _json.loads(func.get("arguments", "{}"))
                        path = args.get("path") or args.get("file_path")
                        if path:
                            import mimetypes

                            mime = mimetypes.guess_type(path)[0] or "text/plain"
                            artifacts.append(Artifact(path=path, media_type=mime))
                    except _json.JSONDecodeError:
                        pass
        return artifacts
