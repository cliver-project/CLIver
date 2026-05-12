"""CliAgent — base class for CLI subprocess-based agents.

This module uses asyncio.create_subprocess_exec (not shell exec) for safe subprocess execution.
All commands are passed as lists to prevent shell injection vulnerabilities.
"""

from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, AsyncIterator, Dict, List, Optional, Set

from cliver.agent import Agent, AgentChunk, AgentResult, Artifact

if TYPE_CHECKING:
    from cliver.config import AgentConfig, ModelConfig, ProviderConfig
    from cliver.llm.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

ENV_VAR_MAPPING: Dict[str, Dict[str, str]] = {
    "anthropic": {
        "api_key": "ANTHROPIC_API_KEY",
        "api_url": "ANTHROPIC_API_URL",
        "model": "ANTHROPIC_MODEL",
    },
    "openai": {
        "api_key": "OPENAI_API_KEY",
        "api_url": "OPENAI_BASE_URL",
        "model": "OPENAI_MODEL",
    },
    "google": {
        "api_key": "GEMINI_API_KEY",
        "model": "GEMINI_MODEL",
    },
    "deepseek": {
        "api_key": "DEEPSEEK_API_KEY",
        "api_url": "DEEPSEEK_API_URL",
        "model": "DEEPSEEK_MODEL",
    },
}


class CliAgent(Agent):
    """Base class for all CLI subprocess-based agents."""

    DEFAULT_COMMAND: str = ""
    DEFAULT_ARGS: List[str] = []
    DEFAULT_OUTPUT_FORMAT: List[str] = ["--output-format", "json"]

    def __init__(
        self,
        name: str,
        config: "AgentConfig",
        model_config: Optional["ModelConfig"] = None,
        provider_config: Optional["ProviderConfig"] = None,
        rate_limiter: Optional["RateLimiter"] = None,
        **kwargs,
    ):
        rate_limit_key = None
        if provider_config and rate_limiter:
            rate_limit_key = f"{provider_config.api_url}|{provider_config.api_key or ''}"
        super().__init__(
            name=name, config=config,
            rate_limiter=rate_limiter, rate_limit_key=rate_limit_key,
        )
        self._command = config.command or self.DEFAULT_COMMAND
        self._args = config.args if config.args is not None else list(self.DEFAULT_ARGS)
        self._output_format = list(self.DEFAULT_OUTPUT_FORMAT)
        self._model_config = model_config
        self._provider_config = provider_config
        self._output_dir: Optional[Path] = None
        self._pre_snapshot: Optional[Set[str]] = None

    async def initialize(self, context: dict = None) -> None:
        if not shutil.which(self._command):
            raise RuntimeError(
                f"CLI agent '{self._command}' not found. "
                f"Install it or check your PATH."
            )
        self._output_dir = Path(tempfile.mkdtemp(prefix=f"cliver-{self.name}-"))
        if context:
            if context.get("working_dir"):
                self.config.working_dir = context["working_dir"]

    async def _do_run(self, prompt: str, **kwargs) -> AgentResult:
        start = time.monotonic()
        working = Path(self.config.working_dir or ".")
        self._pre_snapshot = self._snapshot_dir(working)

        cmd = [self._command] + self._args + self._output_format + [prompt]
        env = self._build_env()

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=self.config.working_dir or None,
            )
            stdout, stderr = await proc.communicate()
        except FileNotFoundError:
            duration = int((time.monotonic() - start) * 1000)
            return AgentResult(
                text="", status="error",
                error=f"Command not found: {self._command}",
                duration_ms=duration,
            )

        duration = int((time.monotonic() - start) * 1000)

        if proc.returncode != 0:
            return AgentResult(
                text=stderr.decode(errors="replace"),
                status="error",
                error=f"{self._command} exited with code {proc.returncode}",
                duration_ms=duration,
            )

        try:
            raw_json = json.loads(stdout.decode())
        except json.JSONDecodeError:
            return AgentResult(
                text=stdout.decode(errors="replace"),
                status="completed",
                artifacts=self._diff_artifacts(working),
                duration_ms=duration,
            )

        result = self._parse_response(raw_json)
        result.duration_ms = duration
        result.raw = raw_json

        json_artifacts = self._extract_artifacts(raw_json)
        if json_artifacts:
            result.artifacts = json_artifacts
        else:
            result.artifacts = self._diff_artifacts(working)

        return result

    def _parse_response(self, raw_json: dict) -> AgentResult:
        text = raw_json.get("result", raw_json.get("response", str(raw_json)))
        return AgentResult(text=str(text), status="completed")

    def _extract_artifacts(self, raw_json: dict) -> List[Artifact]:
        return []

    def _build_env(self) -> dict:
        env = dict(os.environ)
        if self._provider_config:
            provider_type = self._provider_config.type
            mapping = ENV_VAR_MAPPING.get(provider_type, {})
            if "api_key" in mapping and self._provider_config.api_key:
                env[mapping["api_key"]] = self._provider_config.api_key
            if "api_url" in mapping and self._provider_config.api_url:
                env[mapping["api_url"]] = self._provider_config.api_url

        if self._model_config:
            provider_type = self._provider_config.type if self._provider_config else ""
            mapping = ENV_VAR_MAPPING.get(provider_type, {})
            if "model" in mapping:
                env[mapping["model"]] = self._model_config.api_model_name

        if self.config.env:
            env.update(self.config.env)
        return env

    def _snapshot_dir(self, path: Path) -> Set[str]:
        if not path.exists():
            return set()
        return {str(p) for p in path.rglob("*") if p.is_file()}

    def _diff_artifacts(self, working: Path) -> List[Artifact]:
        if self._pre_snapshot is None:
            return []
        current = self._snapshot_dir(working)
        new_files = current - self._pre_snapshot
        artifacts = []
        for f in sorted(new_files):
            p = Path(f)
            mime = mimetypes.guess_type(f)[0] or "application/octet-stream"
            artifacts.append(Artifact(
                path=f, media_type=mime,
                size=p.stat().st_size if p.exists() else None,
            ))
        return artifacts

    async def stream(self, prompt: str, **kwargs) -> AsyncIterator[AgentChunk]:
        cmd = [self._command] + self._args
        stream_format = self._get_stream_format()
        if stream_format:
            cmd += stream_format
        else:
            cmd += self._output_format
        cmd.append(prompt)
        env = self._build_env()
        start = time.monotonic()

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=self.config.working_dir or None,
            )
        except FileNotFoundError:
            yield AgentChunk(
                chunk_type="done",
                final_result=AgentResult(
                    text="", status="error",
                    error=f"Command not found: {self._command}",
                ),
            )
            return

        full_text = []
        async for line in proc.stdout:
            text = line.decode(errors="replace").rstrip("\n")
            if stream_format:
                try:
                    chunk_json = json.loads(text)
                    chunk_text = self._parse_stream_chunk(chunk_json)
                    full_text.append(chunk_text)
                    yield AgentChunk(text=chunk_text, chunk_type="text")
                except json.JSONDecodeError:
                    full_text.append(text)
                    yield AgentChunk(text=text, chunk_type="text")
            else:
                full_text.append(text)
                yield AgentChunk(text=text + "\n", chunk_type="text")

        await proc.wait()
        duration = int((time.monotonic() - start) * 1000)
        yield AgentChunk(
            chunk_type="done",
            final_result=AgentResult(
                text="".join(full_text),
                status="completed" if proc.returncode == 0 else "error",
                duration_ms=duration,
            ),
        )

    def _get_stream_format(self) -> Optional[List[str]]:
        return None

    def _parse_stream_chunk(self, chunk_json: dict) -> str:
        return chunk_json.get("text", chunk_json.get("content", str(chunk_json)))

    async def cleanup(self) -> None:
        if self._output_dir and self._output_dir.exists():
            shutil.rmtree(self._output_dir, ignore_errors=True)
