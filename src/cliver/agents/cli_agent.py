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


class CliAgent(Agent):
    """Base class for all CLI subprocess-based agents.

    Subclasses must implement ``_build_command`` and ``_build_env``.
    """

    DEFAULT_COMMAND: str = ""
    DEFAULT_ARGS: List[str] = []

    # Protocol-critical flags that must never be overridden by user config.
    # Key = flag, Value = whether the flag takes a following value argument.
    BLOCKED_ARGS: Dict[str, bool] = {}
    # Env var names for API key, base URL, and model. Override in subclasses.
    ENV_MAPPING: Dict[str, str] = {}

    def _build_command(self, prompt: str) -> List[str]:
        """Build the full command line. Must be implemented by subclasses."""
        raise NotImplementedError(f"{type(self).__name__} must implement _build_command")

    def _build_env(self) -> dict:
        """Build the environment for the subprocess. Must be implemented by subclasses."""
        raise NotImplementedError(f"{type(self).__name__} must implement _build_env")

    def _base_env(self) -> dict:
        """Common env vars: API key, base URL, and user config.env overrides.

        Each implementation's ``_build_env`` is responsible for setting
        provider-specific model env vars (e.g. ``ANTHROPIC_MODEL``,
        ``CLAUDE_CODE_SUBAGENT_MODEL``) from ``self._model_config``.
        Extra env vars that don't warrant a config field can be passed
        through ``config.env`` in the YAML.
        """
        env = dict(os.environ)
        mapping = self.ENV_MAPPING
        if self._provider_config:
            api_key = self._provider_config.get_api_key()
            if "api_key" in mapping and api_key:
                env[mapping["api_key"]] = api_key
            if "api_url" in mapping and self._provider_config.api_url:
                env[mapping["api_url"]] = self._provider_config.api_url
        if self.config.env:
            env.update(self.config.env)
        return env

    @staticmethod
    def _filter_args(args: List[str], blocked: Dict[str, bool]) -> List[str]:
        """Remove protocol-critical flags from user-provided args."""
        if not args:
            return args
        filtered: List[str] = []
        skip_next = False
        for arg in args:
            if skip_next:
                skip_next = False
                continue
            flag = arg.split("=", 1)[0] if "=" in arg else arg
            if flag in blocked:
                if blocked[flag] and "=" not in arg:
                    skip_next = True
                continue
            filtered.append(arg)
        return filtered

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
            name=name,
            config=config,
            rate_limiter=rate_limiter,
            rate_limit_key=rate_limit_key,
        )
        self._command = config.command or self.DEFAULT_COMMAND
        self._resolved_command: str = self._command
        user_args = config.args if config.args is not None else list(self.DEFAULT_ARGS)
        self._user_args = self._filter_args(user_args, self.BLOCKED_ARGS)
        self._model_config = model_config
        self._provider_config = provider_config
        self._output_dir: Optional[Path] = None
        self._pre_snapshot: Optional[Set[str]] = None

    async def initialize(self, context: dict = None) -> None:
        resolved = shutil.which(self._command)
        if not resolved:
            raise RuntimeError(f"CLI agent '{self._command}' not found. Install it or check your PATH.")
        self._resolved_command = resolved
        self._output_dir = Path(tempfile.mkdtemp(prefix=f"cliver-{self.name}-"))
        if context and context.get("working_dir"):
            self.config.working_dir = context["working_dir"]
        else:
            from cliver.util import get_config_dir

            self.config.working_dir = str(Path(get_config_dir()) / "agents" / self.name / "runs")
        Path(self.config.working_dir).mkdir(parents=True, exist_ok=True)

    async def _do_run(self, prompt: str, **kwargs) -> AgentResult:
        """Run via streaming JSONL — parse all events and accumulate the result."""
        start = time.monotonic()
        working = Path(self.config.working_dir or ".")
        self._pre_snapshot = self._snapshot_dir(working)

        cmd = self._build_command(prompt)
        env = self._build_env()

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=self.config.working_dir or None,
            )
        except FileNotFoundError:
            return AgentResult(
                text="",
                status="error",
                error=f"Command not found: {self._command}",
                duration_ms=int((time.monotonic() - start) * 1000),
            )

        full_text = []
        json_buffer = ""
        stderr_data = []
        stderr_task = asyncio.ensure_future(self._read_stderr(proc, stderr_data))

        async for line in proc.stdout:
            text = line.decode(errors="replace").rstrip("\n")
            try:
                chunk_json = json.loads(text)
                chunk_text = self._parse_stream_chunk(chunk_json)
                if chunk_text:
                    full_text.append(chunk_text)
            except json.JSONDecodeError:
                json_buffer += text
                try:
                    chunk_json = json.loads(json_buffer)
                    json_buffer = ""
                    chunk_text = self._parse_stream_chunk(chunk_json)
                    if chunk_text:
                        full_text.append(chunk_text)
                except json.JSONDecodeError:
                    pass

        if json_buffer:
            try:
                chunk_json = json.loads(json_buffer)
                chunk_text = self._parse_stream_chunk(chunk_json)
                if chunk_text:
                    full_text.append(chunk_text)
            except json.JSONDecodeError:
                full_text.append(json_buffer)

        await stderr_task
        await proc.wait()
        duration = int((time.monotonic() - start) * 1000)

        if proc.returncode != 0:
            return AgentResult(
                text="".join(full_text),
                status="error",
                error=f"{self._command} exited with code {proc.returncode}",
                duration_ms=duration,
            )

        return AgentResult(
            text="".join(full_text),
            status="completed",
            artifacts=self._diff_artifacts(working),
            duration_ms=duration,
        )

    async def _read_stderr(self, proc, stderr_data: list) -> None:
        async for line in proc.stderr:
            stderr_data.append(line.decode(errors="replace"))

    def _parse_response(self, raw_json: dict) -> AgentResult:
        text = raw_json.get("result", raw_json.get("response", str(raw_json)))
        return AgentResult(text=str(text), status="completed")

    def _extract_artifacts(self, raw_json: dict) -> List[Artifact]:
        return []

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
            artifacts.append(
                Artifact(
                    path=f,
                    media_type=mime,
                    size=p.stat().st_size if p.exists() else None,
                )
            )
        return artifacts

    async def stream(self, prompt: str, **kwargs) -> AsyncIterator[AgentChunk]:
        cmd = self._build_command(prompt)
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
                    text="",
                    status="error",
                    error=f"Command not found: {self._command}",
                ),
            )
            return

        full_text = []
        json_buffer = ""
        async for line in proc.stdout:
            text = line.decode(errors="replace").rstrip("\n")
            try:
                chunk_json = json.loads(text)
                chunk_text = self._parse_stream_chunk(chunk_json)
                full_text.append(chunk_text)
                yield AgentChunk(text=chunk_text, chunk_type="text")
            except json.JSONDecodeError:
                json_buffer += text
                try:
                    chunk_json = json.loads(json_buffer)
                    json_buffer = ""
                    chunk_text = self._parse_stream_chunk(chunk_json)
                    full_text.append(chunk_text)
                    yield AgentChunk(text=chunk_text, chunk_type="text")
                except json.JSONDecodeError:
                    pass

        if json_buffer:
            try:
                chunk_json = json.loads(json_buffer)
                chunk_text = self._parse_stream_chunk(chunk_json)
                full_text.append(chunk_text)
                yield AgentChunk(text=chunk_text, chunk_type="text")
            except json.JSONDecodeError:
                full_text.append(json_buffer)
                yield AgentChunk(text=json_buffer, chunk_type="text")

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

    def _parse_stream_chunk(self, chunk_json: dict) -> str:
        return chunk_json.get("text", chunk_json.get("content", str(chunk_json)))

    async def cleanup(self) -> None:
        if self._output_dir and self._output_dir.exists():
            shutil.rmtree(self._output_dir, ignore_errors=True)
