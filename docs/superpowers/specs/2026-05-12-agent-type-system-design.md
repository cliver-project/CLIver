# Agent Type System Design (A1)

**Date:** 2026-05-12
**Status:** Approved
**Activity:** A1 — Agent Type System Refactoring

## Goal

Introduce a unified Agent abstraction layer that supports multiple AI agent backends (CLIver built-in, Claude Code, Gemini CLI, OpenCode, and user-defined CLI agents) through a type-based registry with consistent lifecycle, result format, rate limiting, and credential management.

## Architecture

```
AgentFactory (creates + caches agents, owns shared RateLimiter)
  |
  +- CliverAgent(Agent)   -> wraps AgentCore (internal LLM)
  |
  +- CliAgent(Agent)      -> base for all CLI subprocess agents
       +- ClaudeAgent      -> claude CLI defaults + JSON parsing
       +- GeminiAgent      -> gemini CLI defaults + JSON parsing
       +- OpenCodeAgent    -> opencode CLI defaults + JSON parsing

All agents produce the same AgentResult / AgentChunk types.
```

## File Structure

```
src/cliver/
  agent.py                # Agent ABC, AgentResult, AgentChunk, Artifact dataclasses
  agents/
    __init__.py            # Re-exports: AGENT_REGISTRY, AgentFactory
    factory.py             # AgentFactory, AGENT_REGISTRY, type resolution
    cliver_agent.py        # CliverAgent -- wraps AgentCore
    cli_agent.py           # CliAgent base -- subprocess management, JSON parsing, artifact discovery
    claude_agent.py        # ClaudeAgent -- claude CLI defaults + response parser
    gemini_agent.py        # GeminiAgent -- gemini CLI defaults + response parser
    opencode_agent.py      # OpenCodeAgent -- opencode CLI defaults + response parser
  config.py                # Modified: add AgentConfig, agents dict, default_agent to AppConfig
```

---

## Data Models

### AgentResult

Unified result type returned by ALL agent types (CliverAgent and CliAgent alike). CliverAgent converts its BaseMessage response into AgentResult. CliAgent parses subprocess JSON output into AgentResult. Callers never need to know which agent type produced it.

```python
@dataclass
class Artifact:
    path: str              # absolute or relative file path
    media_type: str        # MIME type: image/png, application/pdf, video/mp4, etc.
    size: Optional[int]    # bytes, None if unknown
    description: str = ""  # agent's description of the artifact

@dataclass
class AgentResult:
    text: str                          # primary text output (Markdown, plain text, etc.)
    status: str                        # "completed" | "error" | "timeout"
    artifacts: List[Artifact] = field(default_factory=list)
    duration_ms: int = 0               # wall-clock execution time
    model: Optional[str] = None        # model name used (e.g. "anthropic/claude-sonnet-4-20250514")
    token_usage: Optional[dict] = None # {"input": N, "output": N} if available
    error: Optional[str] = None        # human-readable error message when status="error"
    raw: Optional[dict] = None         # raw JSON response (CLI agents only, for debugging)
```

### AgentChunk (streaming)

```python
@dataclass
class AgentChunk:
    text: str = ""                              # incremental text
    chunk_type: str = "text"                    # "text" | "status" | "artifact" | "done"
    artifact: Optional[Artifact] = None         # set when chunk_type="artifact"
    final_result: Optional[AgentResult] = None  # set on last chunk (chunk_type="done")
```

### Unified Result Contract

Both CliverAgent and CliAgent MUST produce identical AgentResult structure:

- CliverAgent: Calls AgentCore.process_user_input() -> receives BaseMessage -> converts to AgentResult. Text from message.content, artifacts from tool call results that produced files (images, audio, etc.), token usage from TokenTracker, model from AgentCore config.

- CliAgent: Runs subprocess with --output-format json -> receives JSON dict -> parses via _parse_response() into AgentResult. Each subclass (Claude/Gemini/OpenCode) overrides _parse_response() and _extract_artifacts() to handle their specific JSON format.

- Callers (Task, Notebook cell, Gateway) always receive AgentResult and never need to check which agent type produced it.

---

## Agent ABC and Lifecycle

```python
class Agent(ABC):
    name: str
    config: AgentConfig

    async def initialize(self, context: dict = None) -> None:
        """Called once before first run.

        Context may contain:
        - working_dir: str -- task/issue specific working directory
        - env: dict -- additional environment variables
        - output_dir: str -- directory for artifact output
        """
        pass  # default no-op, subclasses override

    async def run(self, prompt: str, *,
                  images: List[str] = None,
                  files: List[str] = None,
                  timeout_s: int = None,
                  **kwargs) -> AgentResult:
        """Execute prompt and return result.

        Rate limiting is applied before _do_run().
        Timeout defaults to self.config.timeout_s.
        Retry logic wraps _do_run() up to self.config.max_retries times.
        """
        if self._rate_limiter and self._rate_limit_key:
            await self._rate_limiter.acquire(self._rate_limit_key)

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

        return AgentResult(
            text="", status="timeout" if "Timeout" in last_error else "error",
            error=last_error, duration_ms=effective_timeout * 1000,
        )

    @abstractmethod
    async def _do_run(self, prompt: str, **kwargs) -> AgentResult:
        """Subclass implements actual execution logic."""
        ...

    async def stream(self, prompt: str, *,
                     images: List[str] = None,
                     files: List[str] = None,
                     timeout_s: int = None,
                     **kwargs) -> AsyncIterator[AgentChunk]:
        """Stream execution results. Default: wraps run() as single chunk."""
        result = await self.run(prompt, images=images, files=files,
                               timeout_s=timeout_s, **kwargs)
        yield AgentChunk(text=result.text, chunk_type="done",
                         final_result=result)

    async def cleanup(self) -> None:
        """Called on shutdown. Cleanup temp dirs, etc."""
        pass
```

### Lifecycle Flow

```
Task/Notebook creates agent via AgentFactory
  |
  +- agent = factory.create("researcher")
  +- await agent.initialize({"working_dir": "/path/to/project"})
  |
  +- result = await agent.run("Analyze this code")    # may be called multiple times
  +- result = await agent.run("Now fix the bug")
  |
  +- await agent.cleanup()                            # on task/notebook completion
```

---

## CliverAgent

Wraps the existing AgentCore. No changes to AgentCore itself.

```python
class CliverAgent(Agent):
    def __init__(self, name, config, agent_core, model_config=None,
                 provider_config=None, rate_limiter=None, **kwargs):
        self._agent_core = agent_core
        # ... store config, model, rate_limiter

    async def _do_run(self, prompt, *, images=None, files=None, **kwargs) -> AgentResult:
        start = time.monotonic()
        try:
            message = await self._agent_core.process_user_input(
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
                token_usage=self._get_token_usage(),
            )
        except Exception as e:
            duration = int((time.monotonic() - start) * 1000)
            return AgentResult(
                text="", status="error",
                error=str(e), duration_ms=duration,
            )

    async def stream(self, prompt, **kwargs) -> AsyncIterator[AgentChunk]:
        """True streaming via AgentCore.stream_user_input()."""
        start = time.monotonic()
        full_text = []
        async for chunk in self._agent_core.stream_user_input(
            user_input=prompt,
            model=self.config.model,
            **kwargs,
        ):
            text = chunk.content if hasattr(chunk, 'content') else str(chunk)
            full_text.append(text)
            yield AgentChunk(text=text, chunk_type="text")

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
        if not self.config.role:
            return None
        role = self.config.role
        return lambda: f"\n\nYour role: {role}"

    def _extract_artifacts_from_message(self, message) -> List[Artifact]:
        """Extract file artifacts from AgentCore tool call results."""
        artifacts = []
        # Parse tool call results for file paths (image generation, file writes)
        return artifacts
```

---

## CliAgent -- CLI Subprocess Base

Common base for all CLI-based agents. Handles subprocess execution, JSON output parsing, timeout, retry, and artifact discovery.

```python
class CliAgent(Agent):
    # Subclasses override these
    DEFAULT_COMMAND: str = ""
    DEFAULT_ARGS: List[str] = []
    DEFAULT_OUTPUT_FORMAT: List[str] = ["--output-format", "json"]

    def __init__(self, name, config, model_config=None,
                 provider_config=None, rate_limiter=None, **kwargs):
        self._command = config.command or self.DEFAULT_COMMAND
        self._args = config.args if config.args is not None else list(self.DEFAULT_ARGS)
        self._output_format = list(self.DEFAULT_OUTPUT_FORMAT)
        self._model_config = model_config
        self._provider_config = provider_config
        self._output_dir: Optional[Path] = None
        self._pre_snapshot: Optional[Set[str]] = None

    async def initialize(self, context: dict = None) -> None:
        """Verify CLI exists, create output directory, apply context."""
        import shutil
        if not shutil.which(self._command):
            raise RuntimeError(
                f"CLI agent '{self._command}' not found. "
                f"Install it or check your PATH."
            )
        self._output_dir = Path(tempfile.mkdtemp(prefix=f"cliver-{self.name}-"))
        if context and context.get("working_dir"):
            self.config.working_dir = context["working_dir"]

    async def _do_run(self, prompt, *, images=None, files=None, **kwargs) -> AgentResult:
        start = time.monotonic()

        # Snapshot files before execution (for artifact diff fallback)
        working = Path(self.config.working_dir or ".")
        self._pre_snapshot = self._snapshot_dir(working)

        # Build command
        cmd = [self._command] + self._args + self._output_format + [prompt]
        env = self._build_env()

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            cwd=self.config.working_dir or ".",
        )
        stdout, stderr = await proc.communicate()
        duration = int((time.monotonic() - start) * 1000)

        if proc.returncode != 0:
            return AgentResult(
                text=stderr.decode(errors="replace"),
                status="error",
                error=f"{self._command} exited with code {proc.returncode}",
                duration_ms=duration,
            )

        # Parse JSON response
        try:
            raw_json = json.loads(stdout.decode())
        except json.JSONDecodeError:
            # JSON parse failed -- return raw stdout as text
            return AgentResult(
                text=stdout.decode(errors="replace"),
                status="completed",
                artifacts=self._diff_artifacts(working),
                duration_ms=duration,
            )

        result = self._parse_response(raw_json)
        result.duration_ms = duration
        result.raw = raw_json

        # Artifact discovery: JSON first, then file diff fallback
        json_artifacts = self._extract_artifacts(raw_json)
        if json_artifacts:
            result.artifacts = json_artifacts
        else:
            result.artifacts = self._diff_artifacts(working)

        return result

    def _parse_response(self, raw_json: dict) -> AgentResult:
        """Override in subclasses for agent-specific JSON format."""
        text = raw_json.get("result", raw_json.get("response", str(raw_json)))
        return AgentResult(text=str(text), status="completed")

    def _extract_artifacts(self, raw_json: dict) -> List[Artifact]:
        """Override in subclasses to extract files from JSON response."""
        return []

    def _build_env(self) -> dict:
        """Build subprocess environment with provider credentials."""
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
        """Take a snapshot of files in directory for diff-based artifact discovery."""
        if not path.exists():
            return set()
        return {str(p) for p in path.rglob("*") if p.is_file()}

    def _diff_artifacts(self, working: Path) -> List[Artifact]:
        """Find new files created since pre_snapshot."""
        if self._pre_snapshot is None:
            return []
        current = self._snapshot_dir(working)
        new_files = current - self._pre_snapshot

        artifacts = []
        for f in new_files:
            p = Path(f)
            mime = mimetypes.guess_type(f)[0] or "application/octet-stream"
            artifacts.append(Artifact(
                path=f, media_type=mime,
                size=p.stat().st_size if p.exists() else None,
            ))
        return artifacts

    async def stream(self, prompt, **kwargs) -> AsyncIterator[AgentChunk]:
        """Stream via line-by-line stdout or stream-json if supported."""
        cmd = [self._command] + self._args

        stream_format = self._get_stream_format()
        if stream_format:
            cmd += stream_format
        else:
            cmd += self._output_format

        cmd.append(prompt)
        env = self._build_env()
        start = time.monotonic()

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            cwd=self.config.working_dir or ".",
        )

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
        """Override to provide stream-json format args. None = not supported."""
        return None

    def _parse_stream_chunk(self, chunk_json: dict) -> str:
        """Override to parse agent-specific stream-json chunks."""
        return chunk_json.get("text", chunk_json.get("content", str(chunk_json)))

    async def cleanup(self) -> None:
        if self._output_dir and self._output_dir.exists():
            import shutil
            shutil.rmtree(self._output_dir, ignore_errors=True)


# Provider type -> environment variable name mapping
ENV_VAR_MAPPING = {
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
```

---

## Built-in CLI Agent Subclasses

### ClaudeAgent

```python
class ClaudeAgent(CliAgent):
    DEFAULT_COMMAND = "claude"
    DEFAULT_ARGS = ["-p"]
    DEFAULT_OUTPUT_FORMAT = ["--output-format", "json"]

    def _parse_response(self, raw_json: dict) -> AgentResult:
        return AgentResult(
            text=str(raw_json.get("result", "")),
            status="error" if raw_json.get("is_error") else "completed",
            model=raw_json.get("model"),
            token_usage=raw_json.get("usage"),
            error=raw_json.get("result") if raw_json.get("is_error") else None,
        )

    def _extract_artifacts(self, raw_json: dict) -> List[Artifact]:
        artifacts = []
        for msg in raw_json.get("messages", []):
            if msg.get("role") == "assistant":
                for tc in msg.get("tool_calls", []):
                    if tc.get("name") in ("Write", "Edit"):
                        path = tc.get("input", {}).get("file_path")
                        if path:
                            mime = mimetypes.guess_type(path)[0] or "text/plain"
                            artifacts.append(Artifact(path=path, media_type=mime))
        return artifacts

    def _get_stream_format(self) -> Optional[List[str]]:
        return ["--output-format", "stream-json"]
```

### GeminiAgent

```python
class GeminiAgent(CliAgent):
    DEFAULT_COMMAND = "gemini"
    DEFAULT_ARGS = ["-p"]
    DEFAULT_OUTPUT_FORMAT = ["--output-format", "json"]

    def _parse_response(self, raw_json: dict) -> AgentResult:
        stats = raw_json.get("stats", {})
        return AgentResult(
            text=str(raw_json.get("response", "")),
            status="error" if raw_json.get("error") else "completed",
            token_usage=stats.get("token_usage"),
            error=raw_json.get("error", {}).get("message") if raw_json.get("error") else None,
        )

    def _extract_artifacts(self, raw_json: dict) -> List[Artifact]:
        artifacts = []
        stats = raw_json.get("stats", {})
        for f in stats.get("file_modifications", []):
            path = f.get("path", f) if isinstance(f, dict) else str(f)
            mime = mimetypes.guess_type(path)[0] or "text/plain"
            artifacts.append(Artifact(path=path, media_type=mime))
        return artifacts

    def _get_stream_format(self) -> Optional[List[str]]:
        return ["--output-format", "stream-json"]
```

### OpenCodeAgent

```python
class OpenCodeAgent(CliAgent):
    DEFAULT_COMMAND = "opencode"
    DEFAULT_ARGS = ["-p"]
    DEFAULT_OUTPUT_FORMAT = ["-f", "json"]

    def _parse_response(self, raw_json: dict) -> AgentResult:
        return AgentResult(
            text=str(raw_json.get("response", raw_json.get("result", ""))),
            status="error" if raw_json.get("error") else "completed",
            error=raw_json.get("error", {}).get("message") if raw_json.get("error") else None,
        )
```

---

## AgentFactory

```python
AGENT_REGISTRY: Dict[str, Type[Agent]] = {
    "cliver": CliverAgent,
    "claude": ClaudeAgent,
    "gemini": GeminiAgent,
    "opencode": OpenCodeAgent,
}

class AgentFactory:
    def __init__(self, config: AppConfig, agent_core: AgentCore):
        self._config = config
        self._agent_core = agent_core
        self._rate_limiter = RateLimiter()
        self._agents: Dict[str, Agent] = {}
        self._configure_rate_limits()

    def _configure_rate_limits(self):
        for name, provider in self._config.providers.items():
            if provider.rate_limit:
                key = f"{provider.api_url}|{provider.api_key or ''}"
                self._rate_limiter.configure(
                    key,
                    rpm=provider.rate_limit.rpm,
                    tpm=provider.rate_limit.tpm,
                )

    def create(self, name: str = None) -> Agent:
        name = name or self._config.default_agent or "default"

        if name in self._agents:
            return self._agents[name]

        agent_config = self._resolve_agent_config(name)
        model_config, provider_config = self._resolve_model(agent_config.model)

        agent_cls = AGENT_REGISTRY.get(agent_config.type, CliAgent)

        agent = agent_cls(
            name=name,
            config=agent_config,
            model_config=model_config,
            provider_config=provider_config,
            rate_limiter=self._rate_limiter,
            agent_core=self._agent_core if agent_config.type == "cliver" else None,
        )
        self._agents[name] = agent
        return agent

    def _resolve_agent_config(self, name: str) -> AgentConfig:
        agents = self._config.agents or {}
        if name in agents:
            return agents[name]
        return AgentConfig(type="cliver", model=self._config.default_model)

    def _resolve_model(self, model_name: str = None):
        if not model_name:
            model_name = self._config.default_model
        if not model_name:
            return None, None
        model_config = self._config.models.get(model_name)
        provider_config = model_config._provider_config if model_config else None
        return model_config, provider_config

    async def cleanup_all(self) -> None:
        for agent in self._agents.values():
            await agent.cleanup()
        self._agents.clear()
```

---

## AgentConfig

Added to src/cliver/config.py:

```python
class AgentConfig(BaseModel):
    type: str = Field(default="cliver", description="Agent type: cliver, claude, gemini, opencode, or custom")
    description: Optional[str] = Field(default=None, description="Human-readable purpose")
    role: Optional[str] = Field(default=None, description="System prompt / persona (cliver only)")
    model: Optional[str] = Field(default=None, description="Model name from models config")
    skills: List[str] = Field(default_factory=list, description="Pre-activated skills (cliver only)")
    command: Optional[str] = Field(default=None, description="CLI command override")
    args: Optional[List[str]] = Field(default=None, description="CLI args override")
    env: Optional[Dict[str, str]] = Field(default=None, description="Extra env vars for subprocess")
    working_dir: Optional[str] = Field(default=None, description="Working directory")
    timeout_s: int = Field(default=300, description="Execution timeout in seconds")
    max_retries: int = Field(default=0, description="Retry count on failure")
    auto_fallback: Optional[bool] = Field(default=None, description="Model auto-fallback (cliver only)")
```

AppConfig additions:

```python
class AppConfig(BaseModel):
    # ... existing fields ...
    agents: Dict[str, AgentConfig] = Field(default_factory=dict)
    default_agent: Optional[str] = Field(default=None)
```

### Config Example

```yaml
providers:
  anthropic:
    api_url: https://api.anthropic.com
    api_key: '<anthropic_key>'
    rate_limit:
      rpm: 60
      tpm: 100000
    models:
      claude-sonnet-4-20250514: {}

  deepseek:
    api_url: https://api.deepseek.com
    api_key: '<deepseek_key>'
    rate_limit:
      rpm: 30
    models:
      deepseek-r1: {}

agents:
  researcher:
    type: cliver
    model: deepseek/deepseek-r1
    role: "Research assistant specializing in academic paper analysis"
    skills: [brainstorm]

  coder:
    type: claude
    model: anthropic/claude-sonnet-4-20250514
    working_dir: ./src
    timeout_s: 600

  reviewer:
    type: gemini
    timeout_s: 300

  custom_agent:
    type: aider
    command: aider
    args: ["--message"]
    timeout_s: 600

default_agent: researcher
```

---

## Integration with Existing Code

### CLI (src/cliver/cli.py)

Add after AgentCore creation (line ~145):

```python
self.agent_factory = AgentFactory(self.config_manager.config, self.agent_core)
# CLI chat continues using agent_core directly (backward compatible)
```

### Gateway (src/cliver/gateway/gateway.py)

```python
# In _on_startup or create_app:
self._agent_factory = AgentFactory(resolved_config, self._agent_core)

# In _run_task:
agent_name = task.agent
agent = self._agent_factory.create(agent_name)
await agent.initialize({"working_dir": task_working_dir})
result = await agent.run(task.prompt)
# result is AgentResult -- use result.text, result.artifacts, etc.
await agent.cleanup()
```

### Task (src/cliver/task_manager.py)

```python
class TaskDefinition(BaseModel):
    agent: Optional[str] = None  # agent name, null = default_agent
    # ... rest unchanged
```

### agent_profile.py

```python
_agent_factory: Optional[AgentFactory] = None

def set_agent_factory(factory): ...
def get_agent_factory() -> AgentFactory: ...
```

---

## Rate Limiting

Rate limiting is managed at the Agent layer via a shared RateLimiter instance owned by AgentFactory.

- Rate limit buckets keyed by "{provider_url}|{api_key}" -- same API key shares the same bucket regardless of which agent uses it.
- CliverAgent and ClaudeAgent using the same Anthropic API key share one rate limit bucket.
- Agent.run() calls self._rate_limiter.acquire(key) before _do_run().
- The existing RateLimiter in AgentCore will be gradually migrated. During transition, both coexist.

---

## Artifact Discovery Strategy

Two-layer approach for maximum reliability:

1. Primary: JSON structured output parsing -- Each CLI agent runs with --output-format json. Subclass _extract_artifacts() parses agent-specific JSON to find created/modified files. This is the most reliable source.

2. Fallback: File system diff -- Before execution, CliAgent snapshots the working directory. After execution, it diffs to find new files. Used when JSON parsing yields no artifacts.

CliverAgent discovers artifacts from AgentCore tool call results (e.g., image generation tool returning a file path).

---

## Testing Strategy

- Unit tests for AgentResult/AgentChunk construction
- Unit tests for CliAgent._build_env() with various provider configs
- Unit tests for each subclass _parse_response() with sample JSON fixtures
- Unit tests for _extract_artifacts() with real Claude/Gemini/OpenCode JSON samples
- Unit tests for _diff_artifacts() file system diffing
- Integration test for CliverAgent wrapping AgentCore (mock AgentCore)
- Integration test for CliAgent subprocess execution (mock subprocess)
- Integration test for AgentFactory.create() with various configs
- Test rate limiter sharing across agents

---

## What This Design Does NOT Cover

- MCP server integration with CLI agents (future)
- Agent-to-agent communication (future)
- Parallel agent execution (future)
- Agent marketplace / sharing (covered by A9)
- Notebook cell integration (covered by A3/A4)
