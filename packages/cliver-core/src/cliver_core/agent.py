"""AgentCore — Re-Act loop with CLIverMessage, Provider, and MCPClient.

MCPClient is imported lazily inside methods so that AgentCore itself has no
module-level third-party dependencies beyond pydantic; importing cliver_core
stays light even though the MCP SDK is a hard dependency.

No langchain dependency. No model fallback. No compression (v1).
"""

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any, AsyncIterator, Awaitable, Callable

from cliver_core import capabilities
from cliver_core.bindings import CapabilityNotSupported, ModelBinding, ModelBindings
from cliver_core.events import (
    EventHandler,
    ToolEvent,
    ToolEventType,
)
from cliver_core.media import MediaContent
from cliver_core.media_jobs import MediaJob
from cliver_core.messages import (
    CLIverMessage,
    CLIverMessageChunk,
    ToolCall,
    UsageInfo,
)
from cliver_core.provider import (
    CLIverRequest,
    CLIverResponse,
    EmbeddingsResult,
    RerankItem,
)
from cliver_core.skill import Skill, SkillError, SkillManager, SkillNotFoundError
from cliver_core.tool import CLIverTool, ToolRegistry

if TYPE_CHECKING:
    from cliver_core.mcp import MCPClient


def _load_mcp_client():
    """Lazy import MCPClient — keeps the mcp SDK out of `import cliver_core`."""
    from cliver_core.mcp import MCPClient

    return MCPClient


logger = logging.getLogger(__name__)

_MAX_CONSECUTIVE_ERRORS = 5


class AgentCore:
    """Re-Act loop: send messages to LLM, execute tools, repeat.

    One AgentCore instance per model/provider combination.
    For multiple models, create multiple AgentCore instances
    (they can share the same ``mcp_client``, ``builtin_tools``, and
    ``skill_manager``).

    **Instance state** (bound at construction, reused across calls):
        bindings (``bindings.llm`` = chat model), builtin_tools, mcp_client,
        skill_manager, on_event, builtin_system_prompt

    **Call state** (passed per chat/stream, varies each turn):
        user_input, system_prompt, conversation, extra_tools, mcp_servers, options

    ``system_prompt`` (call-time) is appended to ``builtin_system_prompt``
    (construction-time) — the caller only needs to pass persona / extra context.

    ``bindings`` carries every model purpose — ``llm`` (required) drives
    chat/stream, and the remaining purposes (embed, rerank, image, tts, ...)
    back the embedded model APIs below.  An unbound non-llm purpose raises
    :class:`CapabilityNotSupported` — "not supported".  ``self.llm`` is the
    chat :class:`~cliver_core.bindings.ModelBinding`; its provider is built on
    first use, so read the provider via ``self.llm.get_provider()`` (its
    ``.provider`` attribute stays ``None`` until then).
    """

    def __init__(
        self,
        bindings: ModelBindings,
        builtin_tools: list[CLIverTool] | None = None,
        mcp_client: "MCPClient | None" = None,
        skill_manager: SkillManager | None = None,
        *,
        on_event: EventHandler | None = None,
        max_consecutive_errors: int = _MAX_CONSECUTIVE_ERRORS,
        builtin_system_prompt: str | None = None,
        logger: logging.Logger | None = None,
    ):
        # Every model purpose lives in ``bindings`` — chat included: the
        # Re-Act loop's provider + model come from ``bindings.llm``.
        llm = bindings.get("llm") if bindings is not None else None
        if llm is None:
            raise ValueError(
                "AgentCore requires an LLM binding — set bindings.llm "
                "(ModelBinding.llm(model='...', api_key='...', base_url='...')) for chat/stream."
            )
        self.llm = llm
        self.bindings = bindings
        self.tool_registry = ToolRegistry(builtin_tools or [])
        self.mcp_client = mcp_client
        self.skill_manager = skill_manager
        self.on_event = on_event
        self.max_consecutive_errors = max_consecutive_errors
        self._builtin_system_prompt = builtin_system_prompt
        self.logger = logger or logging.getLogger(__name__)

    # ── Public API ────────────────────────────────────────────

    async def chat(
        self,
        user_input: str,
        *,
        system_prompt: str | None = None,
        conversation: list[CLIverMessage] | None = None,
        media: list[MediaContent] | None = None,
        extra_tools: list[CLIverTool] | None = None,
        mcp_servers: list[str] | None = None,
        tool_filter: Callable[[CLIverTool], bool] | None = None,
        options: dict[str, Any] | None = None,
        max_iterations: int = 50,
    ) -> CLIverResponse:
        """Non-streaming chat.  Returns the final response.

        ``media`` is attached to the user message as content blocks
        (images as data URLs, audio/video as text placeholders).

        ``options`` are wire request options for the model: the llm binding's
        defaults (``bindings.llm.params``) apply underneath, and the call site
        wins on conflict (presence wins, an explicit ``None`` included).
        """
        messages = self._build_messages(user_input, system_prompt, conversation, media)
        if self.mcp_client:
            await self.mcp_client.start()
        all_tools = await self._gather_tools(extra_tools, mcp_servers, tool_filter)
        opts = {**self.llm.params, **(options or {})}

        return await self._run_loop(messages, all_tools, opts, max_iterations)

    async def stream(
        self,
        user_input: str,
        *,
        system_prompt: str | None = None,
        conversation: list[CLIverMessage] | None = None,
        media: list[MediaContent] | None = None,
        extra_tools: list[CLIverTool] | None = None,
        mcp_servers: list[str] | None = None,
        tool_filter: Callable[[CLIverTool], bool] | None = None,
        options: dict[str, Any] | None = None,
        max_iterations: int = 50,
    ) -> AsyncIterator[CLIverMessageChunk]:
        """Streaming chat.  Yields chunks, executes tools between iterations.

        ``options`` merge exactly as in :meth:`chat` — binding defaults under,
        call-site over.
        """
        messages = self._build_messages(user_input, system_prompt, conversation, media)
        if self.mcp_client:
            await self.mcp_client.start()
        all_tools = await self._gather_tools(extra_tools, mcp_servers, tool_filter)
        opts = {**self.llm.params, **(options or {})}

        async for chunk in self._run_stream_loop(messages, all_tools, opts, max_iterations):
            yield chunk

    async def run_skill(
        self,
        skill: Skill | str,
        user_input: str,
        *,
        system_prompt: str | None = None,
        conversation: list[CLIverMessage] | None = None,
        media: list[MediaContent] | None = None,
        extra_tools: list[CLIverTool] | None = None,
        mcp_servers: list[str] | None = None,
        options: dict[str, Any] | None = None,
        max_iterations: int = 50,
    ) -> CLIverResponse:
        """Run one skill: pin its instructions, restrict tools, run the loop.

        ``skill`` is a skill name (resolved through this agent's
        :class:`~cliver_core.skill.SkillManager`) or an already-loaded
        :class:`~cliver_core.skill.Skill` object — a router that picked it
        from ``list_skills()`` can pass it directly.  The skill body is
        appended to the single system message (caller ``system_prompt``
        first, skill instructions after); a declared ``allowed-tools``
        frontmatter narrows the advertised tools to that set for the run —
        a non-empty allowlist matching zero available tools fails fast.
        Every other :meth:`chat` parameter applies unchanged.
        """
        if isinstance(skill, str):
            manager = self.skill_manager
            if manager is None:
                raise SkillError(
                    "no SkillManager configured — pass skill_manager=... to AgentCore to enable run_skill()"
                )
            loaded = manager.get_skill(skill)
            if loaded is None:
                available = manager.get_skill_names()
                raise SkillNotFoundError(
                    f"skill '{skill}' not found. Available skills: {', '.join(available) or '(none)'}"
                )
        else:
            loaded = skill

        skill_prompt = self._format_skill_prompt(loaded)
        combined_system = "\n\n".join(part for part in (system_prompt, skill_prompt) if part) or None

        tool_filter: Callable[[CLIverTool], bool] | None = None
        if loaded.allowed_tools is not None:
            allowed = frozenset(loaded.allowed_tools)
            if allowed:
                advertised = await self._gather_tools(extra_tools, mcp_servers, None)
                names = {t.name for t in advertised}
                if not (allowed & names):
                    raise SkillError(
                        f"skill '{loaded.name}' allows only tools {sorted(allowed)}, none of which are "
                        f"available (available: {sorted(names)}). Register the tools or connect the "
                        "MCP servers the skill needs."
                    )
                dropped = sorted(allowed - names)
                if dropped:
                    self.logger.warning(
                        "skill '%s' allows tools that are not currently available (dropped for this run): %s",
                        loaded.name,
                        ", ".join(dropped),
                    )

            def skill_tool_filter(candidate: CLIverTool) -> bool:
                return candidate.name in allowed

            tool_filter = skill_tool_filter

        return await self.chat(
            user_input,
            system_prompt=combined_system,
            conversation=conversation,
            media=media,
            extra_tools=extra_tools,
            mcp_servers=mcp_servers,
            tool_filter=tool_filter,
            options=options,
            max_iterations=max_iterations,
        )

    @staticmethod
    def _format_skill_prompt(skill: Skill) -> str:
        """Frame a skill body for injection (progressive-disclosure level 2).

        The location line is required for skills that bundle scripts: the
        model cannot run ``scripts/*.py`` without knowing where they live.
        """
        return f"# Active skill: {skill.name}\n\nSkill location: {skill.base_dir}\n\n{skill.body}"

    # ── Embedded model APIs (bindings-gated) ──────────────────
    #
    # Each method works only when a model is bound for its purpose in
    # ``bindings``; otherwise it raises CapabilityNotSupported.  The bound
    # ModelBinding is passed to the standalone implementation in
    # ``cliver_core.capabilities`` as-is — model configuration (provider,
    # endpoint, model name, defaults) stays encapsulated in the binding and
    # is never re-passed as loose arguments; only call-site parameters are.
    # ``submit_*()`` variants return a MediaJob immediately — the generation
    # runs in the background on the current event loop.

    async def image(
        self,
        prompt: str,
        *,
        media: list[MediaContent] | None = None,
        output_dir: str | None = None,
        **options,
    ) -> CLIverResponse:
        """Generate an image with the bound image model → ``capabilities.generate_image``."""
        return await capabilities.generate_image(
            prompt=prompt,
            binding=self._require_binding("image"),
            media=media,
            output_dir=output_dir,
            **options,
        )

    def submit_image(self, prompt: str, **kwargs) -> MediaJob:
        """Start image generation; returns a MediaJob (see :meth:`image`)."""
        return self._submit("image", lambda: self.image(prompt, **kwargs))

    async def video(
        self,
        prompt: str,
        *,
        media: list[MediaContent] | None = None,
        output_dir: str | None = None,
        **options,
    ) -> CLIverResponse:
        """Generate a video with the bound video model → ``capabilities.generate_video``."""
        return await capabilities.generate_video(
            prompt=prompt,
            binding=self._require_binding("video"),
            media=media,
            output_dir=output_dir,
            **options,
        )

    def submit_video(self, prompt: str, **kwargs) -> MediaJob:
        """Start video generation; returns a MediaJob (see :meth:`video`)."""
        return self._submit("video", lambda: self.video(prompt, **kwargs))

    async def tts(self, prompt: str, *, output_dir: str | None = None, **options) -> CLIverResponse:
        """Synthesize speech from *prompt* with the bound TTS model → ``capabilities.text_to_speech``."""
        return await capabilities.text_to_speech(
            text=prompt,
            binding=self._require_binding("tts"),
            output_dir=output_dir,
            **options,
        )

    def submit_tts(self, prompt: str, **kwargs) -> MediaJob:
        """Start speech synthesis; returns a MediaJob (see :meth:`tts`)."""
        return self._submit("tts", lambda: self.tts(prompt, **kwargs))

    async def audio(self, prompt: str, *, output_dir: str | None = None, **options) -> CLIverResponse:
        """Legacy alias of :meth:`tts` (uses the ``audio`` binding)."""
        return await capabilities.text_to_speech(
            text=prompt,
            binding=self._require_binding("audio"),
            output_dir=output_dir,
            **options,
        )

    def submit_audio(self, prompt: str, **kwargs) -> MediaJob:
        """Start audio generation; returns a MediaJob (see :meth:`audio`)."""
        return self._submit("audio", lambda: self.audio(prompt, **kwargs))

    async def asr(
        self,
        audio: MediaContent | bytes | str,
        *,
        language: str | None = None,
        **options,
    ) -> CLIverResponse:
        """Transcribe *audio* with the bound ASR model → ``capabilities.transcribe_audio``.

        *audio* may be a :class:`MediaContent`, raw ``bytes`` (assumed WAV),
        or a file path / URL.  ``language=None`` leaves any binding default in place.
        """
        return await capabilities.transcribe_audio(
            audio=audio,
            binding=self._require_binding("asr"),
            language=language,
            **options,
        )

    def submit_asr(self, audio: MediaContent | bytes | str, **kwargs) -> MediaJob:
        """Start transcription; returns a MediaJob (see :meth:`asr`)."""
        return self._submit("asr", lambda: self.asr(audio, **kwargs))

    async def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        top_k: int | None = None,
        **options,
    ) -> list[RerankItem]:
        """Rerank *documents* against *query* with the bound rerank model → ``capabilities.rerank``.

        ``top_k=None`` leaves any binding default in place.
        """
        return await capabilities.rerank(
            query=query,
            documents=documents,
            binding=self._require_binding("rerank"),
            top_k=top_k,
            **options,
        )

    async def embed(self, texts: str | list[str], **options) -> EmbeddingsResult:
        """Embed *texts* with the bound embedding model → ``capabilities.embed``."""
        return await capabilities.embed(
            texts=texts,
            binding=self._require_binding("embed"),
            **options,
        )

    # ── Embedded API plumbing ─────────────────────────────────

    def _require_binding(self, purpose: str) -> ModelBinding:
        binding = self.bindings.get(purpose)
        if binding is None:
            example = "tts" if purpose == "audio" else purpose  # 'audio' is a legacy alias of tts
            raise CapabilityNotSupported(
                f"No model bound for '{purpose}' — set bindings.{purpose} "
                f"(e.g. ModelBinding.{example}(model='...', api_key='...', base_url='...')) to enable it."
            )
        return binding

    def _submit(self, purpose: str, factory: Callable[[], Awaitable[Any]]) -> MediaJob:
        """Schedule *factory* as a background job and return its handle.

        The binding is checked synchronously, so unsupported capabilities
        fail fast at submit time.  *factory* is a zero-arg callable (not a
        coroutine) so nothing is left un-awaited when the check fails.
        """
        self._require_binding(purpose)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError as exc:
            raise RuntimeError(
                "submit_*() requires a running event loop — use the awaitable "
                f"form (agent.{purpose}(...)) outside async code."
            ) from exc
        return MediaJob(purpose, loop.create_task(factory()))

    # ── Lifecycle ─────────────────────────────────────────────

    async def aclose(self) -> None:
        """Release resources — close the providers the bindings built.

        Covers ``bindings.llm`` too: a chat provider the binding built (from
        ``api_key`` / ``base_url``) is closed here.  A caller-supplied
        ``provider=`` is never closed — whoever built it owns it.
        """
        await self.bindings.aclose()

    # ── Re-Act Loop (non-streaming) ───────────────────────────

    async def _run_loop(
        self,
        messages: list[CLIverMessage],
        tools: list[CLIverTool],
        options: dict[str, Any],
        max_iterations: int,
    ) -> CLIverResponse:
        consecutive_errors = 0
        diagnostic_trace: list[dict] = []
        # Built on first use and cached by the binding (lazy by design).
        provider = self.llm.get_provider()

        for _iteration in range(max_iterations):
            request = CLIverRequest(
                messages=messages,
                tools=tools or None,
                model=self.llm.model,
                options=options,
            )

            response = await provider.chat(request)
            msg = response.message

            # Collect diagnostic data for this Re-Act iteration
            diagnostic_trace.append(
                {
                    "iteration": _iteration,
                    "model": self.llm.model,
                    "provider": provider.provider_name(),
                    "request": response.request_info,
                    "response": msg.native,
                    "message": msg.model_dump(exclude={"native", "native_protocol", "trace"}),
                }
            )

            if not msg.tool_calls:
                msg.trace = diagnostic_trace
                return response

            messages.append(msg)
            consecutive_errors, stop = await self._execute_tool_calls(
                messages, msg.tool_calls, consecutive_errors, tools
            )
            if stop:
                return CLIverResponse(
                    message=CLIverMessage(
                        role="assistant",
                        content="Tool calls failed repeatedly. Please check the tool arguments or try a different "
                        "approach.",
                        trace=diagnostic_trace,
                    ),
                    usage=response.usage,
                )

        return CLIverResponse(
            message=CLIverMessage(
                role="assistant",
                content="Reached maximum iterations without a final answer.",
                trace=diagnostic_trace,
            )
        )

    # ── Re-Act Loop (streaming) ───────────────────────────────

    async def _run_stream_loop(
        self,
        messages: list[CLIverMessage],
        tools: list[CLIverTool],
        options: dict[str, Any],
        max_iterations: int,
    ) -> AsyncIterator[CLIverMessageChunk]:
        import json

        consecutive_errors = 0
        diagnostic_trace: list[dict] = []
        # Built on first use and cached by the binding (lazy by design).
        provider = self.llm.get_provider()

        for _iteration in range(max_iterations):
            request = CLIverRequest(
                messages=messages,
                tools=tools or None,
                model=self.llm.model,
                options=options,
            )

            final_native: dict[str, Any] | None = None
            final_usage: UsageInfo | None = None

            async for raw in provider.stream(request):
                if raw.native is not None:
                    # Terminal chunk — the SDK-aggregated native assistant
                    # message; consumed here, not forwarded.
                    final_native = raw.native
                    final_usage = raw.usage
                    continue
                if raw.content or raw.vendor_ext:
                    yield CLIverMessageChunk(
                        content=raw.content,
                        vendor_ext=raw.vendor_ext,
                    )

            if final_native is None:
                raise RuntimeError("Provider stream ended without a final native message")
            # One conversion, exactly like the non-streaming path — the
            # streamed assistant keeps its native wire form and
            # passthrough-converts on the next request.
            assistant = CLIverMessage.from_native(final_native, provider.protocol)
            tool_calls = assistant.tool_calls or []

            # Collect diagnostic data for this streaming iteration.
            # Streaming has no response object — the request echo is built
            # inline and the message entry is fields-only.
            diagnostic_trace.append(
                {
                    "iteration": _iteration,
                    "model": self.llm.model,
                    "provider": provider.provider_name(),
                    "request": {
                        "model": request.model,
                        "options": request.options,
                        "system_prompt": next(
                            (m.content for m in request.messages if m.role == "system"),
                            None,
                        ),
                        "messages": [
                            m.model_dump(exclude={"native", "native_protocol", "trace"}) for m in request.messages
                        ],
                        "tools": [t.model_dump(exclude={"execute"}) for t in (request.tools or [])],
                    },
                    "message": assistant.model_dump(exclude={"native", "native_protocol"}),
                    "usage": final_usage.model_dump() if final_usage else None,
                }
            )

            if not tool_calls:
                # Final answer — emit trace as a sentinel chunk
                yield CLIverMessageChunk(
                    vendor_ext={"__llm_trace__": json.dumps(diagnostic_trace)},
                )
                return

            messages.append(assistant)

            consecutive_errors, stop = await self._execute_tool_calls(messages, tool_calls, consecutive_errors, tools)
            if stop:
                yield CLIverMessageChunk(
                    content="Tool calls failed repeatedly. Please check the tool arguments or try a different "
                    "approach.",
                )
                yield CLIverMessageChunk(
                    vendor_ext={"__llm_trace__": json.dumps(diagnostic_trace)},
                )
                return

        yield CLIverMessageChunk(
            content="Reached maximum iterations without a final answer.",
        )
        yield CLIverMessageChunk(
            vendor_ext={"__llm_trace__": json.dumps(diagnostic_trace)},
        )

    # ── Tool execution (shared by both loops) ─────────────────

    async def _execute_tool_calls(
        self,
        messages: list[CLIverMessage],
        tool_calls: list[ToolCall],
        consecutive_errors: int,
        tools: list[CLIverTool],
    ) -> tuple[int, bool]:
        """Execute a batch of tool calls and append results to messages.

        ``tools`` is the exact set this call advertised to the model
        (``_gather_tools`` output) — execution resolves against it, so
        call-time ``extra_tools`` run alongside registry tools.

        Returns (consecutive_errors, should_stop).
        """
        by_name = {t.name: t for t in tools}
        for tc in tool_calls:
            result = await self._execute_tool(tc, by_name.get(tc.name))
            messages.append(
                CLIverMessage(
                    role="tool",
                    content=self._format_tool_result(result),
                    tool_call_id=tc.id,
                )
            )

            if self._is_error(result):
                consecutive_errors += 1
                if consecutive_errors >= self.max_consecutive_errors:
                    return consecutive_errors, True
            else:
                consecutive_errors = 0

        return consecutive_errors, False

    # ── Helpers ────────────────────────────────────────────────

    def _build_messages(
        self,
        user_input: str,
        system_prompt: str | None,
        conversation: list[CLIverMessage] | None,
        media: list[MediaContent] | None = None,
    ) -> list[CLIverMessage]:
        # Merge builtin system prompt + caller's extra context into one message.
        # Conversation shaping (Anthropic's _split_system, the message's native
        # builders) expects a single system message — merging here avoids
        # provider-level complexity.
        parts: list[str] = []
        if self._builtin_system_prompt:
            parts.append(self._builtin_system_prompt)
        if system_prompt:
            parts.append(system_prompt)
        merged_system = "\n\n".join(parts) if parts else None

        messages: list[CLIverMessage] = []
        if merged_system:
            messages.append(CLIverMessage(role="system", content=merged_system))
        if conversation:
            messages.extend(conversation)

        if media:
            # Embed media into the user message as content blocks
            content_parts: list[dict] = [{"type": "text", "text": user_input}]
            from cliver_core.media import add_media_content_to_message_parts

            add_media_content_to_message_parts(content_parts, media)
            messages.append(CLIverMessage(role="user", content=content_parts))
        else:
            messages.append(CLIverMessage(role="user", content=user_input))
        return messages

    async def _gather_tools(
        self,
        extra: list[CLIverTool] | None,
        mcp_servers: list[str] | None,
        tool_filter: Callable[[CLIverTool], bool] | None = None,
    ) -> list[CLIverTool]:
        tools = list(self.tool_registry.all_tools)
        if self.mcp_client:
            tools.extend(await self.mcp_client.get_tools(servers=mcp_servers or None))
        if extra:
            tools.extend(extra)
        if tool_filter:
            tools = [t for t in tools if tool_filter(t)]
        return tools

    async def _execute_tool(self, tc: ToolCall, tool: CLIverTool | None = None) -> list[dict]:
        """Execute a single tool call. Emits TOOL_START/TOOL_END/TOOL_ERROR events.

        ``tool`` is the advertised definition for this call (registry or
        ``extra_tools``); the registry lookup is the fallback for a call to a
        registered-but-unadvertised builtin.
        """
        await self._emit(
            ToolEvent(
                event=ToolEventType.START,
                tool_name=tc.name,
                tool_call_id=tc.id,
                args=tc.args,
            )
        )
        start = time.monotonic()
        try:
            if tc.name.startswith("mcp__"):
                if not self.mcp_client:
                    result = [{"error": "No MCP client configured"}]
                else:
                    result = await self.mcp_client.call_tool(tc.name, tc.args)
            else:
                resolved = tool or self.tool_registry.get(tc.name)
                if resolved is None:
                    result = [{"error": f"Tool '{tc.name}' not found"}]
                else:
                    result = await asyncio.to_thread(resolved.execute, **tc.args)

            duration = (time.monotonic() - start) * 1000
            await self._emit(
                ToolEvent(
                    event=ToolEventType.END,
                    tool_name=tc.name,
                    tool_call_id=tc.id,
                    result=_truncate(self._format_tool_result(result), 500),
                    duration_ms=duration,
                )
            )
            return result
        except Exception as e:
            duration = (time.monotonic() - start) * 1000
            self.logger.warning("Tool '%s' failed: %s", tc.name, e)
            await self._emit(
                ToolEvent(
                    event=ToolEventType.ERROR,
                    tool_name=tc.name,
                    tool_call_id=tc.id,
                    error=str(e),
                    duration_ms=duration,
                )
            )
            return [{"error": str(e)}]

    async def _emit(self, event: ToolEvent) -> None:
        if self.on_event:
            await self.on_event(event)

    @staticmethod
    def _is_error(result: list[dict]) -> bool:
        """Check if any result dict contains an 'error' key."""
        if not result:
            return False
        return any(isinstance(r, dict) and "error" in r for r in result)

    @staticmethod
    def _format_tool_result(result: list[dict]) -> str:
        """Format tool result for a ToolMessage."""
        if not result:
            return "(no output)"

        if len(result) == 1 and isinstance(result[0], dict):
            r = result[0]
            if "error" in r:
                return f"Error: {r['error']}"
            if "text" in r:
                return str(r["text"])
            if "tool_result" in r:
                return str(r["tool_result"])

        import json

        return json.dumps(result, ensure_ascii=False, default=str)


def _truncate(text: str, max_len: int) -> str:
    """Truncate a string for event display."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."
