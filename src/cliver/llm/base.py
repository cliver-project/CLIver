import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, AsyncIterator, List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessageChunk
from langchain_core.messages.base import BaseMessage
from langchain_core.tools import BaseTool

from cliver.config import ModelConfig
from cliver.llm.llm_utils import parse_tool_calls_from_content
from cliver.media import MediaContent
from cliver.model_capabilities import ModelCapability

logger = logging.getLogger(__name__)


class LLMInferenceEngine(ABC):
    def __init__(self, config: ModelConfig, user_agent: str = None, agent_name: str = "CLIver"):
        self.config = config or {}
        self.user_agent = user_agent
        self.agent_name = agent_name
        self.llm = None

    def supports_capability(self, capability: ModelCapability) -> bool:
        """Check if the model supports a specific capability."""
        return self.config.get_model_capabilities().supports(capability)

    # This method focus on the real LLM inference only.
    # Exceptions are NOT caught here — they propagate to the Re-Act loop
    # in AgentCore._process_messages(), which classifies and handles them
    # (retry, compress, failover) via error_classifier.py.
    async def infer(
        self,
        messages: list[BaseMessage],
        tools: Optional[list[BaseTool]],
        **kwargs: Any,
    ) -> BaseMessage:
        converted_messages = self.convert_messages_to_engine_specific(messages)
        self._log_message_summary(converted_messages)
        _llm = await self._reconstruct_llm(self.llm, tools)
        response = await _llm.ainvoke(converted_messages, **kwargs)
        return response

    # Exceptions are NOT caught here — they propagate to the Re-Act loop
    # in AgentCore._stream_messages(), which classifies and handles them
    # (retry, compress, failover) via error_classifier.py.
    async def stream(
        self,
        messages: List[BaseMessage],
        tools: Optional[list[BaseTool]],
        **kwargs: Any,
    ) -> AsyncIterator[BaseMessageChunk]:
        """Stream responses from the LLM."""
        converted_messages = self.convert_messages_to_engine_specific(messages)
        self._log_message_summary(converted_messages)
        _llm = await self._reconstruct_llm(self.llm, tools)
        # noinspection PyTypeChecker
        async for chunk in _llm.astream(input=converted_messages, **kwargs):
            yield chunk

    def _log_message_summary(self, messages: List[BaseMessage]) -> None:
        """Log a summary of message types and content blocks for debugging."""
        if not logger.isEnabledFor(logging.DEBUG):
            return
        for i, msg in enumerate(messages):
            role = msg.__class__.__name__
            content = msg.content
            if isinstance(content, str):
                logger.debug("  msg[%d] %s: text (%d chars)", i, role, len(content))
            elif isinstance(content, list):
                parts = []
                for block in content:
                    if isinstance(block, dict):
                        btype = block.get("type", "?")
                        if btype == "text":
                            parts.append(f"text({len(block.get('text', ''))}ch)")
                        elif btype in ("image_url", "image"):
                            parts.append(f"{btype}(base64)")
                        else:
                            parts.append(btype)
                    else:
                        parts.append(type(block).__name__)
                logger.debug("  msg[%d] %s: [%s]", i, role, ", ".join(parts))

    @abstractmethod
    def convert_messages_to_engine_specific(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        pass

    def extract_media_from_response(self, response: BaseMessage) -> List[MediaContent]:
        """
        Extract media content from LLM response.

        This method should be overridden by specific LLM engines to handle
        their specific response formats for multimedia content.

        Args:
            response: BaseMessage response from the LLM

        Returns:
            List of MediaContent objects extracted from the response
        """
        # Default implementation returns empty list
        # Specific engines should override this method
        return []

    async def transcribe_audio(self, file_path: Path, language: Optional[str] = None) -> Optional[str]:
        """Transcribe audio to text. Override in subclasses that support it.

        Returns the transcribed text, or None if not supported by this engine.
        """
        return None

    def parse_tool_calls(self, response: BaseMessage, model: str) -> list[dict] | None:
        """Parse tool calls from the LLM response.

        Returns raw tool calls in the format the LLM produced:
        [{"name": "...", "args": {...}, "id": "..."}]

        Engine subclasses can override this for provider-specific parsing.
        Normalization (arg coercion, validation, MCP splitting) is handled
        separately by normalize_tool_calls() in AgentCore.
        """
        if response is None:
            return None
        return parse_tool_calls_from_content(response)

    def system_message(self) -> str:
        """
        Build the builtin system prompt for CLIver.

        This method can be overridden by engine subclasses.
        User-provided system messages are appended separately by AgentCore.
        """
        sections = [self._section_identity(self.agent_name)]
        sections.append(self._section_self_awareness())
        sections.append(self._section_tool_usage())
        sections.append(self._section_interaction_guidelines())
        sections.append(self._section_response_format())

        return "\n\n".join(sections)

    # -- System prompt sections ------------------------------------------------

    @staticmethod
    def _section_identity(agent_name: str) -> str:
        import os
        from datetime import datetime, timezone

        cwd = os.getcwd()
        try:
            from cliver.util import format_datetime, get_effective_timezone

            tz = get_effective_timezone()
            tz_name = str(tz)
            now_aware = datetime.now(timezone.utc).astimezone(tz)
            utc_offset = now_aware.strftime("%z")
            now_local = format_datetime(fmt="%Y-%m-%d %H:%M:%S")
        except Exception:
            tz_name = "unknown"
            utc_offset = ""
            now_local = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        return (
            "# Identity\n\n"
            f"You are **{agent_name}**, a general-purpose AI agent. "
            "You help users accomplish a wide variety of tasks — answering questions, "
            "searching the web, reading and writing files, running commands, managing containers, "
            "and anything else the user asks for.\n\n"
            "You can operate in different environments: command-line interfaces, "
            "embedded applications, or as a backend service. "
            "Adapt your tone, depth, and approach to whatever the user needs.\n\n"
            "## Environment\n\n"
            f"- Working directory: `{cwd}`\n"
            f"- Local time: {now_local}\n"
            f"- Timezone: {tz_name} (UTC{utc_offset})\n\n"
            "- All file operations (read, write, list, grep) should be relative to this directory "
            "unless the user explicitly specifies an absolute path outside it.\n"
            "- Do NOT list or access `/`, `/etc`, `/usr`, or other system directories "
            "unless the user specifically asks for it.\n"
            "- When the user asks to create or save a file without specifying a path, "
            "save it in the current working directory."
        )

    @staticmethod
    def _section_self_awareness() -> str:
        from cliver.util import get_config_dir

        config_dir = get_config_dir()
        return (
            "# Self-Awareness\n\n"
            "You are powered by CLIver, a configurable AI agent platform. "
            "You can inspect and modify your own configuration, identity, "
            "skills, tasks, and workflows.\n\n"
            "## Key files you can read and edit\n\n"
            f"- Config: `{config_dir}/config.yaml` — models, providers, gateway, session settings\n"
            f"- Identity: `{config_dir}/agents/*/identity.md` — your persona and behavior\n"
            f"- Memory: `{config_dir}/agents/*/memory.md` — persistent knowledge\n"
            f"- Skills: `.cliver/skills/` (project) or `{config_dir}/skills/` (global) — SKILL.md files\n"
            f"- Tasks: `{config_dir}/agents/*/tasks/` — YAML task definitions with cron schedules\n\n"
            "## Commands\n\n"
            "Slash commands: model, config, gateway, session, permissions, "
            "mcp, skill, skills, identity, agent, cost, provider, task, workflow. "
            "Use the CliverHelp tool for syntax."
        )

    @staticmethod
    def _section_tool_usage() -> str:
        return (
            "# Tool Usage\n\n"
            "You have access to tools that extend your capabilities. "
            "The available tools, their parameter schemas (names, types, required/optional), "
            "and descriptions are provided alongside this message.\n\n"
            "## How to call tools\n\n"
            "- Use the **structured tool-calling mechanism** provided by the model API.\n"
            "- Use the **exact tool name** as given — do not invent or guess tool names.\n"
            "- Supply arguments that match the parameter schema (correct types, required fields).\n"
            "- You may call **multiple tools** in a single response when the calls are independent.\n\n"
            "## Fallback format (for models without structured tool calling)\n\n"
            "If the model does not support structured tool calling, respond with **only** "
            "the following JSON (no surrounding text):\n\n"
            "```json\n"
            "{\n"
            '  "tool_calls": [\n'
            "    {\n"
            '      "name": "exact_tool_name",\n'
            '      "args": { "param": "value" },\n'
            '      "id": "unique_uuid",\n'
            '      "type": "tool_call"\n'
            "    }\n"
            "  ]\n"
            "}\n"
            "```\n\n"
            "## Iterative tool use\n\n"
            "After each tool call you will receive the result. You may make **additional tool calls** "
            "based on the results until you have enough information to provide a final answer. "
            "This process can involve multiple rounds.\n\n"
            "If you already have enough information, respond directly without calling any tools."
        )

    @staticmethod
    def _section_interaction_guidelines() -> str:
        return (
            "# Interaction Guidelines\n\n"
            "## Asking the user\n\n"
            "Use the `Ask` tool when you need to:\n"
            "- Clarify ambiguous instructions\n"
            "- Confirm before destructive or irreversible actions\n"
            "- Choose between multiple valid approaches\n"
            "- Gather missing information that you cannot determine on your own\n\n"
            "Do **not** ask for confirmation on routine, safe, or clearly specified tasks.\n\n"
            "## Skills\n\n"
            "Use the `Skill` tool to discover and activate specialized skills. "
            "Call `Skill(skill_name='list')` to see available skills. "
            "When a task matches a skill's domain, activate it to get expert instructions.\n\n"
            "## Planning\n\n"
            "When you receive a request, assess its complexity:\n\n"
            "1. **Simple** (1-2 steps): Respond directly. Call tools as needed. No plan required.\n"
            "2. **Medium** (3-5 steps): Use `TodoWrite` to create a task list, then work through "
            "each task. Mark items `in_progress` when you begin and `completed` when done. "
            "Use `TodoRead` to review your progress at any time.\n"
            "3. **Complex** (5+ steps, design decisions, multi-file changes): "
            "Use the structured planning pipeline via builtin skills:\n"
            "   - `Skill('brainstorm')` — explore, clarify, design, get user approval\n"
            "   - `Skill('write-plan')` — create a detailed implementation plan\n"
            "   - `Skill('execute-plan')` — systematic execution with verification\n"
            "   Each skill guides you through its process step by step. "
            "You can also use any skill independently when appropriate.\n\n"
            "When using `TodoWrite`:\n"
            "- Create the full plan **first**, then start executing\n"
            "- Each call replaces the entire list — always include all items\n"
            "- Mark each task `in_progress` before starting, `completed` when done\n"
            "- After completing all tasks, provide a final summary to the user\n\n"
            "## Memory & Identity\n\n"
            "You have persistent memory and an identity profile that survive across conversations.\n\n"
            "**Memory** (`MemoryRead` / `MemoryWrite`): a curated knowledge base organized by topic.\n"
            "- Organize by topic headings (`## Project Setup`, `## User Preferences`), not chronologically\n"
            "- Before appending, read existing memory to avoid duplicates\n"
            "- Periodically consolidate: use `rewrite` mode to merge related entries, remove outdated ones, "
            "and keep the document concise\n"
            "- Keep entries factual and concise — no narratives or session-specific details\n"
            "- Do **not** save trivial, temporary, or obvious information\n\n"
            "**Identity** (`Identity`): a living markdown document describing who "
            "the user is (name, location, role, preferences) and how you should behave. "
            "Unlike memory, identity is **rewritten as a whole** — always include all existing "
            "information plus updates. Read the current identity first to avoid losing data.\n\n"
            "Update identity when the user shares personal info or states behavior preferences. "
            "Use memory for everything else — facts, events, decisions.\n\n"
            "## Error handling\n\n"
            "If a tool call fails, analyse the error and try an alternative approach "
            "rather than repeating the same call. "
            "If you are unable to resolve the issue, explain what went wrong to the user.\n\n"
            "## Security\n\n"
            "Never read, display, or log credentials, API keys, private keys, "
            "or other secrets. The `Read` tool will block access to known sensitive "
            "files (`.env`, `credentials.json`, `*.pem`, `*.key`, etc.). "
            "If you need to reference a secret, use a placeholder like `<API_KEY>` "
            "instead of the actual value."
        )

    @staticmethod
    def _section_response_format() -> str:
        return (
            "# Response Format\n\n"
            "- Respond in **Markdown** format for readability.\n"
            "- Be concise and direct; avoid unnecessary preamble.\n"
            "- When presenting structured data, use tables, lists, or code blocks as appropriate.\n"
            "- When a tool call is needed, respond with **only** the tool call — "
            "do not mix tool calls with prose in the same response."
        )

    async def _reconstruct_llm(self, _llm: BaseChatModel, tools: Optional[list[BaseTool]]) -> BaseChatModel:
        if tools and len(tools) > 0:
            capabilities = self.config.get_capabilities()
            if ModelCapability.TOOL_CALLING in capabilities:
                if ModelCapability.FUNCTION_CALLING in capabilities:
                    # Native OpenAI: use strict function calling
                    _llm = _llm.bind_tools(tools, strict=True)
                else:
                    # Other providers: bind tools without the strict field.
                    # Passing strict=False explicitly adds "strict": false to
                    # each function schema, which some providers reject.
                    _llm = _llm.bind_tools(tools)
        return _llm
