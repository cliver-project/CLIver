import logging
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessageChunk
from langchain_core.messages.base import BaseMessage
from langchain_core.tools import BaseTool

from cliver.config import ModelConfig
from cliver.llm.errors import get_friendly_error_message
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
    async def infer(
        self,
        messages: list[BaseMessage],
        tools: Optional[list[BaseTool]],
        **kwargs: Any,
    ) -> BaseMessage:
        try:
            # Convert messages to LLM engine format if needed
            converted_messages = self.convert_messages_to_engine_specific(messages)
            _llm = await self._reconstruct_llm(self.llm, tools)
            response = await _llm.ainvoke(converted_messages, **kwargs)
            return response
        except Exception as e:
            logger.debug(f"Error in infer: {str(e)}", exc_info=True)
            friendly_error_msg = [
                get_friendly_error_message(e, "LLM inference"),
                f"\tmodel: {self.config.name}",
            ]
            return AIMessage(
                content=f"Error: {'\n'.join(friendly_error_msg)}",
                additional_kwargs={"type": "error"},
            )

    async def stream(
        self,
        messages: List[BaseMessage],
        tools: Optional[list[BaseTool]],
        **kwargs: Any,
    ) -> AsyncIterator[BaseMessageChunk]:
        """Stream responses from the LLM."""
        try:
            # Convert messages to OpenAI multi-media format if needed
            converted_messages = self.convert_messages_to_engine_specific(messages)
            _llm = await self._reconstruct_llm(self.llm, tools)
            # noinspection PyTypeChecker
            async for chunk in _llm.astream(input=converted_messages, **kwargs):
                yield chunk
        except Exception as e:
            logger.debug(f"Error in OpenAI stream: {str(e)}", exc_info=True)
            friendly_error_msg = [
                get_friendly_error_message(e, "OpenAI inference"),
                f"\tmodel: {self.config.name}",
            ]
            # noinspection PyArgumentList
            yield AIMessageChunk(
                content=f"Error: {'\n'.join(friendly_error_msg)}",
                additional_kwargs={"type": "error"},
            )

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

    def parse_tool_calls(self, response: BaseMessage, model: str) -> list[dict] | None:
        """Parse tool calls from the LLM response.

        Returns raw tool calls in the format the LLM produced:
        [{"name": "...", "args": {...}, "id": "..."}]

        Engine subclasses can override this for provider-specific parsing.
        Normalization (arg coercion, validation, MCP splitting) is handled
        separately by normalize_tool_calls() in TaskExecutor.
        """
        if response is None:
            return None
        return parse_tool_calls_from_content(response)

    def system_message(self) -> str:
        """
        Build the builtin system prompt for CLIver.

        This method can be overridden by engine subclasses.
        User-provided system messages are appended separately by TaskExecutor.
        """
        sections = [self._section_identity(self.agent_name)]
        sections.append(self._section_tool_usage())
        sections.append(self._section_interaction_guidelines())
        sections.append(self._section_response_format())

        return "\n\n".join(sections)

    # -- System prompt sections ------------------------------------------------

    @staticmethod
    def _section_identity(agent_name: str) -> str:
        return (
            "# Identity\n\n"
            f"You are **{agent_name}**, a general-purpose AI agent. "
            "You help users accomplish a wide variety of tasks — answering questions, "
            "searching the web, reading and writing files, running commands, managing containers, "
            "and anything else the user asks for.\n\n"
            "You can operate in different environments: command-line interfaces, "
            "embedded applications, or as a backend service. "
            "Adapt your tone, depth, and approach to whatever the user needs."
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
            "Use the `ask_user_question` tool when you need to:\n"
            "- Clarify ambiguous instructions\n"
            "- Confirm before destructive or irreversible actions\n"
            "- Choose between multiple valid approaches\n"
            "- Gather missing information that you cannot determine on your own\n\n"
            "Do **not** ask for confirmation on routine, safe, or clearly specified tasks.\n\n"
            "## Skills\n\n"
            "Use the `skill` tool to discover and activate specialized skills. "
            "Call `skill(skill_name='list')` to see available skills. "
            "When a task matches a skill's domain, activate it to get expert instructions.\n\n"
            "## Planning\n\n"
            "When you receive a request, assess its complexity:\n\n"
            "1. **Simple** (1-2 steps): Respond directly. Call tools as needed. No plan required.\n"
            "2. **Medium** (3-5 steps): Use `todo_write` to create a task list, then work through "
            "each task. Mark items `in_progress` when you begin and `completed` when done. "
            "Use `todo_read` to review your progress at any time.\n"
            "3. **Complex** (5+ steps, different models needed, long-running): Use `create_workflow` "
            "to generate a workflow YAML. Each step runs in its own context with its own model. "
            "The workflow engine handles execution, caching, and pause/resume.\n\n"
            "When using `todo_write`:\n"
            "- Create the full plan **first**, then start executing\n"
            "- Each call replaces the entire list — always include all items\n"
            "- Mark each task `in_progress` before starting, `completed` when done\n"
            "- After completing all tasks, provide a final summary to the user\n\n"
            "When using `create_workflow`:\n"
            "- Each step should do ONE thing well (3-8 steps ideal)\n"
            "- Reference previous step outputs with `{{ step_id.outputs.key }}`\n"
            "- Reference workflow inputs with `{{ inputs.key }}`\n"
            "- Specify different models per step when beneficial\n"
            "- Keep workflows linear — no complex branching\n\n"
            "## Memory & Identity\n\n"
            "You have persistent memory and an identity profile that survive across conversations.\n\n"
            "**Memory** (`memory_read` / `memory_write`): append-only log of facts. "
            "Use for event-based knowledge — things that happened, decisions made, corrections received.\n\n"
            "**Identity** (`identity_update`): a living markdown document describing who "
            "the user is (name, location, role, preferences) and how you should behave. "
            "Unlike memory, identity is **rewritten as a whole** — always include all existing "
            "information plus updates. Read the current identity first to avoid losing data.\n\n"
            "Update identity when the user shares personal info or states behavior preferences. "
            "Use memory for everything else — facts, events, decisions.\n\n"
            "Do **not** save trivial, temporary, or session-specific information.\n\n"
            "## Error handling\n\n"
            "If a tool call fails, analyse the error and try an alternative approach "
            "rather than repeating the same call. "
            "If you are unable to resolve the issue, explain what went wrong to the user."
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
                # Only use strict=True for providers that fully support OpenAI's
                # strict function calling (native OpenAI). Most OpenAI-compatible
                # providers (DeepSeek, Qwen, GLM, vLLM) don't support it.
                use_strict = ModelCapability.FUNCTION_CALLING in capabilities
                _llm = _llm.bind_tools(tools, strict=use_strict)
        return _llm
