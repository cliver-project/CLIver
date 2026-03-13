import logging
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGenerationChunk, GenerationChunk, LLMResult

from cliver.llm.llm_utils import (
    extract_reasoning,
    is_thinking,
    parse_tool_calls_from_content,
    remove_thinking_sections,
)

logger = logging.getLogger(__name__)


class CliverAsyncCallbackHandler(AsyncCallbackHandler):
    """
    Async callback handler for streaming mode that handles:
    - Tool call extraction
    - Reasoning/thinking content extraction (via API fields or <think> tags)
    - Proper error handling and logging
    """

    def __init__(self):
        super().__init__()
        self.thinking_content = None
        self.is_thinking_mode = False
        self.extracted_tool_calls = []
        self.final_response = None
        self.accumulated_content = ""
        self._reasoning_parts: list[str] = []

    async def on_llm_new_token(
        self,
        token: str,
        *,
        chunk: Optional[Union[GenerationChunk, ChatGenerationChunk]] = None,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Handle new tokens and detect thinking mode."""
        self.accumulated_content += token

        # Check for structured reasoning in chunk's additional_kwargs
        if chunk and hasattr(chunk, "message"):
            msg = chunk.message
            chunk_kwargs = getattr(msg, "additional_kwargs", {}) or {}
            reasoning = chunk_kwargs.get("reasoning_content") or chunk_kwargs.get("reasoning")
            if reasoning and isinstance(reasoning, str):
                self._reasoning_parts.append(reasoning)
                self.is_thinking_mode = True
                return

        # Fallback: detect <think> tags in accumulated content
        self.is_thinking_mode = is_thinking(self.accumulated_content)

    async def on_llm_end(self, result: LLMResult, **kwargs: Any) -> None:
        """Handle LLM end event and extract final results."""
        try:
            if hasattr(result, "generations") and result.generations:
                generations = result.generations
                if generations and generations[0]:
                    generation = generations[0][0]  # First choice
                    self.final_response = generation.message
                    self.extracted_tool_calls = parse_tool_calls_from_content(self.final_response)

                    # Extract reasoning: structured API field first, <think> tags as fallback
                    reasoning = extract_reasoning(self.final_response)
                    if reasoning:
                        self.thinking_content = reasoning
                    elif self._reasoning_parts:
                        self.thinking_content = "".join(self._reasoning_parts)

                    # Remove <think> tags from final response content
                    clean_content = remove_thinking_sections(
                        str(self.final_response.content) if self.final_response.content else ""
                    )
                    self.final_response = AIMessage(content=clean_content)

        except Exception as e:
            logger.error(f"Error in on_llm_end: {str(e)}", exc_info=True)

    def get_thinking_content(self) -> Optional[str]:
        """Get extracted reasoning/thinking content."""
        return self.thinking_content

    def get_full_response(self) -> Optional[BaseMessage]:
        """Get the full accumulated response."""
        return self.final_response

    def get_tool_calls(self) -> List[Dict[str, Any]]:
        """Get extracted tool calls."""
        return self.extracted_tool_calls

    def is_in_thinking_mode(self) -> bool:
        """Check if the response is in thinking mode."""
        return self.is_thinking_mode
