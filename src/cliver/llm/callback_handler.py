import logging
import re
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGenerationChunk, GenerationChunk, LLMResult

from cliver.llm.llm_utils import is_thinking, parse_tool_calls_from_content, remove_thinking_sections

logger = logging.getLogger(__name__)

# Regex to extract the inner content from a thinking block
_THINK_EXTRACT = re.compile(
    r"<think(?:ing)?>(.*?)(?:</think(?:ing)?>|$)",
    re.IGNORECASE | re.DOTALL,
)


class CliverAsyncCallbackHandler(AsyncCallbackHandler):
    """
    Async callback handler for streaming mode that handles:
    - Tool call extraction
    - Thinking mode support with '<think>...</think>' and '<thinking>...</thinking>' tags
    - Proper error handling and logging
    """

    def __init__(self):
        super().__init__()
        self.thinking_content = None
        self.is_thinking_mode = False
        self.extracted_tool_calls = []
        self.final_response = None
        self.accumulated_content = ""

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

        # Detect thinking state using the shared utility
        self.is_thinking_mode = is_thinking(self.accumulated_content)

        # Extract thinking content if present
        match = _THINK_EXTRACT.search(self.accumulated_content)
        if match:
            self.thinking_content = match.group(1)

    async def on_llm_end(self, result: LLMResult, **kwargs: Any) -> None:
        """Handle LLM end event and extract final results."""
        try:
            # Extract final message from response
            if hasattr(result, "generations") and result.generations:
                generations = result.generations
                if generations and generations[0]:
                    generation = generations[0][0]  # First choice
                    self.final_response = generation.message
                    self.extracted_tool_calls = parse_tool_calls_from_content(self.final_response)
                    # I don't want the thinking content in the final response, it can be got
                    # using the get_thinking_content
                    # Create a new message with cleaned content
                    clean_content = remove_thinking_sections(
                        str(self.final_response.content) if self.final_response.content else ""
                    )
                    self.final_response = AIMessage(content=clean_content)

        except Exception as e:
            logger.error(f"Error in on_llm_end: {str(e)}", exc_info=True)

    def get_thinking_content(self) -> Optional[str]:
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
