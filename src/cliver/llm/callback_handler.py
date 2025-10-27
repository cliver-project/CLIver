import logging
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGenerationChunk, GenerationChunk, LLMResult

from cliver.llm.llm_utils import parse_tool_calls_from_content, remove_thinking_sections

logger = logging.getLogger(__name__)


class CliverAsyncCallbackHandler(AsyncCallbackHandler):
    """
    Async callback handler for streaming mode that handles:
    - Tool call extraction
    - Thinking mode support with '<thinking>...</thinking>' and other patterns
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
        # Accumulate the token
        self.accumulated_content += token

        # Always check for both opening and closing tags
        lower_content = self.accumulated_content.lower()

        # Look for opening <thinking> tag
        thinking_start = lower_content.find("<thinking>")
        thinking_end = lower_content.find("</thinking>")

        # Determine thinking mode state
        if thinking_start != -1:
            if thinking_end != -1 and thinking_end > thinking_start:
                # We have both start and end tags, and end comes after start
                self.is_thinking_mode = False
                # Extract the thinking content
                self.thinking_content = self.accumulated_content[thinking_start + 10 : thinking_end]
            else:
                # We have start tag but no end tag (or end tag comes before start)
                self.is_thinking_mode = True
                # Extract current thinking content
                if len(self.accumulated_content) > thinking_start + 10:
                    self.thinking_content = self.accumulated_content[thinking_start + 10 :]
        else:
            # No start tag found
            self.is_thinking_mode = False

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
