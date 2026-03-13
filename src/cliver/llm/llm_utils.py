import logging
import re
from typing import Optional

import json_repair
from langchain_core.messages.base import BaseMessage

logger = logging.getLogger(__name__)


def parse_tool_calls_from_content(response: BaseMessage) -> Optional[list[dict]]:
    """Parse tool calls from response content when LLM doesn't properly use tool binding."""
    if response is None:
        return None
    if hasattr(response, "tool_calls") and response.tool_calls:
        return response.tool_calls
    if hasattr(response, "content") and response.content:
        if isinstance(response.content, dict) and "tool_calls" in response.content:
            content_dict = dict(response.content)
            return content_dict.get("tool_calls", [])
    if hasattr(response, "content") and response.content and '"tool_calls"' in str(response.content):
        try:
            content_str = str(response.content)

            # Look for tool_calls pattern in the content
            # This pattern matches the JSON structure we expect
            pattern = r'\{[^{]*"tool_calls":\s*\[[^\]]*\][^\}]*\}'
            match = re.search(pattern, content_str, re.DOTALL)

            if match:
                # Extract the complete JSON object containing tool_calls
                tool_calls_section = match.group(0)
                parsed = json_repair.loads(tool_calls_section)
                tool_calls = parsed.get("tool_calls", [])
                return tool_calls

            # If the above pattern doesn't work, try to find just the tool_calls array
            pattern = r'"tool_calls":\s*(\[[^\]]*\])'
            match = re.search(pattern, content_str, re.DOTALL)

            if match:
                # Extract just the tool_calls array
                tool_calls_array = match.group(1)
                tool_calls = json_repair.loads(tool_calls_array)
                return tool_calls
        except Exception as e:
            # If parsing fails, return None
            logger.debug(f"Error parsing tool calls: {str(e)}", exc_info=True)
            return None
    return None


# Supported thinking tag formats:
#   <think>...</think>    — DeepSeek, Qwen3, GLM-4.5/4.7 (standard)
#   <thinking>...</thinking> — legacy / custom models
_THINK_OPEN_TAGS = ["<think>", "<thinking>"]
_THINK_CLOSE_TAGS = ["</think>", "</thinking>"]
_THINK_REMOVE_PATTERN = re.compile(
    r"<think(?:ing)?>.*?</think(?:ing)?>",
    re.IGNORECASE | re.DOTALL,
)


def is_thinking(content: str) -> bool:
    """Check if content is currently inside an unclosed thinking block."""
    lower_content = content.lower()
    # Find the last opening tag
    last_open = max(lower_content.rfind(tag) for tag in _THINK_OPEN_TAGS)
    if last_open == -1:
        return False
    # Check if there's a closing tag after the last opening tag
    for tag in _THINK_CLOSE_TAGS:
        close_pos = lower_content.find(tag, last_open)
        if close_pos != -1:
            return False
    # Open tag found but no closing tag after it
    return True


def is_thinking_content(content: str) -> bool:
    """Check if content contains a complete thinking block."""
    return bool(_THINK_REMOVE_PATTERN.search(content))


def remove_thinking_sections(content: str) -> str:
    """Remove thinking sections from content to find tool calls in the remaining text."""
    return _THINK_REMOVE_PATTERN.sub("", content)


_THINK_EXTRACT_PATTERN = re.compile(
    r"<think(?:ing)?>(.*?)(?:</think(?:ing)?>|$)",
    re.IGNORECASE | re.DOTALL,
)


def extract_reasoning(message: BaseMessage) -> Optional[str]:
    """
    Extract reasoning/thinking content from an LLM response.

    Follows the qwen-code pattern:
    1. Check additional_kwargs for 'reasoning_content' (DeepSeek API)
    2. Check additional_kwargs for 'reasoning' (OpenAI Responses API / vLLM)
    3. Fall back to parsing <think>/<thinking> tags from content

    Args:
        message: The LLM response message

    Returns:
        The reasoning text if found, None otherwise
    """
    if message is None:
        return None

    kwargs = getattr(message, "additional_kwargs", {}) or {}

    # Primary: structured reasoning field from API (DeepSeek native API)
    reasoning = kwargs.get("reasoning_content")
    if reasoning:
        return reasoning

    # Secondary: 'reasoning' field (OpenAI Responses API / vLLM)
    reasoning = kwargs.get("reasoning")
    if isinstance(reasoning, str) and reasoning:
        return reasoning
    if isinstance(reasoning, dict):
        # OpenAI Responses API wraps reasoning in a dict with 'content' list
        content_list = reasoning.get("content", [])
        if content_list:
            parts = [item.get("text", "") for item in content_list if isinstance(item, dict)]
            text = "".join(parts)
            if text:
                return text

    # Fallback: parse <think>/<thinking> tags from content
    content = getattr(message, "content", None)
    if content and isinstance(content, str):
        match = _THINK_EXTRACT_PATTERN.search(content)
        if match:
            return match.group(1).strip()

    return None
