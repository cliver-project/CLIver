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
