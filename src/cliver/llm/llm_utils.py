import logging
from typing import Optional
import re
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
    if (
        hasattr(response, "content")
        and response.content
        and '"tool_calls"' in str(response.content)
    ):
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
            logger.debug(
                f"Error parsing tool calls: {str(e)}", exc_info=True)
            return None
    return None

def is_thinking(content: str) -> bool:
    """Check if content is thinking."""
    # Look for opening <thinking> tag
    lower_content = content.lower()
    thinking_start = lower_content.find("<thinking>")
    thinking_end = lower_content.find("</thinking>")
    if thinking_start != -1:
        if thinking_end != -1 and thinking_end > thinking_start:
            # We have both start and end tags, and end comes after start
            return False
        else:
            # We have start tag but no end tag (or end tag comes before start)
            return True
    else:
        return False

def is_thinking_content(content: str) -> bool:
    """Check if content indicates thinking mode."""
    thinking_patterns = [
        r'<thinking>.*?</thinking>',      # <thinking>...</thinking>
    ]

    for pattern in thinking_patterns:
        if re.search(pattern, content, re.IGNORECASE | re.DOTALL):
            return True
    return False

def remove_thinking_sections(content: str) -> str:
    """Remove thinking sections from content to find tool calls in the remaining text."""
    # Remove thinking sections
    thinking_patterns = [
        r'<thinking>.*?</thinking>',
    ]

    clean_content = content
    for pattern in thinking_patterns:
        clean_content = re.sub(pattern, '', clean_content, flags=re.IGNORECASE | re.DOTALL)

    return clean_content
