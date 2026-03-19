import json
import logging
import re
import uuid
from typing import Optional

import json_repair
from langchain_core.messages.base import BaseMessage

logger = logging.getLogger(__name__)


# ── Tool call normalization ──────────────────────────────────────────────
#
# Converts raw parsed tool calls (from the engine layer) into the execution
# format used by TaskExecutor.  This is intentionally a standalone function
# — not an overridable engine method — so normalization can never be lost.


def normalize_tool_calls(tool_calls: list[dict]) -> list[dict] | None:
    """Normalize raw parsed tool calls into the execution format.

    Args:
        tool_calls: Raw tool calls from engine.parse_tool_calls.
            Expected: [{"name": "...", "args": {...}, "id": "..."}]

    Returns:
        Normalized tool calls for TaskExecutor, or None if all invalid.
        Format: [{"tool_name": "...", "mcp_server": "...", "args": {...}, "tool_call_id": "..."}]
    """
    if not tool_calls:
        return None

    normalized = []
    for tool_call in tool_calls:
        tool_name = tool_call.get("name")
        if not tool_name or not isinstance(tool_name, str):
            logger.warning("Skipping tool call with missing/invalid name: %s", tool_call)
            continue

        mcp_server = ""
        if "#" in tool_name:
            parts = tool_name.split("#", 1)
            mcp_server = parts[0]
            tool_name = parts[1]

        normalized.append(
            {
                "tool_call_id": tool_call.get("id") or str(uuid.uuid4()),
                "tool_name": tool_name,
                "mcp_server": mcp_server,
                "args": _normalize_tool_args(tool_call.get("args")),
            }
        )

    return normalized if normalized else None


def _normalize_tool_args(args) -> dict:
    """Normalize tool call arguments into a clean dict.

    Handles the many formats LLMs produce:
    - dict → return with JSON-string values coerced
    - JSON string → parse to dict
    - None → empty dict
    - other → empty dict with warning
    """
    if args is None:
        return {}
    if isinstance(args, str):
        args = args.strip()
        if not args:
            return {}
        try:
            parsed = json.loads(args)
            if isinstance(parsed, dict):
                return _coerce_json_string_values(parsed)
            logger.warning("Tool args parsed to non-dict type: %s", type(parsed))
            return {}
        except (json.JSONDecodeError, ValueError):
            logger.warning("Could not parse tool args string: %.200s", args)
            return {}
    if isinstance(args, dict):
        return _coerce_json_string_values(args)
    logger.warning("Unexpected tool args type %s, using empty dict", type(args))
    return {}


def _coerce_json_string_values(args: dict) -> dict:
    """Coerce JSON-encoded string values in tool args to native objects.

    Some LLMs serialize list/dict arguments as JSON strings instead of
    actual JSON objects. Detects values starting with [ or { and parses them.
    """
    coerced = {}
    for key, value in args.items():
        if isinstance(value, str) and value and value[0] in ("[", "{"):
            try:
                coerced[key] = json.loads(value)
            except (json.JSONDecodeError, ValueError):
                coerced[key] = value
        else:
            coerced[key] = value
    return coerced


# ── Tool call parsing ────────────────────────────────────────────────────


def parse_tool_calls_from_content(response: BaseMessage) -> Optional[list[dict]]:
    """Parse tool calls from an LLM response.

    Handles multiple formats:
    1. Structured tool_calls attribute (standard OpenAI-compatible)
    2. Content dict with "tool_calls" key
    3. JSON embedded in content text (fallback for models without structured tool calling)
       — strips <think>/<thinking> blocks first to avoid false matches
    """
    if response is None:
        return None

    # 1. Structured tool_calls (the standard path)
    if hasattr(response, "tool_calls") and response.tool_calls:
        return response.tool_calls

    content = getattr(response, "content", None)
    if not content:
        return None

    # 2. Content is already a dict with tool_calls
    if isinstance(content, dict) and "tool_calls" in content:
        return content.get("tool_calls", [])

    # 3. Parse tool_calls from text content (fallback)
    content_str = str(content)
    if '"tool_calls"' not in content_str:
        return None

    try:
        # Strip thinking blocks first — LLMs may mention tool_calls
        # inside <think> tags as reasoning, not as actual calls
        cleaned = remove_thinking_sections(content_str)
        if '"tool_calls"' not in cleaned:
            return None

        # Use json_repair to extract — handles nested brackets, trailing commas,
        # and other malformed JSON that LLMs commonly produce
        # First try: find the outermost JSON object containing tool_calls
        # Use bracket-matching to find the complete JSON object
        idx = cleaned.find('"tool_calls"')
        if idx == -1:
            return None

        # Walk backwards to find the opening brace
        brace_start = cleaned.rfind("{", 0, idx)
        if brace_start == -1:
            return None

        # Use json_repair on the substring from the opening brace
        parsed = json_repair.loads(cleaned[brace_start:])
        if isinstance(parsed, dict) and "tool_calls" in parsed:
            tool_calls = parsed["tool_calls"]
            if isinstance(tool_calls, list):
                return tool_calls

    except Exception as e:
        logger.debug(f"Error parsing tool calls from content: {e}")

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
