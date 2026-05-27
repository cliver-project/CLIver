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
# format used by AgentCore.  This is intentionally a standalone function
# — not an overridable engine method — so normalization can never be lost.


def normalize_tool_calls(tool_calls: list[dict]) -> list[dict] | None:
    """Normalize raw parsed tool calls into the execution format.

    Args:
        tool_calls: Raw tool calls from engine.parse_tool_calls.
            Expected: [{"name": "...", "args": {...}, "id": "..."}]

    Returns:
        Normalized tool calls for AgentCore, or None if all invalid.
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
        if isinstance(value, str) and len(value) > 1 and value[0] in ("[", "{"):
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

    # 2b. Anthropic-style tool_use content blocks (used by MiniMax and other
    #     providers even when configured as openai protocol)
    if isinstance(content, list):
        tool_calls = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                tool_calls.append(
                    {
                        "name": block.get("name", ""),
                        "args": block.get("input", {}),
                        "id": block.get("id", ""),
                    }
                )
        if tool_calls:
            return tool_calls

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

        # Find the JSON object containing tool_calls
        idx = cleaned.find('"tool_calls"')
        if idx == -1:
            return None

        # Walk backwards to find the opening brace
        brace_start = cleaned.rfind("{", 0, idx)
        if brace_start == -1:
            return None

        # Walk forward to find the matching closing brace
        depth = 0
        brace_end = -1
        for i in range(brace_start, len(cleaned)):
            if cleaned[i] == "{":
                depth += 1
            elif cleaned[i] == "}":
                depth -= 1
                if depth == 0:
                    brace_end = i + 1
                    break

        # Parse the JSON block
        json_str = cleaned[brace_start:brace_end] if brace_end != -1 else cleaned[brace_start:]
        parsed = json_repair.loads(json_str)
        if isinstance(parsed, dict) and "tool_calls" in parsed:
            tool_calls = parsed["tool_calls"]
            if isinstance(tool_calls, list):
                # Strip the tool_calls JSON block from the response content
                _strip_tool_calls_json(response, content_str, brace_start, brace_end, cleaned)
                return tool_calls

    except Exception as e:
        logger.debug(f"Error parsing tool calls from content: {e}")

    return None


def _strip_tool_calls_json(
    response: BaseMessage,
    content_str: str,
    brace_start: int,
    brace_end: int,
    cleaned: str,
) -> None:
    """Remove the tool_calls JSON block from the response content."""
    try:
        if brace_end == -1:
            return
        # Map position in cleaned text back to original content
        json_block = cleaned[brace_start:brace_end]
        idx_in_original = content_str.find(json_block)
        if idx_in_original == -1:
            return
        before = content_str[:idx_in_original]
        after = content_str[idx_in_original + len(json_block) :]
        # Clean up leading/trailing whitespace and newlines around the block
        new_content = (before.rstrip() + "\n" + after.lstrip()).strip()
        if hasattr(response, "content") and isinstance(response.content, str):
            # noinspection PyPropertyAccess
            response.content = new_content
    except Exception as e:
        logger.debug(f"Error stripping tool_calls JSON from content: {e}")


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


def strip_tool_calls_from_text(text: str) -> str:
    """Remove embedded ``{"tool_calls": [...]}`` JSON block from text content.

    Used to clean streamed LLM output when a model (e.g. MiniMax) emits
    tool calls as inline JSON rather than structured tool_calls.
    """
    if not text or '"tool_calls"' not in text:
        return text

    try:
        idx = text.find('"tool_calls"')
        if idx == -1:
            return text

        brace_start = text.rfind("{", 0, idx)
        if brace_start == -1:
            return text

        # Walk forward to find the matching closing brace
        depth = 0
        brace_end = -1
        for i in range(brace_start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    brace_end = i + 1
                    break

        if brace_end == -1:
            return text

        before = text[:brace_start]
        after = text[brace_end:]
        return (before.rstrip() + "\n" + after.lstrip()).strip()
    except Exception as e:
        logger.debug("Error stripping tool_calls from text: %s", e)
        return text
