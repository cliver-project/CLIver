"""
Conversation Compressor — manages conversation history size within context window limits.

Two-tier compression:
  Tier 1 — prune stale tool results (cheap, no LLM call, triggers at 50% context)
  Tier 2 — LLM-generated summary of older turns (triggers at 70%+ context)
"""

import logging
from typing import List

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.messages.base import BaseMessage

from cliver.config import ModelConfig

logger = logging.getLogger(__name__)

# Sentinel prefix to identify compressed summary messages
SUMMARY_PREFIX = "[Conversation Summary]"

# Default context window when model doesn't specify one
DEFAULT_CONTEXT_WINDOW = 32768

# Well-known context window sizes by model name pattern
_CONTEXT_WINDOW_DEFAULTS = {
    "qwen": 131072,
    "deepseek": 131072,
    "gpt-4o": 128000,
    "gpt-4-turbo": 128000,
    "gpt-4.1": 1048576,
    "gpt-3.5": 16384,
    "claude": 200000,
    "gemini": 1048576,
    "llama": 131072,
    "mistral": 131072,
    "minimax": 1048576,
    "glm": 131072,
}

COMPRESSION_PROMPT = """Summarize the following conversation concisely. Preserve:
- Key decisions and conclusions reached
- Important facts, names, file paths, and code references mentioned
- The user's goals and current task state
- Any unresolved questions or pending actions

Do NOT include preamble like "Here is a summary". Just provide the summary directly.

Conversation:
{conversation}"""


def get_context_window(config: ModelConfig) -> int:
    """Get context window size for a model, with heuristic defaults."""
    if config.context_window:
        return config.context_window

    name = config.api_model_name.lower()
    for pattern, size in _CONTEXT_WINDOW_DEFAULTS.items():
        if pattern in name:
            return size

    return DEFAULT_CONTEXT_WINDOW


def estimate_tokens(messages: List[BaseMessage]) -> int:
    """Rough token estimate: ~4 chars per token."""
    total_chars = 0
    for m in messages:
        content = m.content
        if isinstance(content, list):
            # Multipart content (e.g., images + text)
            for part in content:
                if isinstance(part, dict) and "text" in part:
                    total_chars += len(part["text"])
                elif isinstance(part, str):
                    total_chars += len(part)
        elif isinstance(content, str):
            total_chars += len(content)
    return total_chars // 4


def estimate_tokens_str(text: str) -> int:
    """Rough token estimate for a plain string."""
    return len(text) // 4


def _is_summary_message(msg: BaseMessage) -> bool:
    """Check if a message is a conversation summary."""
    return isinstance(msg, HumanMessage) and isinstance(msg.content, str) and msg.content.startswith(SUMMARY_PREFIX)


TOOL_PRUNE_MIN_CHARS = 200


def prune_stale_tool_results(messages: List[BaseMessage]) -> List[BaseMessage]:
    """Replace large, already-processed ToolMessage contents with a short stub.

    A ToolMessage is "stale" when an AIMessage appears after it — the LLM has
    already seen and acted on the result.  ToolMessages from the current
    (unfinished) Re-Act iteration are left intact.

    Returns a new list; the original is not mutated.
    """
    if not messages:
        return []

    last_ai_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], AIMessage):
            last_ai_idx = i
            break

    result: list[BaseMessage] = []
    for i, msg in enumerate(messages):
        if (
            isinstance(msg, ToolMessage)
            and i < last_ai_idx
            and isinstance(msg.content, str)
            and len(msg.content) > TOOL_PRUNE_MIN_CHARS
        ):
            stub = (
                f"[Tool result pruned for context efficiency — "
                f"original was {len(msg.content):,} chars. "
                f"You already processed this result in your following response. "
                f"Refer to that response for the details.]"
            )
            result.append(ToolMessage(content=stub, tool_call_id=msg.tool_call_id))
        else:
            result.append(msg)

    return result


def _format_turns_for_compression(messages: List[BaseMessage]) -> str:
    """Format message list into readable text for the compression prompt."""
    lines = []
    for msg in messages:
        if _is_summary_message(msg):
            role = "Previous Summary"
        elif isinstance(msg, HumanMessage):
            role = "User"
        elif isinstance(msg, AIMessage):
            role = "Assistant"
        elif isinstance(msg, ToolMessage):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            lines.append(f"Tool result: {content[:200]}" if len(content) > 200 else f"Tool result: {content}")
            continue
        else:
            continue

        content = msg.content
        if isinstance(content, list):
            # Extract text parts only
            text_parts = []
            for part in content:
                if isinstance(part, dict) and "text" in part:
                    text_parts.append(part["text"])
                elif isinstance(part, str):
                    text_parts.append(part)
            content = "\n".join(text_parts)

        lines.append(f"{role}: {content}")
    return "\n\n".join(lines)


class ConversationCompressor:
    """Compresses conversation history to fit within model context windows.

    Stateless utility — operates on message lists passed in by the caller.
    """

    def __init__(
        self,
        context_window: int,
        threshold: float = 0.7,
        preserve_ratio: float = 0.3,
    ):
        """
        Args:
            context_window: Model's total context window in tokens.
            threshold: Fraction of context window that triggers compression (default 0.7).
            preserve_ratio: Fraction of conversation history to keep verbatim (default 0.3).
        """
        self.context_window = context_window
        self.threshold = threshold
        self.preserve_ratio = preserve_ratio

    def needs_compression(
        self,
        system_messages: List[BaseMessage],
        conversation_history: List[BaseMessage],
        new_input: str,
    ) -> bool:
        """Check if the combined messages exceed the compression threshold."""
        total = (
            estimate_tokens(system_messages) + estimate_tokens(conversation_history) + estimate_tokens_str(new_input)
        )
        budget = int(self.context_window * self.threshold)
        return total > budget

    async def compress(
        self,
        conversation_history: List[BaseMessage],
        llm_engine,
        force: bool = False,
    ) -> List[BaseMessage]:
        """Compress conversation history, keeping recent turns verbatim.

        Args:
            conversation_history: Full conversation history (HumanMessage/AIMessage pairs).
            llm_engine: LLM engine instance to use for generating the summary.
            force: If True, compress regardless of size.

        Returns:
            Compressed history: [SystemMessage(summary), ...recent_turns]
        """
        if not conversation_history:
            return []

        # Need at least 2 messages (1 exchange) to compress
        if len(conversation_history) < 4 and not force:
            return list(conversation_history)

        # Find split point: keep the newest preserve_ratio of turns
        total_turns = len(conversation_history)
        preserve_count = max(2, int(total_turns * self.preserve_ratio))
        # Round up to even number to keep complete user/assistant pairs
        if preserve_count % 2 != 0:
            preserve_count += 1
        preserve_count = min(preserve_count, total_turns)

        split_idx = total_turns - preserve_count
        if split_idx <= 0:
            return list(conversation_history)

        older_turns = conversation_history[:split_idx]
        recent_turns = conversation_history[split_idx:]

        # Try LLM-based compression
        try:
            summary = await self._generate_summary(older_turns, llm_engine)
            summary_msg = HumanMessage(content=f"{SUMMARY_PREFIX}\n{summary}")
            logger.info(
                f"Compressed {len(older_turns)} messages into summary "
                f"(~{estimate_tokens(older_turns)} → ~{estimate_tokens([summary_msg])} tokens)"
            )
            return [summary_msg] + list(recent_turns)
        except Exception as e:
            logger.warning(f"LLM compression failed, falling back to truncation: {e}")
            return self._truncate_fallback(conversation_history)

    async def _generate_summary(
        self,
        messages: List[BaseMessage],
        llm_engine,
    ) -> str:
        """Use the LLM to generate a conversation summary."""
        conversation_text = _format_turns_for_compression(messages)
        prompt = COMPRESSION_PROMPT.format(conversation=conversation_text)

        response = await llm_engine.infer(
            [HumanMessage(content=prompt)],
            tools=None,
        )

        summary = response.content
        if isinstance(summary, list):
            summary = "\n".join(
                part.get("text", str(part)) if isinstance(part, dict) else str(part) for part in summary
            )
        return summary.strip()

    def _truncate_fallback(self, conversation_history: List[BaseMessage]) -> List[BaseMessage]:
        """Simple truncation: keep newest turns within budget."""
        budget = int(self.context_window * self.threshold)
        result = []
        total = 0

        # Work backwards from newest to oldest
        for msg in reversed(conversation_history):
            msg_tokens = estimate_tokens([msg])
            if total + msg_tokens > budget:
                break
            result.insert(0, msg)
            total += msg_tokens

        # Prepend a note about truncation
        if len(result) < len(conversation_history):
            note = HumanMessage(content=f"{SUMMARY_PREFIX}\nEarlier conversation was truncated due to length.")
            result.insert(0, note)

        return result
