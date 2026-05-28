"""Conversation Compressor — manages history size within context window limits.

Two-tier compression:
  Tier 1 — prune stale tool results (cheap, no LLM call, triggers at 50% context)
  Tier 2 — LLM-generated summary of older turns (triggers at 70%+ context)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from cliver.config import ModelConfig
from cliver.messages import CLIverMessage

if TYPE_CHECKING:
    from cliver.agent import Agent

logger = logging.getLogger(__name__)

SUMMARY_PREFIX = "[Conversation Summary]"
DEFAULT_CONTEXT_WINDOW = 32768

_CONTEXT_WINDOW_DEFAULTS: dict[str, int] = {
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
    opts = config.options
    if opts and getattr(opts, "context_window", None) is not None:
        return opts.context_window

    name = config.api_model_name.lower()
    for pattern, size in _CONTEXT_WINDOW_DEFAULTS.items():
        if pattern in name:
            return size

    return DEFAULT_CONTEXT_WINDOW


def estimate_tokens(messages: list[CLIverMessage]) -> int:
    """Rough token estimate: ~4 chars per token."""
    total_chars = 0
    for m in messages:
        content = m.content
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and "text" in part:
                    total_chars += len(str(part["text"]))
                elif isinstance(part, str):
                    total_chars += len(part)
        elif isinstance(content, str):
            total_chars += len(content)
    return total_chars // 4


def estimate_tokens_str(text: str) -> int:
    return len(text) // 4


def _is_summary_message(msg: CLIverMessage) -> bool:
    return msg.role == "user" and isinstance(msg.content, str) and msg.content.startswith(SUMMARY_PREFIX)


TOOL_PRUNE_MIN_CHARS = 200


def prune_stale_tool_results(messages: list[CLIverMessage]) -> list[CLIverMessage]:
    """Replace large, already-processed ToolMessage contents with a short stub.

    A ToolMessage is "stale" when an assistant message appears after it.
    Returns a new list; the original is not mutated.
    """
    if not messages:
        return []

    last_assistant_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].role == "assistant":
            last_assistant_idx = i
            break

    result: list[CLIverMessage] = []
    for i, msg in enumerate(messages):
        if (
            msg.role == "tool"
            and i < last_assistant_idx
            and isinstance(msg.content, str)
            and len(msg.content) > TOOL_PRUNE_MIN_CHARS
        ):
            stub = (
                f"[Tool result pruned for context efficiency — "
                f"original was {len(msg.content):,} chars. "
                f"You already processed this result in your following response. "
                f"Refer to that response for the details.]"
            )
            result.append(CLIverMessage(role="tool", content=stub, tool_call_id=msg.tool_call_id))
        else:
            result.append(msg)

    return result


def _format_turns_for_compression(messages: list[CLIverMessage]) -> str:
    lines = []
    for msg in messages:
        if _is_summary_message(msg):
            role = "Previous Summary"
        elif msg.role == "user":
            role = "User"
        elif msg.role == "assistant":
            role = "Assistant"
        elif msg.role == "tool":
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            lines.append(f"Tool result: {content[:200]}" if len(content) > 200 else f"Tool result: {content}")
            continue
        else:
            continue

        content = msg.content
        if isinstance(content, list):
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
    """Compresses conversation history to fit within model context windows."""

    def __init__(
        self,
        context_window: int,
        threshold: float = 0.7,
        preserve_ratio: float = 0.3,
    ):
        self.context_window = context_window
        self.threshold = threshold
        self.preserve_ratio = preserve_ratio

    def needs_compression(
        self,
        system_messages: list[CLIverMessage],
        conversation_history: list[CLIverMessage],
        new_input: str,
    ) -> bool:
        total = (
            estimate_tokens(system_messages) + estimate_tokens(conversation_history) + estimate_tokens_str(new_input)
        )
        budget = int(self.context_window * self.threshold)
        return total > budget

    async def compress(
        self,
        conversation_history: list[CLIverMessage],
        agent: "Agent",
        force: bool = False,
    ) -> list[CLIverMessage]:
        """Compress conversation history, keeping recent turns verbatim.

        Args:
            conversation_history: Full conversation history.
            agent: Agent instance for the summary LLM call.
            force: If True, compress regardless of size.

        Returns:
            Compressed history with summary + recent turns.
        """
        if not conversation_history:
            return []

        if len(conversation_history) < 4 and not force:
            return list(conversation_history)

        total_turns = len(conversation_history)
        preserve_count = max(2, int(total_turns * self.preserve_ratio))
        if preserve_count % 2 != 0:
            preserve_count += 1
        preserve_count = min(preserve_count, total_turns)

        split_idx = total_turns - preserve_count
        if split_idx <= 0:
            return list(conversation_history)

        older_turns = conversation_history[:split_idx]
        recent_turns = conversation_history[split_idx:]

        try:
            summary = await self._generate_summary(older_turns, agent)
            summary_msg = CLIverMessage(role="user", content=f"{SUMMARY_PREFIX}\n{summary}")
            logger.info(
                "Compressed %d messages into summary (~%d → ~%d tokens)",
                len(older_turns),
                estimate_tokens(older_turns),
                estimate_tokens([summary_msg]),
            )
            return [summary_msg] + list(recent_turns)
        except Exception as e:
            logger.warning("LLM compression failed, falling back to truncation: %s", e)
            return self._truncate_fallback(conversation_history)

    async def _generate_summary(
        self,
        messages: list[CLIverMessage],
        agent: "Agent",
    ) -> str:
        conversation_text = _format_turns_for_compression(messages)
        prompt = COMPRESSION_PROMPT.format(conversation=conversation_text)

        response = await agent.chat(prompt=prompt, max_iterations=1)
        return (response.message.text or "").strip()

    def _truncate_fallback(self, conversation_history: list[CLIverMessage]) -> list[CLIverMessage]:
        budget = int(self.context_window * self.threshold)
        result: list[CLIverMessage] = []
        total = 0

        for msg in reversed(conversation_history):
            msg_tokens = estimate_tokens([msg])
            if total + msg_tokens > budget:
                break
            result.insert(0, msg)
            total += msg_tokens

        if len(result) < len(conversation_history):
            note = CLIverMessage(
                role="user",
                content=f"{SUMMARY_PREFIX}\nEarlier conversation was truncated due to length.",
            )
            result.insert(0, note)

        return result
