"""Give the agent a typed Python function as a tool.

Run it with ``DEEPSEEK_API_KEY`` set:

    python samples/blog/word_count_tool/word_count_tool.py
"""

from __future__ import annotations

import asyncio

from cliver_core import AgentCore, ModelBindings, presets, tool
from cliver_core.provider import CLIverResponse


# region: tool
@tool(name="word_count", description="Count the words in a text.")
def word_count(text: str) -> list[dict]:
    """Count whitespace-separated words in *text*."""
    return [{"text": str(len(text.split()))}]


# endregion: tool


# region: agent
PROMPT = "Use the word_count tool to count the words in: 'Agents are loops with tools'."


async def ask_word_count(agent: AgentCore | None = None) -> CLIverResponse:
    owned = agent is None
    agent = agent or AgentCore(
        ModelBindings(llm=presets.deepseek()),
        builtin_tools=[word_count],
    )
    try:
        return await agent.chat(PROMPT)
    finally:
        if owned:
            await agent.aclose()


# endregion: agent


async def main() -> None:
    response = await ask_word_count()
    print(response.message.content)


if __name__ == "__main__":
    asyncio.run(main())
