"""Stream a CLIver answer chunk by chunk.

Run it with ``DEEPSEEK_API_KEY`` set:

    python samples/blog/stream_demo/stream_demo.py
"""

from __future__ import annotations

import asyncio

from cliver_core import AgentCore, ModelBindings, presets

# region: stream
PROMPT = "Explain what an agent loop is in two sentences."


async def stream_answer(agent: AgentCore | None = None) -> str:
    owned = agent is None
    agent = agent or AgentCore(ModelBindings(llm=presets.deepseek()))
    parts: list[str] = []
    try:
        async for chunk in agent.stream(PROMPT):
            if chunk.content:
                parts.append(chunk.content)
                print(chunk.content, end="", flush=True)
    finally:
        if owned:
            await agent.aclose()
    print()
    return "".join(parts)


# endregion: stream


async def main() -> None:
    answer = await stream_answer()
    print(f"({len(answer)} characters)")


if __name__ == "__main__":
    asyncio.run(main())
