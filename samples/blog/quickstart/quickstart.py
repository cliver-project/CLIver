"""The smallest useful CLIver agent: one model, the builtin tools, one call.

Run it with ``DEEPSEEK_API_KEY`` set:

    python samples/blog/quickstart/quickstart.py
"""

from __future__ import annotations

import asyncio

from cliver_core import AgentCore, ModelBindings, presets
from cliver_core.provider import CLIverResponse
from cliver_core.tools import discover_builtin_tools

# region: agent
PROMPT = "List the files in this directory."


async def run(agent: AgentCore | None = None) -> CLIverResponse:
    owned = agent is None
    agent = agent or AgentCore(
        ModelBindings(llm=presets.deepseek()),
        builtin_tools=discover_builtin_tools(),
    )
    try:
        return await agent.chat(PROMPT)
    finally:
        if owned:
            await agent.aclose()


async def main() -> None:
    response = await run()
    print(response.message.content)
    if response.usage_total is not None:
        print(f"tokens: {response.usage_total.total_tokens}")


if __name__ == "__main__":
    asyncio.run(main())
# endregion: agent
