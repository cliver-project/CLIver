"""Run one Agent Skill with run_skill().

The skill declares ``allowed-tools: Read``; the agent is built with Read and
Write, so the Write tool is not advertised during the skill run.

Run it with ``DEEPSEEK_API_KEY`` set:

    python samples/blog/skill_hello/skill_hello.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from cliver_core import AgentCore, ExecutionPolicy, ModelBinding, ModelBindings, SkillManager, presets
from cliver_core.provider import CLIverResponse, Provider
from cliver_core.tools.read_file import read_file
from cliver_core.tools.write_file import write_file

SKILLS_DIR = Path(__file__).parent / "skills"
PROMPT = "Greet me in Japanese."


# region: agent
def build_agent(*, provider: Provider | None = None) -> AgentCore:
    bindings = (
        ModelBindings(llm=presets.deepseek())
        if provider is None
        else ModelBindings(llm=ModelBinding.llm(model="scripted", provider=provider))
    )
    return AgentCore(
        bindings,
        builtin_tools=[read_file, write_file],
        skill_manager=SkillManager(sources=[SKILLS_DIR]),
        # An empty policy is a marker: run_skill() pins its cwd to the skill
        # directory, so "references/phrases.md" resolves next to SKILL.md.
        execution_policy=ExecutionPolicy(),
    )


async def run(agent: AgentCore) -> CLIverResponse:
    return await agent.run_skill("hello", PROMPT)


# endregion: agent


async def main() -> None:
    agent = build_agent()
    try:
        response = await run(agent)
    finally:
        await agent.aclose()
    print(response.message.content)


if __name__ == "__main__":
    asyncio.run(main())
