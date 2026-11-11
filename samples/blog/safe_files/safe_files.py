"""Gate filesystem writes behind a permission callback and path scopes.

Run it with ``DEEPSEEK_API_KEY`` set:

    python samples/blog/safe_files/safe_files.py
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from cliver_core import (
    AgentCore,
    Decision,
    ExecutionPolicy,
    ModelBinding,
    ModelBindings,
    PermissionCallback,
    ToolCallContext,
    presets,
)
from cliver_core.provider import CLIverResponse, Provider
from cliver_core.tools.read_file import read_file
from cliver_core.tools.write_file import write_file

PROMPT = (
    "Read drafts/plan.md, then write a one-sentence summary to summary.md. "
    "If the write is refused, tell me the exact reason."
)


def seed_workspace(workspace: Path) -> Path:
    """Create the demo workspace: one draft file inside ``drafts/``."""
    drafts = workspace / "drafts"
    drafts.mkdir(parents=True, exist_ok=True)
    (drafts / "plan.md").write_text("Ship the permission post on Wednesday.\n", encoding="utf-8")
    return workspace


def make_workspace() -> Path:
    return seed_workspace(Path(tempfile.mkdtemp(prefix="cliver-safe-files-")))


# region: policy
def make_policy(workspace: Path) -> tuple[PermissionCallback, ExecutionPolicy]:
    async def approve(ctx: ToolCallContext) -> Decision:
        if ctx.tool_name != "Write":
            return Decision.allow()
        target = (workspace / str(ctx.args.get("file_path", ""))).resolve()
        if target.is_relative_to(workspace / "drafts"):
            return Decision.allow()
        return Decision.deny("writes are limited to the drafts/ directory")

    policy = ExecutionPolicy(
        cwd=workspace,
        read_paths=[workspace],
        write_paths=[workspace / "drafts"],
    )
    return approve, policy


# endregion: policy


# region: agent
def build_agent(workspace: Path, *, provider: Provider | None = None) -> AgentCore:
    approve, policy = make_policy(workspace)
    bindings = (
        ModelBindings(llm=presets.deepseek())
        if provider is None
        else ModelBindings(llm=ModelBinding.llm(model="scripted", provider=provider))
    )
    return AgentCore(
        bindings,
        builtin_tools=[read_file, write_file],
        permission_callback=approve,
        execution_policy=policy,
    )


async def run(agent: AgentCore) -> CLIverResponse:
    return await agent.chat(PROMPT)


# endregion: agent


async def main() -> None:
    with tempfile.TemporaryDirectory(prefix="cliver-safe-files-") as tmp:
        workspace = seed_workspace(Path(tmp))
        agent = build_agent(workspace)
        try:
            response = await run(agent)
        finally:
            await agent.aclose()
        print(response.message.content)
        files = sorted(p.relative_to(workspace).as_posix() for p in workspace.rglob("*") if p.is_file())
        print(f"\nfiles: {files}")


if __name__ == "__main__":
    asyncio.run(main())
