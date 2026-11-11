"""The permission gate and the path scopes, offline."""

from __future__ import annotations

from pathlib import Path

from _shared.testing import ScriptedProvider, tool_call_response
from cliver_core import (
    AgentCore,
    Decision,
    ExecutionPolicy,
    ModelBinding,
    ModelBindings,
    ToolCallContext,
)
from cliver_core.tools.read_file import read_file
from cliver_core.tools.write_file import write_file
from safe_files import PROMPT, build_agent, run, seed_workspace


def _tool_messages(provider: ScriptedProvider) -> list[str]:
    return [message.content for message in provider.requests[-1].messages if message.role == "tool"]


async def test_a_write_outside_the_scope_is_denied_before_it_runs(tmp_path: Path) -> None:
    workspace = seed_workspace(tmp_path)
    provider = ScriptedProvider(
        [
            tool_call_response("Read", {"file_path": "drafts/plan.md"}, call_id="call-1"),
            tool_call_response("Write", {"file_path": "summary.md", "content": "Ship the post."}, call_id="call-2"),
            "I could not write summary.md: writes are limited to the drafts/ directory.",
        ]
    )
    agent = build_agent(workspace, provider=provider)

    response = await run(agent)

    assert "limited to the drafts/" in (response.message.content or "")
    assert not (workspace / "summary.md").exists()
    read_result, write_result = _tool_messages(provider)
    assert "Ship the permission post" in read_result  # Read is SAFE: it ran without asking
    assert "Permission denied: writes are limited to the drafts/" in write_result


async def test_the_path_scope_denies_a_write_the_callback_allows(tmp_path: Path) -> None:
    workspace = seed_workspace(tmp_path)

    async def allow_all(_ctx: ToolCallContext) -> Decision:
        return Decision.allow()

    provider = ScriptedProvider(
        [
            tool_call_response("Write", {"file_path": "summary.md", "content": "Ship the post."}),
            "The write was refused by the path scope.",
        ]
    )
    agent = AgentCore(
        ModelBindings(llm=ModelBinding.llm(model="scripted", provider=provider)),
        builtin_tools=[read_file, write_file],
        permission_callback=allow_all,
        execution_policy=ExecutionPolicy(cwd=workspace, write_paths=[workspace / "drafts"]),
    )

    await agent.chat(PROMPT)

    assert "outside the allowed write paths" in _tool_messages(provider)[0]
    assert not (workspace / "summary.md").exists()


async def test_an_approved_write_inside_drafts_runs(tmp_path: Path) -> None:
    workspace = seed_workspace(tmp_path)
    provider = ScriptedProvider(
        [
            tool_call_response("Write", {"file_path": "drafts/note.md", "content": "Ship it."}),
            "Wrote drafts/note.md.",
        ]
    )
    agent = build_agent(workspace, provider=provider)

    await run(agent)

    assert (workspace / "drafts" / "note.md").read_text(encoding="utf-8") == "Ship it."
