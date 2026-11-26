"""A tiny agent with one mutating tool, built for offline tests."""

from __future__ import annotations

from pathlib import Path

from cliver_core import (
    AgentCore,
    Decision,
    EventHandler,
    ModelBinding,
    ModelBindings,
    ToolCallContext,
)
from cliver_core.provider import Provider
from cliver_core.tool import tool


# region: tool
@tool(name="SaveNote", description="Save a note to notes/<title>.md.")
def save_note(title: str, body: str) -> list[dict]:
    """Write one note file under notes/ in the current working directory."""
    path = Path("notes") / f"{title}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return [{"text": f"Saved {path.as_posix()}."}]


# endregion: tool


# region: agent
async def deny_notes(_context: ToolCallContext) -> Decision:
    return Decision.deny("notes are disabled in this test")


def build_agent(provider: Provider, *, on_event: EventHandler | None = None) -> AgentCore:
    return AgentCore(
        ModelBindings(llm=ModelBinding.llm(model="scripted", provider=provider)),
        builtin_tools=[save_note],
        permission_callback=deny_notes,
        on_event=on_event,
    )


# endregion: agent
