"""Skill discovery is filesystem I/O and must not run on the event loop.

``SkillManager`` keeps its synchronous API for direct callers, but AgentCore
preloads the cache with ``apreload()`` (a worker thread) before its first
synchronous skill access, so a slow source does not stall the loop and the
cached lookup afterwards is cheap.
"""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path

import pytest

import cliver_core.skill as skill_module
from cliver_core.agent import AgentCore
from cliver_core.bindings import ModelBinding, ModelBindings
from cliver_core.skill import SkillManager

FIXTURES = Path(__file__).parent / "fixtures" / "skills"


def test_apreload_discovers_off_the_event_loop(monkeypatch) -> None:
    threads: list[threading.Thread] = []
    real_discover = skill_module._discover_source

    def slow_discover(source_path, label):
        threads.append(threading.current_thread())
        time.sleep(0.1)
        return real_discover(source_path, label)

    monkeypatch.setattr(skill_module, "_discover_source", slow_discover)
    manager = SkillManager(sources=[FIXTURES])

    async def main() -> tuple[threading.Thread, int]:
        loop_thread = threading.current_thread()
        ticks = 0
        stop = asyncio.Event()

        async def ticker() -> None:
            nonlocal ticks
            while not stop.is_set():
                ticks += 1
                await asyncio.sleep(0.01)

        task = asyncio.ensure_future(ticker())
        await manager.apreload()
        stop.set()
        await task
        return loop_thread, ticks

    loop_thread, ticks = asyncio.run(main())

    assert threads and threads[0] is not loop_thread
    assert ticks >= 3  # the loop kept running while discovery blocked
    assert manager.list_skills()  # sync access uses the warmed cache
    assert len(threads) == 1


def _agent(manager: SkillManager) -> AgentCore:
    return AgentCore(ModelBindings(llm=ModelBinding.llm(model="mock-model")), skill_manager=manager)


async def _fake_run_loop(*_args, **_kwargs) -> None:
    return None


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["chat", "run_skill"])
async def test_entry_points_preload_skills_off_the_loop(monkeypatch, entry: str) -> None:
    threads: list[threading.Thread] = []
    real_discover = skill_module._discover_source

    def recording_discover(source_path, label):
        threads.append(threading.current_thread())
        return real_discover(source_path, label)

    monkeypatch.setattr(skill_module, "_discover_source", recording_discover)
    agent = _agent(SkillManager(sources=[FIXTURES]))
    agent._run_loop = _fake_run_loop  # type: ignore[method-assign]

    if entry == "chat":
        await agent.chat("hi")
    else:
        await agent.run_skill("test-code-review", "hi")

    assert threads and threads[0] is not threading.current_thread()
    assert len(threads) == 1
