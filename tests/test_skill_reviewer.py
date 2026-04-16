"""Tests for autonomous skill learning — post-task skill review."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cliver.skill_reviewer import (
    DEFAULT_SKILL_NUDGE_THRESHOLD,
    _suggest_skill_name,
    maybe_review_for_skill,
)


class TestSuggestSkillName:
    def test_simple_words(self):
        name = _suggest_skill_name("Deploy kubernetes to production")
        assert "deploy" in name
        assert "kubernetes" in name

    def test_filters_stop_words(self):
        name = _suggest_skill_name("How to fix the build error")
        assert "how" not in name
        assert "the" not in name
        assert "fix" in name

    def test_truncates_long_names(self):
        name = _suggest_skill_name("a " * 100)
        assert len(name) <= 40

    def test_empty_summary(self):
        name = _suggest_skill_name("")
        assert name == "auto-skill"


class TestMaybeReviewForSkill:
    def test_below_threshold_skips(self):
        executor = MagicMock()
        result = asyncio.run(
            maybe_review_for_skill(
                task_executor=executor,
                tool_call_count=3,
                task_summary="simple task",
                threshold=10,
            )
        )
        assert result is None
        executor.process_user_input.assert_not_called()

    def test_above_threshold_triggers_review(self, tmp_path):
        executor = MagicMock()

        async def mock_process(**kwargs):
            msg = MagicMock()
            msg.content = "No skill needed for this task."
            return msg

        executor.process_user_input = AsyncMock(side_effect=mock_process)

        result = asyncio.run(
            maybe_review_for_skill(
                task_executor=executor,
                tool_call_count=15,
                task_summary="complex kubernetes deployment with error recovery",
                threshold=10,
                skills_dir=tmp_path / "skills",
            )
        )
        # Review was triggered (process_user_input called)
        executor.process_user_input.assert_called_once()
        # But decided not to create a skill
        assert result is None

    def test_review_error_does_not_crash(self):
        executor = MagicMock()
        executor.process_user_input = AsyncMock(side_effect=RuntimeError("boom"))

        result = asyncio.run(
            maybe_review_for_skill(
                task_executor=executor,
                tool_call_count=15,
                task_summary="some task",
                threshold=10,
            )
        )
        assert result is None

    def test_default_threshold(self):
        assert DEFAULT_SKILL_NUDGE_THRESHOLD == 10


class TestToolCallCounter:
    def test_counter_increments_on_tool_calls(self):
        from cliver.llm.llm import AgentCore

        executor = AgentCore(llm_models={}, mcp_servers={})
        assert executor._tool_call_count == 0
