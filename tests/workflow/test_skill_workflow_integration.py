"""
Tests for Skill ↔ Workflow integration (Phase 6).

Verifies that LLM workflow steps can reference skills by name
and that skill content is injected into the LLM call.
"""

import textwrap
from unittest.mock import AsyncMock, Mock

import pytest

from cliver.llm import TaskExecutor
from cliver.workflow.steps.llm_step import LLMStepExecutor
from cliver.workflow.workflow_models import ExecutionContext, LLMStep


@pytest.fixture
def mock_task_executor():
    return Mock(spec=TaskExecutor)


@pytest.fixture
def skills_dir(tmp_path):
    """Create test skills for workflow integration."""
    # web-search skill
    ws = tmp_path / "skills" / "web-search"
    ws.mkdir(parents=True)
    (ws / "SKILL.md").write_text(textwrap.dedent("""\
        ---
        name: web-search
        description: Search the web for information.
        ---

        # Web Search

        Use the web_search tool to find information online.
    """))

    # code-review skill
    cr = tmp_path / "skills" / "code-review"
    cr.mkdir(parents=True)
    (cr / "SKILL.md").write_text(textwrap.dedent("""\
        ---
        name: code-review
        description: Review code for quality and security.
        ---

        # Code Review

        Focus on security vulnerabilities and best practices.
    """))

    return tmp_path / "skills"


# ---------------------------------------------------------------------------
# LLMStep model
# ---------------------------------------------------------------------------


class TestLLMStepSkillsField:
    def test_skills_field_is_optional(self):
        step = LLMStep(id="s", name="S", prompt="hello")
        assert step.skills is None

    def test_skills_field_accepts_list(self):
        step = LLMStep(id="s", name="S", prompt="hello", skills=["web-search", "code-review"])
        assert step.skills == ["web-search", "code-review"]

    def test_model_per_step(self):
        step = LLMStep(id="s", name="S", prompt="hello", model="qwen")
        assert step.model == "qwen"


# ---------------------------------------------------------------------------
# Skill appender
# ---------------------------------------------------------------------------


class TestSkillAppender:
    def test_no_skills_returns_none(self, mock_task_executor):
        step = LLMStep(id="s", name="S", prompt="hello")
        executor = LLMStepExecutor(step, mock_task_executor)
        assert executor._build_skill_appender() is None

    def test_empty_skills_returns_none(self, mock_task_executor):
        step = LLMStep(id="s", name="S", prompt="hello", skills=[])
        executor = LLMStepExecutor(step, mock_task_executor)
        assert executor._build_skill_appender() is None

    def test_skill_content_injected(self, mock_task_executor, skills_dir, monkeypatch):
        monkeypatch.setattr("cliver.skill_manager.get_config_dir", lambda: skills_dir.parent / "no-global")
        monkeypatch.setattr("cliver.skill_manager.Path.cwd", lambda: skills_dir.parent)
        # Create .cliver/skills structure
        import shutil
        local_skills = skills_dir.parent / ".cliver" / "skills"
        if not local_skills.exists():
            local_skills.mkdir(parents=True)
            for child in skills_dir.iterdir():
                if child.is_dir():
                    shutil.copytree(child, local_skills / child.name)

        step = LLMStep(id="s", name="S", prompt="hello", skills=["web-search"])
        executor = LLMStepExecutor(step, mock_task_executor)
        appender = executor._build_skill_appender()

        assert appender is not None
        content = appender()
        assert "Activated Skills" in content
        assert "Web Search" in content
        assert "web_search" in content

    def test_multiple_skills_combined(self, mock_task_executor, skills_dir, monkeypatch):
        monkeypatch.setattr("cliver.skill_manager.get_config_dir", lambda: skills_dir.parent / "no-global")
        monkeypatch.setattr("cliver.skill_manager.Path.cwd", lambda: skills_dir.parent)
        import shutil
        local_skills = skills_dir.parent / ".cliver" / "skills"
        if not local_skills.exists():
            local_skills.mkdir(parents=True)
            for child in skills_dir.iterdir():
                if child.is_dir():
                    shutil.copytree(child, local_skills / child.name)

        step = LLMStep(id="s", name="S", prompt="hello", skills=["web-search", "code-review"])
        executor = LLMStepExecutor(step, mock_task_executor)
        appender = executor._build_skill_appender()

        content = appender()
        assert "Web Search" in content
        assert "Code Review" in content

    def test_nonexistent_skill_still_works(self, mock_task_executor, tmp_path, monkeypatch):
        """Nonexistent skills produce a 'not found' message but don't crash."""
        monkeypatch.setattr("cliver.skill_manager.get_config_dir", lambda: tmp_path / "empty")
        monkeypatch.setattr("cliver.skill_manager.Path.cwd", lambda: tmp_path / "empty")

        step = LLMStep(id="s", name="S", prompt="hello", skills=["nonexistent"])
        executor = LLMStepExecutor(step, mock_task_executor)
        appender = executor._build_skill_appender()

        # Should still produce an appender (with "not found" message)
        assert appender is not None
        content = appender()
        assert "not found" in content


# ---------------------------------------------------------------------------
# Integration: skill appender passed to TaskExecutor
# ---------------------------------------------------------------------------


class TestSkillPassedToExecutor:
    @pytest.mark.asyncio
    async def test_system_message_appender_passed(self, mock_task_executor, skills_dir, monkeypatch):
        """Verify the skill appender is passed as system_message_appender to process_user_input."""
        monkeypatch.setattr("cliver.skill_manager.get_config_dir", lambda: skills_dir.parent / "no-global")
        monkeypatch.setattr("cliver.skill_manager.Path.cwd", lambda: skills_dir.parent)
        import shutil
        local_skills = skills_dir.parent / ".cliver" / "skills"
        if not local_skills.exists():
            local_skills.mkdir(parents=True)
            for child in skills_dir.iterdir():
                if child.is_dir():
                    shutil.copytree(child, local_skills / child.name)

        step = LLMStep(id="s", name="S", prompt="research AI", skills=["web-search"])
        executor = LLMStepExecutor(step, mock_task_executor)

        context = ExecutionContext(workflow_name="test", inputs={})

        mock_response = Mock()
        mock_response.content = "AI research results"
        mock_task_executor.process_user_input = AsyncMock(return_value=mock_response)

        await executor.execute(context)

        # Verify system_message_appender was passed
        call_kwargs = mock_task_executor.process_user_input.call_args.kwargs
        assert call_kwargs["system_message_appender"] is not None
        # Call the appender to verify it has skill content
        appender_result = call_kwargs["system_message_appender"]()
        assert "Web Search" in appender_result

    @pytest.mark.asyncio
    async def test_no_skills_passes_none_appender(self, mock_task_executor):
        """Without skills, system_message_appender should be None."""
        step = LLMStep(id="s", name="S", prompt="hello")
        executor = LLMStepExecutor(step, mock_task_executor)

        context = ExecutionContext(workflow_name="test", inputs={})

        mock_response = Mock()
        mock_response.content = "hi"
        mock_task_executor.process_user_input = AsyncMock(return_value=mock_response)

        await executor.execute(context)

        call_kwargs = mock_task_executor.process_user_input.call_args.kwargs
        assert call_kwargs["system_message_appender"] is None
