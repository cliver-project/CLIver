"""Tests for Phase 2: Planner — system prompt, todo_read, plan context injection."""

import pytest
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from cliver.llm.llm import _PLAN_CONTEXT_PREFIX, _inject_plan_context, _is_plan_context_message
from cliver.tools.todo_read import TodoReadTool
from cliver.tools.todo_write import TodoWriteTool, format_todo_summary


@pytest.fixture(autouse=True)
def reset_todos():
    """Reset the module-level todo list before and after each test."""
    import sys

    # Must use sys.modules to get the actual module, not the re-exported tool instance
    mod = sys.modules.get("cliver.tools.todo_write")
    if mod is None:
        import cliver.tools.todo_write  # noqa: F401

        mod = sys.modules["cliver.tools.todo_write"]
    mod._current_todos.clear()
    yield
    mod._current_todos.clear()


# ---------------------------------------------------------------------------
# format_todo_summary
# ---------------------------------------------------------------------------


class TestFormatTodoSummary:
    def test_empty_list(self):
        assert format_todo_summary([]) == "No active plan."

    def test_formats_items(self):
        todos = [
            {"id": "1", "content": "Research", "status": "completed"},
            {"id": "2", "content": "Implement", "status": "in_progress"},
            {"id": "3", "content": "Test", "status": "pending"},
        ]
        result = format_todo_summary(todos)
        assert "[x] (1) Research" in result
        assert "[~] (2) Implement" in result
        assert "[ ] (3) Test" in result
        assert "1/3 completed" in result
        assert "1 in progress" in result
        assert "1 pending" in result


# ---------------------------------------------------------------------------
# todo_read tool
# ---------------------------------------------------------------------------


class TestTodoRead:
    def test_read_empty(self):
        tool = TodoReadTool()
        result = tool._run()
        assert "No active plan" in result

    def test_read_after_write(self):
        write_tool = TodoWriteTool()
        write_tool._run(
            todos=[
                {"id": "1", "content": "Step one", "status": "pending"},
                {"id": "2", "content": "Step two", "status": "pending"},
            ]
        )

        read_tool = TodoReadTool()
        result = read_tool._run()
        assert "Step one" in result
        assert "Step two" in result
        assert "0/2 completed" in result

    def test_read_reflects_updates(self):
        write_tool = TodoWriteTool()
        write_tool._run(
            todos=[
                {"id": "1", "content": "Step one", "status": "completed"},
                {"id": "2", "content": "Step two", "status": "in_progress"},
            ]
        )

        read_tool = TodoReadTool()
        result = read_tool._run()
        assert "[x] (1)" in result
        assert "[~] (2)" in result
        assert "1/2 completed" in result


# ---------------------------------------------------------------------------
# _inject_plan_context
# ---------------------------------------------------------------------------


class TestInjectPlanContext:
    def test_no_injection_when_no_plan(self):
        messages = [HumanMessage(content="hello")]
        _inject_plan_context(messages)
        assert len(messages) == 1  # no plan context added

    def test_injects_plan_when_todos_exist(self):
        write_tool = TodoWriteTool()
        write_tool._run(
            todos=[
                {"id": "1", "content": "Do thing", "status": "pending"},
            ]
        )

        messages = [HumanMessage(content="hello")]
        _inject_plan_context(messages)
        assert len(messages) == 2
        plan_msg = messages[-1]
        assert isinstance(plan_msg, SystemMessage)
        assert _PLAN_CONTEXT_PREFIX in plan_msg.content
        assert "Do thing" in plan_msg.content

    def test_replaces_stale_plan_context(self):
        """Should not stack multiple plan context messages."""
        write_tool = TodoWriteTool()
        write_tool._run(
            todos=[
                {"id": "1", "content": "First", "status": "pending"},
            ]
        )

        messages = [HumanMessage(content="hello")]
        _inject_plan_context(messages)
        assert len(messages) == 2

        # Update the plan
        write_tool._run(
            todos=[
                {"id": "1", "content": "First", "status": "completed"},
                {"id": "2", "content": "Second", "status": "pending"},
            ]
        )

        # Inject again — should replace, not stack
        _inject_plan_context(messages)
        plan_messages = [m for m in messages if _is_plan_context_message(m)]
        assert len(plan_messages) == 1
        assert "Second" in plan_messages[0].content
        assert "First" in plan_messages[0].content

    def test_completion_hint_when_all_done(self):
        write_tool = TodoWriteTool()
        write_tool._run(
            todos=[
                {"id": "1", "content": "Only task", "status": "completed"},
            ]
        )

        messages = [HumanMessage(content="do it")]
        _inject_plan_context(messages)
        plan_msg = messages[-1]
        assert "All planned tasks are completed" in plan_msg.content
        assert "final summary" in plan_msg.content

    def test_no_completion_hint_when_work_remains(self):
        write_tool = TodoWriteTool()
        write_tool._run(
            todos=[
                {"id": "1", "content": "Done", "status": "completed"},
                {"id": "2", "content": "Not done", "status": "pending"},
            ]
        )

        messages = [HumanMessage(content="do it")]
        _inject_plan_context(messages)
        plan_msg = messages[-1]
        assert "All planned tasks are completed" not in plan_msg.content

    def test_preserves_other_messages(self):
        """Plan injection should not remove non-plan system messages."""
        write_tool = TodoWriteTool()
        write_tool._run(
            todos=[
                {"id": "1", "content": "Task", "status": "pending"},
            ]
        )

        messages = [
            SystemMessage(content="You are a helpful agent."),
            HumanMessage(content="do something"),
            ToolMessage(content="result", tool_call_id="123"),
        ]
        _inject_plan_context(messages)
        # Original 3 messages + 1 plan context
        assert len(messages) == 4
        assert messages[0].content == "You are a helpful agent."


# ---------------------------------------------------------------------------
# _is_plan_context_message
# ---------------------------------------------------------------------------


class TestIsPlanContextMessage:
    def test_identifies_plan_message(self):
        msg = SystemMessage(content=f"{_PLAN_CONTEXT_PREFIX}\nSome plan")
        assert _is_plan_context_message(msg) is True

    def test_rejects_normal_system_message(self):
        msg = SystemMessage(content="You are a helpful assistant.")
        assert _is_plan_context_message(msg) is False

    def test_rejects_human_message(self):
        msg = HumanMessage(content=f"{_PLAN_CONTEXT_PREFIX}\nfake")
        assert _is_plan_context_message(msg) is False


# ---------------------------------------------------------------------------
# System prompt includes planning guidance
# ---------------------------------------------------------------------------


class TestSystemPromptPlanning:
    def test_system_prompt_contains_planning_section(self):
        from cliver.llm.base import LLMInferenceEngine

        prompt = LLMInferenceEngine._section_interaction_guidelines()
        assert "## Planning" in prompt
        assert "Simple" in prompt
        assert "Medium" in prompt
        assert "TodoWrite" in prompt
        assert "TodoRead" in prompt

    def test_system_prompt_contains_skill_section(self):
        from cliver.llm.base import LLMInferenceEngine

        prompt = LLMInferenceEngine._section_interaction_guidelines()
        assert "## Skills" in prompt
        assert "skill" in prompt
