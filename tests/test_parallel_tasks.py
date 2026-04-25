"""Tests for the parallel_tasks builtin tool."""

from unittest.mock import AsyncMock, MagicMock, patch

from cliver.tools.parallel_tasks import ParallelTasksTool


class TestParallelTasksTool:
    def test_tool_has_correct_name(self):
        tool = ParallelTasksTool()
        assert tool.name == "Parallel"

    def test_tool_has_description(self):
        tool = ParallelTasksTool()
        assert "parallel" in tool.description.lower()

    @patch("cliver.tools.parallel_tasks.get_agent_core")
    def test_no_executor_returns_error(self, mock_get):
        mock_get.return_value = None
        tool = ParallelTasksTool()
        result = tool._run(tasks=["hello", "world"])
        assert "not available" in result.lower()

    @patch("cliver.tools.parallel_tasks.get_agent_core")
    def test_single_task_rejected(self, mock_get):
        mock_get.return_value = MagicMock()
        tool = ParallelTasksTool()
        result = tool._run(tasks=["just one"])
        assert "one task" in result.lower()

    @patch("cliver.tools.parallel_tasks.get_agent_core")
    def test_empty_tasks_rejected(self, mock_get):
        mock_get.return_value = MagicMock()
        tool = ParallelTasksTool()
        result = tool._run(tasks=[])
        assert "no tasks" in result.lower()

    @patch("cliver.tools.parallel_tasks.get_agent_core")
    def test_parallel_execution(self, mock_get):
        mock_executor = MagicMock()

        async def mock_process(user_input, **kwargs):
            msg = MagicMock()
            msg.content = f"result for: {user_input}"
            return msg

        mock_executor.process_user_input = AsyncMock(side_effect=mock_process)
        mock_get.return_value = mock_executor

        tool = ParallelTasksTool()
        result = tool._run(tasks=["task A", "task B", "task C"])

        assert "3 tasks" in result
        assert "result for: task A" in result
        assert "result for: task B" in result
        assert "result for: task C" in result
        assert mock_executor.process_user_input.call_count == 3

    @patch("cliver.tools.parallel_tasks.get_agent_core")
    def test_handles_task_failure(self, mock_get):
        mock_executor = MagicMock()

        call_count = 0

        async def mock_process(user_input, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("task failed")
            msg = MagicMock()
            msg.content = f"ok: {user_input}"
            return msg

        mock_executor.process_user_input = AsyncMock(side_effect=mock_process)
        mock_get.return_value = mock_executor

        tool = ParallelTasksTool()
        result = tool._run(tasks=["task A", "task B"])

        assert "OK" in result
        assert "FAILED" in result

    def test_tool_registered(self):
        from cliver.tool_registry import ToolRegistry

        registry = ToolRegistry()
        assert "Parallel" in registry.tool_names
