"""Tests for CI/CD capabilities: exit codes, JSON output, timeout, permission flags."""

import asyncio
import json
import time

import pytest
from click.testing import CliRunner

from cliver.llm.errors import TaskTimeoutError


class TestTaskTimeoutError:
    """TaskTimeoutError exception basics."""

    def test_exception_has_partial_result(self):
        err = TaskTimeoutError("timed out", partial_result="partial answer")
        assert str(err) == "timed out"
        assert err.partial_result == "partial answer"

    def test_exception_without_partial_result(self):
        err = TaskTimeoutError("timed out")
        assert err.partial_result is None


class TestReActLoopTimeout:
    """Timeout integration in the Re-Act loop."""

    def test_process_messages_raises_on_deadline(self):
        """_process_messages should raise TaskTimeoutError when deadline is exceeded."""
        from unittest.mock import MagicMock

        from langchain_core.messages import AIMessage, SystemMessage

        from cliver.llm.llm import AgentCore

        executor = AgentCore(
            llm_models={},
            mcp_servers={},
        )

        mock_engine = MagicMock()

        async def slow_infer(messages, tools, **kwargs):
            await asyncio.sleep(0.5)
            return AIMessage(content="done")

        mock_engine.infer = slow_infer
        mock_engine.parse_tool_calls = MagicMock(return_value=None)

        messages = [SystemMessage(content="test")]

        # Deadline already passed
        with pytest.raises(TaskTimeoutError):
            asyncio.run(
                executor._process_messages(
                    mock_engine,
                    "test",
                    messages,
                    50,
                    0,
                    [],
                    None,
                    None,
                    options={},
                    deadline=time.monotonic() - 1,
                )
            )


class TestTimeoutFlag:
    """--timeout flag on chat command."""

    def test_timeout_flag_accepted(self, load_cliver, init_config):
        """--timeout should be accepted without error."""
        result = CliRunner().invoke(
            load_cliver,
            ["chat", "--timeout", "60", "hello"],
            catch_exceptions=False,
            standalone_mode=False,
        )
        # Should reach chat command (may fail on no model, but not on flag parsing)
        assert "no such option: --timeout" not in result.output


class TestOutputFlag:
    """--output json flag on chat command."""

    def test_output_flag_accepted(self, load_cliver, init_config):
        """--output should be accepted without error."""
        result = CliRunner().invoke(
            load_cliver,
            ["chat", "--output", "json", "hello"],
            catch_exceptions=False,
            standalone_mode=False,
        )
        assert "no such option: --output" not in result.output

    def test_output_json_produces_valid_json(self, load_cliver, init_config):
        """--output json should produce valid JSON even on error."""
        result = CliRunner().invoke(
            load_cliver,
            ["chat", "--output", "json", "hello"],
            catch_exceptions=False,
            standalone_mode=False,
        )
        output = result.output.strip()
        data = json.loads(output)
        assert "success" in data
        assert "output" in data
        assert data["success"] is False

    def test_output_text_is_default(self, load_cliver, init_config):
        """Without --output, output should be plain text (not JSON)."""
        result = CliRunner().invoke(
            load_cliver,
            ["chat", "hello"],
            catch_exceptions=False,
            standalone_mode=False,
        )
        try:
            json.loads(result.output.strip())
            raise AssertionError("Output should not be JSON by default")
        except json.JSONDecodeError:
            pass  # Expected — plain text


class TestPermissionFlags:
    """--permission-mode and --allow-tool flags."""

    def test_permission_mode_flag_accepted(self, load_cliver, init_config):
        """--permission-mode should be accepted."""
        result = CliRunner().invoke(
            load_cliver,
            ["chat", "--permission-mode", "yolo", "hello"],
            catch_exceptions=False,
        )
        assert "no such option: --permission-mode" not in result.output

    def test_permission_mode_invalid_value(self, load_cliver, init_config):
        """--permission-mode with invalid value should error."""
        result = CliRunner().invoke(
            load_cliver,
            ["chat", "--permission-mode", "invalid", "hello"],
        )
        assert result.exit_code != 0

    def test_allow_tool_flag_accepted(self, load_cliver, init_config):
        """--allow-tool should be accepted."""
        result = CliRunner().invoke(
            load_cliver,
            ["chat", "--allow-tool", "run_shell_command", "hello"],
            catch_exceptions=False,
        )
        assert "no such option: --allow-tool" not in result.output

    def test_allow_tool_multiple(self, load_cliver, init_config):
        """--allow-tool should accept multiple values."""
        result = CliRunner().invoke(
            load_cliver,
            [
                "chat",
                "--allow-tool",
                "run_shell_command",
                "--allow-tool",
                "read_file",
                "hello",
            ],
            catch_exceptions=False,
        )
        assert "no such option: --allow-tool" not in result.output


class TestExitCodes:
    """Exit codes must propagate from chat command to process."""

    def test_chat_no_model_returns_exit_1(self, load_cliver, init_config):
        """Chat with no model configured should exit with code 1."""
        result = CliRunner().invoke(load_cliver, ["chat", "hello"], catch_exceptions=False, standalone_mode=False)
        assert result.return_value == 1

    def test_known_command_returns_exit_0(self, load_cliver, init_config):
        """Known commands like 'model list' should exit with code 0."""
        result = CliRunner().invoke(load_cliver, ["model", "list"], catch_exceptions=False)
        assert result.exit_code == 0

    def test_help_returns_exit_0(self, load_cliver, init_config):
        """--help should exit with code 0."""
        result = CliRunner().invoke(load_cliver, ["--help"], catch_exceptions=False)
        assert result.exit_code == 0


class TestPromptFlag:
    """Top-level -p / --prompt flag."""

    def test_prompt_flag_accepted(self, load_cliver, init_config):
        """cliver -p 'hello' should be accepted."""
        result = CliRunner().invoke(
            load_cliver,
            ["-p", "hello"],
            catch_exceptions=False,
        )
        assert "no such option: -p" not in result.output

    def test_prompt_long_form(self, load_cliver, init_config):
        """cliver --prompt 'hello' should work."""
        result = CliRunner().invoke(
            load_cliver,
            ["--prompt", "hello"],
            catch_exceptions=False,
        )
        assert "no such option: --prompt" not in result.output

    def test_prompt_with_output_json(self, load_cliver, init_config):
        """cliver -p 'hello' --output json should produce JSON."""
        result = CliRunner().invoke(
            load_cliver,
            ["-p", "hello", "--output", "json"],
            catch_exceptions=False,
        )
        output = result.output.strip()
        data = json.loads(output)
        assert "success" in data

    def test_prompt_with_timeout(self, load_cliver, init_config):
        """cliver -p 'hello' --timeout 60 should be accepted."""
        result = CliRunner().invoke(
            load_cliver,
            ["-p", "hello", "--timeout", "60"],
            catch_exceptions=False,
        )
        assert "no such option: --timeout" not in result.output

    def test_prompt_with_permission_mode(self, load_cliver, init_config):
        """cliver -p 'hello' --permission-mode yolo should be accepted."""
        result = CliRunner().invoke(
            load_cliver,
            ["-p", "hello", "--permission-mode", "yolo"],
            catch_exceptions=False,
        )
        assert "no such option: --permission-mode" not in result.output


class TestCISkill:
    """CI skill should be discoverable."""

    def test_ci_skill_exists(self):
        """The ci skill should be found by SkillManager."""
        from cliver.skill_manager import SkillManager

        manager = SkillManager()
        names = manager.get_skill_names()
        assert "ci" in names

    def test_ci_skill_has_content(self):
        """The ci skill should have non-empty content."""
        from cliver.skill_manager import SkillManager

        manager = SkillManager()
        skill = manager.get_skill("ci")
        assert skill is not None
        assert skill.body and len(skill.body) > 100


class TestCICDIntegration:
    """Integration tests combining multiple CI/CD flags."""

    def test_full_ci_invocation(self, load_cliver, init_config):
        """All CI flags together via top-level -p."""
        result = CliRunner().invoke(
            load_cliver,
            [
                "-p",
                "hello",
                "--output",
                "json",
                "--timeout",
                "60",
                "--permission-mode",
                "yolo",
                "--allow-tool",
                "run_shell_command",
            ],
            catch_exceptions=False,
        )
        output = result.output.strip()
        data = json.loads(output)
        assert "success" in data
        assert "duration_s" in data

    def test_full_ci_via_chat_subcommand(self, load_cliver, init_config):
        """All CI flags together via chat subcommand."""
        result = CliRunner().invoke(
            load_cliver,
            [
                "chat",
                "--output",
                "json",
                "--timeout",
                "60",
                "--permission-mode",
                "yolo",
                "--allow-tool",
                "read_file",
                "hello",
            ],
            catch_exceptions=False,
        )
        output = result.output.strip()
        data = json.loads(output)
        assert "success" in data

    def test_exit_code_with_json_output(self, load_cliver, init_config):
        """JSON output should still set correct exit code."""
        result = CliRunner().invoke(
            load_cliver,
            ["-p", "hello", "--output", "json"],
            catch_exceptions=False,
        )
        # No model configured -> should fail
        data = json.loads(result.output.strip())
        assert data["success"] is False
