"""Tests for CLI command routing.

Verifies that known commands resolve normally, flags work, and
interactive mode doesn't crash. CliverGroup has been removed;
unrecognized text routing to chat is now handled by the TUI
CommandRouter, not at the Click group level.
"""

from click.testing import CliRunner


class TestKnownCommandsStillWork:
    """Known commands must resolve normally — no regression."""

    def test_model_list(self, load_cliver, init_config):
        result = CliRunner().invoke(load_cliver, ["model", "list"])
        assert result.exit_code == 0
        assert "No LLM Models configured." in result.output

    def test_config_show(self, load_cliver, init_config):
        result = CliRunner().invoke(load_cliver, ["config", "show"])
        assert result.exit_code == 0

    def test_config_validate(self, load_cliver, init_config):
        result = CliRunner().invoke(load_cliver, ["config", "validate"])
        assert result.exit_code == 0
        assert "Configuration is valid" in result.output

    def test_config_path(self, load_cliver, init_config):
        result = CliRunner().invoke(load_cliver, ["config", "path"])
        assert result.exit_code == 0
        assert "Configuration file path:" in result.output

    def test_mcp_list(self, load_cliver, init_config):
        result = CliRunner().invoke(load_cliver, ["mcp", "list"])
        assert result.exit_code == 0

    def test_help_command(self, load_cliver, init_config):
        result = CliRunner().invoke(load_cliver, ["help"])
        assert result.exit_code == 0
        assert "COMMAND" in result.output

    def test_help_subcommand(self, load_cliver, init_config):
        result = CliRunner().invoke(load_cliver, ["help", "model"])
        assert result.exit_code == 0
        assert "Manage LLM Models" in result.output


class TestFlagsStillWork:
    """CLI flags must resolve normally."""

    def test_help_flag(self, load_cliver, init_config):
        result = CliRunner().invoke(load_cliver, ["--help"])
        assert result.exit_code == 0
        assert "Cliver:" in result.output
        assert "Commands:" in result.output

    def test_version_flag(self, load_cliver, init_config):
        result = CliRunner().invoke(load_cliver, ["--version"])
        assert result.exit_code == 0
        assert "cliver" in result.output


class TestUnrecognizedTextBehavior:
    """Without CliverGroup, unrecognized args produce Click errors.

    The TUI CommandRouter handles text-to-chat routing interactively.
    """

    def test_unrecognized_text_shows_error(self, load_cliver, init_config):
        """Unrecognized first arg should produce a Click error."""
        result = CliRunner().invoke(load_cliver, ["tell me a joke"])
        assert result.exit_code != 0

    def test_prompt_flag_routes_to_query(self, load_cliver, init_config):
        """cliver -p "hello" routes to LLM query via --prompt flag."""
        result = CliRunner().invoke(load_cliver, ["-p", "hello"])
        # Should reach the LLM query path (may error because no model configured)
        assert "No such command" not in result.output


class TestUnrecognizedDoesNotHijackCommands:
    """Edge cases: make sure routing doesn't break similar-looking inputs."""

    def test_model_is_not_chat(self, load_cliver, init_config):
        """'model' is a known command — should NOT be routed to chat."""
        result = CliRunner().invoke(load_cliver, ["model", "list"])
        assert result.exit_code == 0
        # If it was routed to chat, it would not show this
        assert "Models" in result.output or "No LLM Models" in result.output

    def test_config_is_not_chat(self, load_cliver, init_config):
        """'config' is a known command — should NOT be routed to chat."""
        result = CliRunner().invoke(load_cliver, ["config", "path"])
        assert result.exit_code == 0
        assert "Configuration file path:" in result.output

    def test_prompt_with_model_option(self, load_cliver, init_config):
        """cliver -m qwen -p "hello" should parse correctly."""
        result = CliRunner().invoke(load_cliver, ["-m", "qwen", "-p", "hello"])
        # Should reach query, may error on model not found but not "No such command"
        assert "No such command" not in result.output


class TestInteractiveMode:
    """cliver with no args should enter interactive mode (not crash)."""

    def test_no_args_does_not_crash(self, load_cliver, init_config):
        """cliver (no args) should not show 'No such command'."""
        # CliRunner with no input will cause EOFError in interactive mode,
        # which is expected and handled gracefully
        result = CliRunner().invoke(load_cliver, [], input="\n")
        assert "No such command" not in result.output
