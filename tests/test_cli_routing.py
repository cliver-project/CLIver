"""Tests for CliverGroup command routing.

Verifies that unrecognized CLI args are routed to the chat command,
while known commands, flags, and interactive mode still work correctly.
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


class TestUnrecognizedTextRoutesToChat:
    """Unrecognized first args should be routed to 'chat' command."""

    def test_quoted_sentence(self, load_cliver, init_config):
        """cliver "tell me a joke" → cliver chat "tell me a joke" """
        result = CliRunner().invoke(load_cliver, ["tell me a joke"])
        # Should NOT show "No such command"
        assert "No such command" not in result.output
        # Should have been routed to chat — will fail because no model
        # is configured, which proves it reached the chat command
        assert "Error" in result.output or result.exit_code != 0

    def test_multiple_words(self, load_cliver, init_config):
        """cliver what time is it → cliver chat what time is it"""
        result = CliRunner().invoke(load_cliver, ["what", "time", "is", "it"])
        assert "No such command" not in result.output

    def test_single_word_not_a_command(self, load_cliver, init_config):
        """cliver hello → cliver chat hello"""
        result = CliRunner().invoke(load_cliver, ["hello"])
        assert "No such command" not in result.output

    def test_sentence_with_punctuation(self, load_cliver, init_config):
        """cliver "who are you?" → chat"""
        result = CliRunner().invoke(load_cliver, ["who are you?"])
        assert "No such command" not in result.output

    def test_explicit_chat_still_works(self, load_cliver, init_config):
        """cliver chat "hello" should still work as before."""
        result = CliRunner().invoke(load_cliver, ["chat", "hello"])
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

    def test_chat_with_options_still_works(self, load_cliver, init_config):
        """cliver chat --model qwen "hello" should parse correctly."""
        result = CliRunner().invoke(load_cliver, ["chat", "--model", "qwen", "hello"])
        # Should reach chat, may error on model not found but not "No such command"
        assert "No such command" not in result.output


class TestInteractiveMode:
    """cliver with no args should enter interactive mode (not crash)."""

    def test_no_args_does_not_crash(self, load_cliver, init_config):
        """cliver (no args) should not show 'No such command'."""
        # CliRunner with no input will cause EOFError in interactive mode,
        # which is expected and handled gracefully
        result = CliRunner().invoke(load_cliver, [], input="\n")
        assert "No such command" not in result.output
