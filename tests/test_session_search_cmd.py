"""Tests for /session search CLI command."""

from click.testing import CliRunner


class TestSessionSearchCommand:
    def test_search_command_exists(self, load_cliver, init_config):
        """The search subcommand should be recognized."""
        result = CliRunner().invoke(load_cliver, ["session", "search", "test"])
        assert "No such command" not in result.output

    def test_search_no_results(self, load_cliver, init_config):
        """Search with no matching sessions should show a message."""
        result = CliRunner().invoke(load_cliver, ["session", "search", "nonexistent"])
        assert result.exit_code == 0

    def test_search_requires_query(self, load_cliver, init_config):
        """Search without a query should show usage help."""
        result = CliRunner().invoke(load_cliver, ["session", "search"])
        assert result.exit_code != 0 or "Missing" in result.output or "Usage" in result.output
