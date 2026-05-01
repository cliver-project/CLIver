"""Tests for workflow CLI commands."""

from click.testing import CliRunner


class TestWorkflowCommands:
    def test_workflow_list_empty(self, load_cliver, init_config):
        result = CliRunner().invoke(load_cliver, ["workflow", "list"])
        assert result.exit_code == 0
        assert "No workflows" in result.output or "workflow" in result.output.lower()

    def test_workflow_show_nonexistent(self, load_cliver, init_config):
        result = CliRunner().invoke(load_cliver, ["workflow", "show", "nope"])
        assert "not found" in result.output.lower()

    def test_workflow_delete_nonexistent(self, load_cliver, init_config):
        result = CliRunner().invoke(load_cliver, ["workflow", "delete", "nope"])
        assert "not found" in result.output.lower()

    def test_workflow_command_exists(self, load_cliver, init_config):
        result = CliRunner().invoke(load_cliver, ["workflow", "--help"])
        assert result.exit_code == 0
        assert "list" in result.output
        assert "run" in result.output
        assert "show" in result.output

    def test_workflow_history_subcommand(self, load_cliver, init_config):
        result = CliRunner().invoke(load_cliver, ["workflow", "history", "nonexistent"])
        assert result.exit_code == 0
        assert "no executions" in result.output.lower() or "nonexistent" in result.output.lower()

    def test_workflow_status_subcommand(self, load_cliver, init_config):
        result = CliRunner().invoke(load_cliver, ["workflow", "status", "nope", "--thread", "t1"])
        assert result.exit_code == 0
        assert "no execution" in result.output.lower() or "not found" in result.output.lower()
