"""Tests for the /cost command — token usage display."""

import json

import pytest
from click.testing import CliRunner

from cliver.cli import Cliver
from cliver.token_tracker import TokenTracker, TokenUsage


@pytest.fixture
def tracker(tmp_path):
    return TokenTracker(audit_dir=tmp_path / "audit_logs", agent_name="TestBot")


# ---------------------------------------------------------------------------
# Session summary display
# ---------------------------------------------------------------------------


class TestSessionSummary:
    def test_empty_session(self, tracker, load_cliver, init_config):
        runner = CliRunner()
        cliver_instance = Cliver()
        cliver_instance.token_tracker = tracker

        result = runner.invoke(load_cliver, ["cost"], obj=cliver_instance)
        assert result.exit_code == 0
        assert "No token usage" in result.output

    def test_single_model(self, tracker, load_cliver, init_config):
        tracker.record("qwen", TokenUsage(input_tokens=1000, output_tokens=250))

        runner = CliRunner()
        cliver_instance = Cliver()
        cliver_instance.token_tracker = tracker

        result = runner.invoke(load_cliver, ["cost"], obj=cliver_instance)
        assert result.exit_code == 0
        assert "qwen" in result.output
        assert "1.0K" in result.output or "1000" in result.output

    def test_multiple_models(self, tracker, load_cliver, init_config):
        tracker.record("qwen", TokenUsage(input_tokens=1000, output_tokens=250))
        tracker.record("deepseek", TokenUsage(input_tokens=2000, output_tokens=500))

        runner = CliRunner()
        cliver_instance = Cliver()
        cliver_instance.token_tracker = tracker

        result = runner.invoke(load_cliver, ["cost"], obj=cliver_instance)
        assert result.exit_code == 0
        assert "qwen" in result.output
        assert "deepseek" in result.output
        assert "Total" in result.output


# ---------------------------------------------------------------------------
# Audit log queries
# ---------------------------------------------------------------------------


class TestCostTotal:
    def test_total_from_audit(self, tracker, load_cliver, init_config):
        tracker.record("qwen", TokenUsage(input_tokens=1000, output_tokens=250))

        runner = CliRunner()
        cliver_instance = Cliver()
        cliver_instance.token_tracker = tracker

        result = runner.invoke(load_cliver, ["cost", "total"], obj=cliver_instance)
        assert result.exit_code == 0
        assert "qwen" in result.output

    def test_filter_by_model(self, tracker, load_cliver, init_config):
        tracker.record("qwen", TokenUsage(input_tokens=1000, output_tokens=250))
        tracker.record("deepseek", TokenUsage(input_tokens=2000, output_tokens=500))

        runner = CliRunner()
        cliver_instance = Cliver()
        cliver_instance.token_tracker = tracker

        result = runner.invoke(
            load_cliver, ["cost", "total", "--model", "qwen"], obj=cliver_instance
        )
        assert result.exit_code == 0
        assert "TestBot" in result.output  # shows agent breakdown

    def test_empty_audit(self, tracker, load_cliver, init_config):
        runner = CliRunner()
        cliver_instance = Cliver()
        cliver_instance.token_tracker = TokenTracker(
            audit_dir=tracker.audit_dir.parent / "empty_audit", agent_name="Bot"
        )

        result = runner.invoke(load_cliver, ["cost", "total"], obj=cliver_instance)
        assert result.exit_code == 0
        assert "No token usage data" in result.output

    def test_date_range_filter(self, tmp_path, load_cliver, init_config):
        audit_dir = tmp_path / "audit_logs"
        audit_dir.mkdir(parents=True)

        # Write test records
        log_file = audit_dir / "2026-03.jsonl"
        records = [
            {"ts": "2026-03-01T10:00:00Z", "model": "qwen", "agent": "Bot", "in": 100, "out": 50},
            {"ts": "2026-03-15T10:00:00Z", "model": "qwen", "agent": "Bot", "in": 200, "out": 80},
        ]
        with open(log_file, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        runner = CliRunner()
        cliver_instance = Cliver()
        cliver_instance.token_tracker = TokenTracker(audit_dir=audit_dir, agent_name="Bot")

        result = runner.invoke(
            load_cliver,
            ["cost", "total", "--from", "2026-03-10", "--to", "2026-03-20"],
            obj=cliver_instance,
        )
        assert result.exit_code == 0
        assert "qwen" in result.output


# ---------------------------------------------------------------------------
# Human-readable formatting in output
# ---------------------------------------------------------------------------


class TestFormattingInOutput:
    def test_large_numbers_formatted(self, tracker, load_cliver, init_config):
        tracker.record("qwen", TokenUsage(input_tokens=1_500_000, output_tokens=350_000))

        runner = CliRunner()
        cliver_instance = Cliver()
        cliver_instance.token_tracker = tracker

        result = runner.invoke(load_cliver, ["cost"], obj=cliver_instance)
        assert "1.5M" in result.output
        assert "350.0K" in result.output
