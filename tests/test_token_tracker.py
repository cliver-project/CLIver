"""Tests for TokenTracker — token extraction, recording, querying, formatting."""

from datetime import datetime, timezone

import pytest
from langchain_core.messages import AIMessage

from cliver.token_tracker import (
    TokenTracker,
    TokenUsage,
    extract_usage,
    format_tokens,
)

# ---------------------------------------------------------------------------
# TokenUsage
# ---------------------------------------------------------------------------


class TestTokenUsage:
    def test_defaults_to_zero(self):
        u = TokenUsage()
        assert u.input_tokens == 0
        assert u.output_tokens == 0
        assert u.total_tokens == 0

    def test_total(self):
        u = TokenUsage(input_tokens=100, output_tokens=50)
        assert u.total_tokens == 150

    def test_add(self):
        a = TokenUsage(input_tokens=100, output_tokens=50)
        b = TokenUsage(input_tokens=200, output_tokens=30)
        c = a + b
        assert c.input_tokens == 300
        assert c.output_tokens == 80

    def test_iadd(self):
        a = TokenUsage(input_tokens=100, output_tokens=50)
        a += TokenUsage(input_tokens=200, output_tokens=30)
        assert a.input_tokens == 300
        assert a.output_tokens == 80


# ---------------------------------------------------------------------------
# format_tokens
# ---------------------------------------------------------------------------


class TestFormatTokens:
    def test_small(self):
        assert format_tokens(0) == "0"
        assert format_tokens(42) == "42"
        assert format_tokens(999) == "999"

    def test_thousands(self):
        assert format_tokens(1_000) == "1.0K"
        assert format_tokens(1_500) == "1.5K"
        assert format_tokens(12_345) == "12.3K"
        assert format_tokens(999_999) == "1000.0K"

    def test_millions(self):
        assert format_tokens(1_000_000) == "1.0M"
        assert format_tokens(2_500_000) == "2.5M"
        assert format_tokens(123_456_789) == "123.5M"

    def test_billions(self):
        assert format_tokens(1_000_000_000) == "1.0B"
        assert format_tokens(3_700_000_000) == "3.7B"


# ---------------------------------------------------------------------------
# extract_usage
# ---------------------------------------------------------------------------


class TestExtractUsage:
    def test_from_usage_metadata(self):
        msg = AIMessage(content="test")
        msg.usage_metadata = {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}
        usage = extract_usage(msg)
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50

    def test_from_response_metadata(self):
        msg = AIMessage(
            content="test",
            response_metadata={"token_usage": {"prompt_tokens": 200, "completion_tokens": 80}},
        )
        usage = extract_usage(msg)
        assert usage.input_tokens == 200
        assert usage.output_tokens == 80

    def test_no_metadata_returns_zero(self):
        msg = AIMessage(content="test")
        usage = extract_usage(msg)
        assert usage.total_tokens == 0

    def test_usage_metadata_preferred_over_response_metadata(self):
        msg = AIMessage(
            content="test",
            response_metadata={"token_usage": {"prompt_tokens": 999, "completion_tokens": 999}},
        )
        msg.usage_metadata = {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}
        usage = extract_usage(msg)
        assert usage.input_tokens == 100  # from usage_metadata, not response_metadata

    def test_handles_none_values(self):
        msg = AIMessage(
            content="test",
            response_metadata={"token_usage": {"prompt_tokens": None, "completion_tokens": None}},
        )
        usage = extract_usage(msg)
        assert usage.total_tokens == 0


# ---------------------------------------------------------------------------
# TokenTracker — recording
# ---------------------------------------------------------------------------


class TestTokenTrackerRecording:
    @pytest.fixture
    def tracker(self, tmp_path):
        return TokenTracker(audit_dir=tmp_path / "audit_logs", agent_name="TestBot")

    def test_record_updates_session_totals(self, tracker):
        tracker.record("qwen", TokenUsage(input_tokens=100, output_tokens=50))
        tracker.record("qwen", TokenUsage(input_tokens=200, output_tokens=30))

        summary = tracker.get_session_summary()
        assert summary["qwen"].input_tokens == 300
        assert summary["qwen"].output_tokens == 80

    def test_record_tracks_multiple_models(self, tracker):
        tracker.record("qwen", TokenUsage(input_tokens=100, output_tokens=50))
        tracker.record("deepseek", TokenUsage(input_tokens=200, output_tokens=80))

        summary = tracker.get_session_summary()
        assert "qwen" in summary
        assert "deepseek" in summary
        assert summary["qwen"].input_tokens == 100
        assert summary["deepseek"].input_tokens == 200

    def test_record_tracks_last_usage(self, tracker):
        tracker.record("qwen", TokenUsage(input_tokens=100, output_tokens=50))
        assert tracker.last_usage.input_tokens == 100
        assert tracker.last_model == "qwen"

        tracker.record("deepseek", TokenUsage(input_tokens=200, output_tokens=80))
        assert tracker.last_usage.input_tokens == 200
        assert tracker.last_model == "deepseek"

    def test_record_skips_zero_usage(self, tracker):
        tracker.record("qwen", TokenUsage())
        assert tracker.get_session_summary() == {}
        assert tracker.last_usage is None

    def test_get_session_total(self, tracker):
        tracker.record("qwen", TokenUsage(input_tokens=100, output_tokens=50))
        tracker.record("deepseek", TokenUsage(input_tokens=200, output_tokens=80))

        total = tracker.get_session_total()
        assert total.input_tokens == 300
        assert total.output_tokens == 130

    def test_empty_session_total(self, tracker):
        total = tracker.get_session_total()
        assert total.total_tokens == 0


# ---------------------------------------------------------------------------
# TokenTracker — audit log persistence
# ---------------------------------------------------------------------------


class TestTokenTrackerAuditLog:
    @pytest.fixture
    def tracker(self, tmp_path):
        return TokenTracker(audit_dir=tmp_path / "audit_logs", agent_name="TestBot")

    def test_creates_audit_file(self, tracker):
        tracker.record("qwen", TokenUsage(input_tokens=100, output_tokens=50))
        audit_files = list(tracker.audit_dir.glob("*.jsonl"))
        assert len(audit_files) == 1
        assert audit_files[0].name.endswith(".jsonl")

    def test_audit_records_are_jsonl(self, tracker):
        import json

        tracker.record("qwen", TokenUsage(input_tokens=100, output_tokens=50))
        tracker.record("deepseek", TokenUsage(input_tokens=200, output_tokens=80))

        audit_file = list(tracker.audit_dir.glob("*.jsonl"))[0]
        lines = audit_file.read_text().strip().split("\n")
        assert len(lines) == 2

        rec1 = json.loads(lines[0])
        assert rec1["model"] == "qwen"
        assert rec1["agent"] == "TestBot"
        assert rec1["in"] == 100
        assert rec1["out"] == 50
        assert "ts" in rec1

    def test_audit_dir_created_on_first_record(self, tracker):
        assert not tracker.audit_dir.exists()
        tracker.record("qwen", TokenUsage(input_tokens=1, output_tokens=1))
        assert tracker.audit_dir.exists()


# ---------------------------------------------------------------------------
# TokenTracker — querying
# ---------------------------------------------------------------------------


class TestTokenTrackerQuerying:
    @pytest.fixture
    def tracker(self, tmp_path):
        return TokenTracker(audit_dir=tmp_path / "audit_logs", agent_name="TestBot")

    def test_query_all(self, tracker):
        tracker.record("qwen", TokenUsage(input_tokens=100, output_tokens=50))
        tracker.record("deepseek", TokenUsage(input_tokens=200, output_tokens=80))

        results = tracker.query()
        assert "qwen" in results
        assert "deepseek" in results
        assert results["qwen"]["TestBot"].input_tokens == 100

    def test_query_by_model(self, tracker):
        tracker.record("qwen", TokenUsage(input_tokens=100, output_tokens=50))
        tracker.record("deepseek", TokenUsage(input_tokens=200, output_tokens=80))

        results = tracker.query(model="qwen")
        assert "qwen" in results
        assert "deepseek" not in results

    def test_query_by_agent(self, tracker):
        tracker.record("qwen", TokenUsage(input_tokens=100, output_tokens=50))

        results = tracker.query(agent="TestBot")
        assert len(results) == 1

        results = tracker.query(agent="OtherBot")
        assert len(results) == 0

    def test_query_empty_logs(self, tracker):
        results = tracker.query()
        assert results == {}

    def test_query_aggregates_multiple_records(self, tracker):
        tracker.record("qwen", TokenUsage(input_tokens=100, output_tokens=50))
        tracker.record("qwen", TokenUsage(input_tokens=200, output_tokens=30))

        results = tracker.query()
        assert results["qwen"]["TestBot"].input_tokens == 300
        assert results["qwen"]["TestBot"].output_tokens == 80

    def test_query_with_date_range(self, tracker, tmp_path):
        """Write audit records with specific timestamps, then filter."""
        import json

        audit_dir = tmp_path / "audit_logs"
        audit_dir.mkdir(parents=True)
        log_file = audit_dir / "2026-03.jsonl"

        records = [
            {"ts": "2026-03-01T10:00:00Z", "model": "qwen", "agent": "Bot", "in": 100, "out": 50},
            {"ts": "2026-03-10T10:00:00Z", "model": "qwen", "agent": "Bot", "in": 200, "out": 80},
            {"ts": "2026-03-20T10:00:00Z", "model": "qwen", "agent": "Bot", "in": 300, "out": 100},
        ]
        with open(log_file, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        tracker_with_data = TokenTracker(audit_dir=audit_dir, agent_name="Bot")

        # Query with start date
        results = tracker_with_data.query(
            start=datetime(2026, 3, 5, tzinfo=timezone.utc),
        )
        assert results["qwen"]["Bot"].input_tokens == 500  # 200 + 300

        # Query with end date
        results = tracker_with_data.query(
            end=datetime(2026, 3, 15, tzinfo=timezone.utc),
        )
        assert results["qwen"]["Bot"].input_tokens == 300  # 100 + 200

        # Query with both
        results = tracker_with_data.query(
            start=datetime(2026, 3, 5, tzinfo=timezone.utc),
            end=datetime(2026, 3, 15, tzinfo=timezone.utc),
        )
        assert results["qwen"]["Bot"].input_tokens == 200  # only the middle record


# ---------------------------------------------------------------------------
# Multi-agent audit log
# ---------------------------------------------------------------------------


class TestMultiAgentAudit:
    def test_different_agents_same_audit_dir(self, tmp_path):
        audit_dir = tmp_path / "audit_logs"
        bot_a = TokenTracker(audit_dir=audit_dir, agent_name="BotA")
        bot_b = TokenTracker(audit_dir=audit_dir, agent_name="BotB")

        bot_a.record("qwen", TokenUsage(input_tokens=100, output_tokens=50))
        bot_b.record("qwen", TokenUsage(input_tokens=200, output_tokens=80))

        # Query all — both agents visible
        results = bot_a.query()
        assert results["qwen"]["BotA"].input_tokens == 100
        assert results["qwen"]["BotB"].input_tokens == 200

        # Query by agent
        results = bot_a.query(agent="BotA")
        assert results["qwen"]["BotA"].input_tokens == 100
        assert "BotB" not in results.get("qwen", {})
