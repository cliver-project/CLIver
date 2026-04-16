"""Tests for cost estimation and rate limit tracking."""

import pytest

from cliver.cost_tracker import (
    CostEstimate,
    CostTracker,
    RateLimitInfo,
    format_cost,
    _parse_reset,
)


class TestCostEstimate:
    def test_total_cost(self):
        est = CostEstimate(input_cost=0.01, output_cost=0.02)
        assert est.total_cost == pytest.approx(0.03)

    def test_cached_savings(self):
        est = CostEstimate(input_cost=0.01, output_cost=0.02, cached_savings=0.005)
        assert est.total_with_savings == pytest.approx(0.025)


class TestCostTracker:
    def test_estimate_known_model(self):
        tracker = CostTracker()
        est = tracker.estimate_cost("deepseek-chat", input_tokens=1000, output_tokens=500)
        assert est.total_cost > 0

    def test_estimate_unknown_model(self):
        tracker = CostTracker()
        est = tracker.estimate_cost("unknown-model", input_tokens=1000, output_tokens=500)
        assert est.total_cost == 0

    def test_estimate_with_cache(self):
        tracker = CostTracker()
        no_cache = tracker.estimate_cost("deepseek-chat", input_tokens=1000, output_tokens=500)
        with_cache = tracker.estimate_cost("deepseek-chat", input_tokens=1000, output_tokens=500, cached_tokens=800)
        # Cached should cost less
        assert with_cache.total_cost < no_cache.total_cost

    def test_session_total_accumulates(self):
        tracker = CostTracker()
        tracker.estimate_cost("deepseek-chat", input_tokens=1000, output_tokens=500)
        tracker.estimate_cost("deepseek-chat", input_tokens=2000, output_tokens=1000)
        total = tracker.get_session_total()
        assert total > 0

    def test_prefix_matching(self):
        tracker = CostTracker()
        est = tracker.estimate_cost("deepseek-reasoner-v3", input_tokens=1000, output_tokens=500)
        assert est.total_cost > 0  # matched via "deepseek-reasoner" prefix

    def test_last_cost_tracked(self):
        tracker = CostTracker()
        tracker.estimate_cost("deepseek-chat", input_tokens=1000, output_tokens=500)
        assert tracker.last_cost is not None
        assert tracker.last_model == "deepseek-chat"


class TestRateLimitInfo:
    def test_requests_usage_pct(self):
        info = RateLimitInfo(requests_limit=100, requests_remaining=75)
        assert info.requests_usage_pct == pytest.approx(25.0)

    def test_tokens_usage_pct(self):
        info = RateLimitInfo(tokens_limit=1000000, tokens_remaining=800000)
        assert info.tokens_usage_pct == pytest.approx(20.0)

    def test_none_when_no_data(self):
        info = RateLimitInfo()
        assert info.requests_usage_pct is None
        assert info.tokens_usage_pct is None


class TestRateLimitTracking:
    def test_update_from_headers(self):
        tracker = CostTracker()
        headers = {
            "x-ratelimit-limit-requests": "100",
            "x-ratelimit-remaining-requests": "95",
            "x-ratelimit-limit-tokens": "1000000",
            "x-ratelimit-remaining-tokens": "950000",
        }
        info = tracker.update_rate_limits("deepseek-chat", headers)
        assert info is not None
        assert info.requests_limit == 100
        assert info.requests_remaining == 95
        assert info.tokens_limit == 1000000

    def test_empty_headers(self):
        tracker = CostTracker()
        info = tracker.update_rate_limits("model", {})
        assert info is None

    def test_none_headers(self):
        tracker = CostTracker()
        info = tracker.update_rate_limits("model", None)
        assert info is None


class TestFormatCost:
    def test_small_cost(self):
        assert "$" in format_cost(0.0001)

    def test_medium_cost(self):
        assert format_cost(0.05) == "$0.05"

    def test_large_cost(self):
        assert format_cost(1.50) == "$1.50"


class TestParseReset:
    def test_seconds(self):
        assert _parse_reset("30") == pytest.approx(30.0)

    def test_duration_string(self):
        assert _parse_reset("1m30s") == pytest.approx(90.0)

    def test_milliseconds(self):
        assert _parse_reset("500ms") == pytest.approx(0.5)

    def test_none(self):
        assert _parse_reset(None) is None
