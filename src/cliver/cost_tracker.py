"""
Cost estimation and rate limit tracking for LLM API calls.

Tracks:
- Per-model pricing (input/output per million tokens)
- Cost per call and session total
- Rate limit headers from API responses (x-ratelimit-*)
"""

import logging
import time
from dataclasses import dataclass
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Currency symbols for display
_CURRENCY_SYMBOLS = {
    "USD": "$",
    "CNY": "¥",
    "EUR": "€",
    "JPY": "¥",
}


@dataclass
class RateLimitInfo:
    """Rate limit state from API response headers."""

    requests_limit: Optional[int] = None
    requests_remaining: Optional[int] = None
    requests_reset_seconds: Optional[float] = None
    tokens_limit: Optional[int] = None
    tokens_remaining: Optional[int] = None
    tokens_reset_seconds: Optional[float] = None
    captured_at: float = 0.0  # monotonic timestamp

    @property
    def requests_usage_pct(self) -> Optional[float]:
        if self.requests_limit and self.requests_remaining is not None:
            used = self.requests_limit - self.requests_remaining
            return (used / self.requests_limit) * 100
        return None

    @property
    def tokens_usage_pct(self) -> Optional[float]:
        if self.tokens_limit and self.tokens_remaining is not None:
            used = self.tokens_limit - self.tokens_remaining
            return (used / self.tokens_limit) * 100
        return None


@dataclass
class CostEstimate:
    """Cost estimate for a single LLM call."""

    input_cost: float = 0.0
    output_cost: float = 0.0
    cached_savings: float = 0.0  # how much was saved by cache hits
    currency: str = "USD"

    @property
    def total_cost(self) -> float:
        return self.input_cost + self.output_cost

    @property
    def total_with_savings(self) -> float:
        return self.total_cost - self.cached_savings


class CostTracker:
    """Tracks API costs and rate limits per model."""

    def __init__(self, pricing: Optional[Dict[str, tuple]] = None):
        self.pricing = pricing or {}
        self.session_costs: Dict[str, float] = {}  # model → total cost
        self.last_cost: Optional[CostEstimate] = None
        self.last_model: Optional[str] = None
        self.rate_limits: Dict[str, RateLimitInfo] = {}  # model → latest rate limit

    def estimate_cost(self, model: str, input_tokens: int, output_tokens: int, cached_tokens: int = 0) -> CostEstimate:
        """Estimate cost for a call based on token counts."""
        pricing = self._find_pricing(model)
        if not pricing:
            return CostEstimate()

        input_price, output_price, cached_price, currency = pricing

        # Cached tokens are charged at cached rate, not full input rate
        non_cached_input = max(0, input_tokens - cached_tokens)
        input_cost = (non_cached_input / 1_000_000) * input_price
        cached_cost = (cached_tokens / 1_000_000) * cached_price
        output_cost = (output_tokens / 1_000_000) * output_price
        savings = (cached_tokens / 1_000_000) * (input_price - cached_price)

        estimate = CostEstimate(
            input_cost=input_cost + cached_cost,
            output_cost=output_cost,
            cached_savings=savings,
            currency=currency,
        )

        # Track session total
        self.last_cost = estimate
        self.last_model = model
        if model not in self.session_costs:
            self.session_costs[model] = 0.0
        self.session_costs[model] += estimate.total_cost

        return estimate

    def get_session_total(self) -> float:
        """Get total session cost across all models."""
        return sum(self.session_costs.values())

    def update_rate_limits(self, model: str, headers: dict) -> Optional[RateLimitInfo]:
        """Parse rate limit headers from an API response."""
        if not headers:
            return None

        info = RateLimitInfo(captured_at=time.monotonic())

        # Standard OpenAI rate limit headers
        if "x-ratelimit-limit-requests" in headers:
            info.requests_limit = _safe_int(headers.get("x-ratelimit-limit-requests"))
        if "x-ratelimit-remaining-requests" in headers:
            info.requests_remaining = _safe_int(headers.get("x-ratelimit-remaining-requests"))
        if "x-ratelimit-reset-requests" in headers:
            info.requests_reset_seconds = _parse_reset(headers.get("x-ratelimit-reset-requests"))
        if "x-ratelimit-limit-tokens" in headers:
            info.tokens_limit = _safe_int(headers.get("x-ratelimit-limit-tokens"))
        if "x-ratelimit-remaining-tokens" in headers:
            info.tokens_remaining = _safe_int(headers.get("x-ratelimit-remaining-tokens"))
        if "x-ratelimit-reset-tokens" in headers:
            info.tokens_reset_seconds = _parse_reset(headers.get("x-ratelimit-reset-tokens"))

        # Only store if we got any data
        if info.requests_limit or info.tokens_limit:
            self.rate_limits[model] = info
            return info
        return None

    def _find_pricing(self, model: str) -> Optional[tuple]:
        """Find pricing for a model by exact or prefix matching."""
        model_lower = model.lower()
        if model_lower in self.pricing:
            return self.pricing[model_lower]
        for pattern, pricing in self.pricing.items():
            if model_lower.startswith(pattern):
                return pricing
        return None


def format_cost(cost: float, currency: str = "USD") -> str:
    """Format a cost value with the appropriate currency symbol."""
    symbol = _CURRENCY_SYMBOLS.get(currency, currency + " ")
    if cost < 0.001:
        return f"{symbol}{cost:.4f}"
    if cost < 0.01:
        return f"{symbol}{cost:.3f}"
    if cost < 1.0:
        return f"{symbol}{cost:.2f}"
    return f"{symbol}{cost:.2f}"


def _safe_int(value) -> Optional[int]:
    """Safely parse an int from a header value."""
    if value is None:
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def _parse_reset(value) -> Optional[float]:
    """Parse a rate limit reset value (seconds or duration string like '1m30s')."""
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        pass
    # Parse duration strings like "1m30s", "500ms", "2s"
    import re

    total = 0.0
    for match in re.finditer(r"(\d+(?:\.\d+)?)(ms|s|m|h)", str(value)):
        num, unit = float(match.group(1)), match.group(2)
        if unit == "ms":
            total += num / 1000
        elif unit == "s":
            total += num
        elif unit == "m":
            total += num * 60
        elif unit == "h":
            total += num * 3600
    return total if total > 0 else None
