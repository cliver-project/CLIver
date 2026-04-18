import pytest
from cliver.config import ProviderConfig, RateLimitConfig


class TestRateLimitConfig:
    def test_basic(self):
        rl = RateLimitConfig(requests=5000, period="5h")
        assert rl.requests == 5000
        assert rl.period == "5h"
        assert rl.margin == 0.1  # default

    def test_custom_margin(self):
        rl = RateLimitConfig(requests=1000, period="1h", margin=0.2)
        assert rl.margin == 0.2


class TestProviderConfig:
    def test_basic(self):
        p = ProviderConfig(name="minimax", type="openai", api_url="https://api.minimaxi.com/v1")
        assert p.name == "minimax"
        assert p.type == "openai"
        assert p.api_url == "https://api.minimaxi.com/v1"
        assert p.api_key is None
        assert p.rate_limit is None

    def test_with_api_key(self):
        p = ProviderConfig(name="mm", type="openai", api_url="http://x", api_key="sk-123")
        assert p.get_api_key() == "sk-123"

    def test_with_rate_limit(self):
        p = ProviderConfig(
            name="mm",
            type="openai",
            api_url="http://x",
            rate_limit=RateLimitConfig(requests=5000, period="5h"),
        )
        assert p.rate_limit.requests == 5000

    def test_model_dump_excludes_name_and_nulls(self):
        p = ProviderConfig(name="mm", type="openai", api_url="http://x")
        dumped = p.model_dump()
        assert "name" not in dumped
        assert "api_key" not in dumped
        assert dumped["type"] == "openai"
        assert dumped["api_url"] == "http://x"
