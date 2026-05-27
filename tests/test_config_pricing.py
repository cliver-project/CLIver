"""Tests for PricingConfig and ModelConfig.get_resolved_pricing()."""

from cliver.config import ModelConfig, PricingConfig, ProviderConfig


class TestPricingConfig:
    def test_all_fields_optional(self):
        p = PricingConfig()
        assert p.currency is None
        assert p.input is None
        assert p.output is None
        assert p.cached_input is None

    def test_full_config(self):
        p = PricingConfig(currency="USD", input=2.5, output=10.0, cached_input=1.25)
        assert p.currency == "USD"
        assert p.input == 2.5
        assert p.output == 10.0
        assert p.cached_input == 1.25

    def test_partial_config(self):
        p = PricingConfig(input=1.0, output=4.0)
        assert p.input == 1.0
        assert p.output == 4.0
        assert p.currency is None
        assert p.cached_input is None


class TestGetResolvedPricing:
    def _make_model(self, provider_pricing=None):
        model = ModelConfig(name="test-model", provider="test-provider", model="test-model")
        if provider_pricing:
            provider = ProviderConfig(
                name="test-provider",
                type="openai",
                api_url="http://localhost",
                pricing=provider_pricing,
            )
            model._provider_config = provider
        return model

    def test_no_pricing_anywhere(self):
        model = self._make_model()
        assert model.get_resolved_pricing() is None

    def test_provider_pricing_only(self):
        model = self._make_model(
            provider_pricing=PricingConfig(currency="CNY", input=1.0, output=4.0, cached_input=0.25)
        )
        assert model.get_resolved_pricing() == (1.0, 4.0, 0.25, "CNY")

    def test_cached_input_defaults_to_input(self):
        model = self._make_model(provider_pricing=PricingConfig(input=5.0, output=10.0))
        assert model.get_resolved_pricing() == (5.0, 10.0, 5.0, "USD")

    def test_currency_defaults_to_usd(self):
        model = self._make_model(provider_pricing=PricingConfig(input=1.0, output=2.0))
        result = model.get_resolved_pricing()
        assert result[3] == "USD"

    def test_incomplete_pricing_returns_none(self):
        model = self._make_model(provider_pricing=PricingConfig(input=1.0))
        assert model.get_resolved_pricing() is None

    def test_provider_has_only_currency(self):
        model = self._make_model(
            provider_pricing=PricingConfig(currency="EUR", input=3.0, output=6.0),
        )
        assert model.get_resolved_pricing() == (3.0, 6.0, 3.0, "EUR")
