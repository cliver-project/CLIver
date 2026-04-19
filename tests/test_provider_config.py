import yaml

from cliver.config import ConfigManager, ModelConfig, ProviderConfig, RateLimitConfig


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


class TestModelConfigProviderResolution:
    def _make_provider(self, **kwargs):
        defaults = {"name": "mm", "type": "openai", "api_url": "https://api.mm.com/v1", "api_key": "sk-prov"}
        defaults.update(kwargs)
        return ProviderConfig(**defaults)

    def test_url_from_provider(self):
        prov = self._make_provider()
        mc = ModelConfig(name="m1", provider="mm")
        mc._provider_config = prov
        assert mc.get_resolved_url() == "https://api.mm.com/v1"

    def test_url_model_override(self):
        prov = self._make_provider()
        mc = ModelConfig(name="m1", provider="mm", url="https://custom.url/v1")
        mc._provider_config = prov
        assert mc.get_resolved_url() == "https://custom.url/v1"

    def test_api_key_from_provider(self):
        prov = self._make_provider()
        mc = ModelConfig(name="m1", provider="mm")
        mc._provider_config = prov
        assert mc.get_api_key() == "sk-prov"

    def test_api_key_model_override(self):
        prov = self._make_provider()
        mc = ModelConfig(name="m1", provider="mm", api_key="sk-model")
        mc._provider_config = prov
        assert mc.get_api_key() == "sk-model"

    def test_provider_type(self):
        prov = self._make_provider(type="anthropic")
        mc = ModelConfig(name="m1", provider="mm")
        mc._provider_config = prov
        assert mc.get_provider_type() == "anthropic"

    def test_provider_type_legacy(self):
        """Without a linked ProviderConfig, provider field IS the type."""
        mc = ModelConfig(name="m1", provider="openai", url="http://x")
        assert mc.get_provider_type() == "openai"

    def test_resolved_url_no_provider_no_url(self):
        mc = ModelConfig(name="m1", provider="openai")
        assert mc.get_resolved_url() is None

    def test_model_dump_excludes_provider_config(self):
        prov = self._make_provider()
        mc = ModelConfig(name="m1", provider="mm", url="http://x")
        mc._provider_config = prov
        dumped = mc.model_dump()
        assert "_provider_config" not in dumped


class TestConfigLoadingWithProviders:
    def test_load_providers_section(self, tmp_path):
        config_yaml = {
            "providers": {
                "minimax": {
                    "type": "openai",
                    "api_url": "https://api.minimaxi.com/v1",
                    "api_key": "sk-test",
                    "rate_limit": {"requests": 5000, "period": "5h"},
                }
            },
            "models": {
                "m1": {
                    "provider": "minimax",
                    "name_in_provider": "MiniMax-M2.7",
                }
            },
            "default_model": "m1",
        }
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_yaml, f)

        cm = ConfigManager(tmp_path)
        assert "minimax" in cm.config.providers
        prov = cm.config.providers["minimax"]
        assert prov.type == "openai"
        assert prov.rate_limit.requests == 5000

        m1 = cm.config.models["m1"]
        assert m1.get_provider_type() == "openai"
        assert m1.get_resolved_url() == "https://api.minimaxi.com/v1"
        assert m1.get_api_key() == "sk-test"


class TestAutoMigration:
    def test_legacy_config_creates_synthetic_providers(self, tmp_path):
        config_yaml = {
            "models": {
                "qwen": {
                    "provider": "openai",
                    "url": "http://localhost:8080/v1",
                    "api_key": "sk-old",
                    "name_in_provider": "qwen3",
                }
            },
            "default_model": "qwen",
        }
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_yaml, f)

        cm = ConfigManager(tmp_path)
        m = cm.config.models["qwen"]
        assert m.get_provider_type() == "openai"
        assert m.get_resolved_url() == "http://localhost:8080/v1"
        assert m.get_api_key() == "sk-old"

    def test_legacy_models_share_provider_when_same_url(self, tmp_path):
        config_yaml = {
            "models": {
                "m1": {"provider": "openai", "url": "http://api.com/v1", "api_key": "sk-1"},
                "m2": {"provider": "openai", "url": "http://api.com/v1", "api_key": "sk-1"},
            },
            "default_model": "m1",
        }
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_yaml, f)

        cm = ConfigManager(tmp_path)
        m1 = cm.config.models["m1"]
        m2 = cm.config.models["m2"]
        assert m1._provider_config is m2._provider_config


class TestProviderConfigMediaUrls:
    def test_image_url(self):
        p = ProviderConfig(
            name="mm",
            type="openai",
            api_url="http://x",
            image_url="https://api.minimaxi.com/v1/image_generation",
        )
        assert p.image_url == "https://api.minimaxi.com/v1/image_generation"

    def test_audio_url(self):
        p = ProviderConfig(
            name="mm",
            type="openai",
            api_url="http://x",
            audio_url="https://api.minimaxi.com/v1/audio_generation",
        )
        assert p.audio_url == "https://api.minimaxi.com/v1/audio_generation"

    def test_urls_default_none(self):
        p = ProviderConfig(name="mm", type="openai", api_url="http://x")
        assert p.image_url is None
        assert p.audio_url is None

    def test_model_dump_excludes_null_urls(self):
        p = ProviderConfig(name="mm", type="openai", api_url="http://x")
        dumped = p.model_dump()
        assert "image_url" not in dumped
        assert "audio_url" not in dumped

    def test_model_dump_includes_set_urls(self):
        p = ProviderConfig(
            name="mm",
            type="openai",
            api_url="http://x",
            image_url="http://img",
        )
        dumped = p.model_dump()
        assert dumped["image_url"] == "http://img"

    def test_load_config_with_image_url(self, tmp_path):
        import yaml

        from cliver.config import ConfigManager

        config_yaml = {
            "providers": {
                "mm": {
                    "type": "openai",
                    "api_url": "http://x",
                    "image_url": "http://img",
                }
            },
        }
        (tmp_path / "config.yaml").write_text(yaml.dump(config_yaml))
        cm = ConfigManager(tmp_path)
        assert cm.config.providers["mm"].image_url == "http://img"
