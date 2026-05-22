import json
from cliver.model.models import Provider, Endpoint, Model


class TestProvider:
    def test_defaults(self):
        p = Provider(name="openai", type="openai")
        assert p.id is not None
        assert len(p.id) == 8
        assert p.name == "openai"
        assert p.type == "openai"
        assert p.api_key is None
        assert p.rate_limit is None
        assert p.created_at is not None
        assert p.updated_at is not None

    def test_json_roundtrip(self):
        p = Provider(
            name="openai",
            type="openai",
            api_key="{{ key('openai_key') }}",
            rate_limit={"requests": 500, "period": "1m", "margin": 0.1},
        )
        d = p.model_dump()
        assert isinstance(d["rate_limit"], dict)
        assert d["rate_limit"]["requests"] == 500


class TestEndpoint:
    def test_creation(self):
        e = Endpoint(provider_id="a1b2c3d4", base_url="https://api.openai.com/v1")
        assert e.id is not None
        assert e.provider_id == "a1b2c3d4"


class TestModel:
    def test_creation(self):
        m = Model(
            provider_id="a1b2c3d4",
            endpoint_id="e5f6g7h8",
            name="gpt-4o",
            capabilities=["text_to_text", "image_to_text", "tool_calling"],
            options={"temperature": 0.7, "max_tokens": 4096},
        )
        assert m.name == "gpt-4o"
        assert "text_to_text" in m.capabilities

    def test_capabilities_json_serialization(self):
        m = Model(
            provider_id="a1b2c3d4",
            endpoint_id="e5f6g7h8",
            name="gpt-4o",
            capabilities=["text_to_text", "image_to_text"],
        )
        d = m.model_dump()
        assert isinstance(d["capabilities"], list)
