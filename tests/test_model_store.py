import pytest
from pathlib import Path

from cliver.model.store import ModelStore
from cliver.model.models import Provider, Endpoint, Model


@pytest.fixture
def store(tmp_path: Path) -> ModelStore:
    db_path = tmp_path / "test.db"
    return ModelStore(db_path)


class TestProviderCRUD:
    def test_create_provider(self, store):
        p = store.create_provider(
            name="openai",
            type="openai",
            api_key="{{ key('openai_key') }}",
        )
        assert p.name == "openai"
        assert p.type == "openai"

    def test_list_providers(self, store):
        store.create_provider(name="openai", type="openai")
        store.create_provider(name="deepseek", type="deepseek")
        providers = store.list_providers()
        assert len(providers) == 2

    def test_get_provider(self, store):
        p = store.create_provider(name="openai", type="openai")
        got = store.get_provider(p.id)
        assert got is not None
        assert got.name == "openai"

    def test_update_provider(self, store):
        p = store.create_provider(name="openai", type="openai")
        updated = store.update_provider(p.id, api_key="{{ key('new_key') }}")
        assert updated.api_key == "{{ key('new_key') }}"

    def test_delete_provider_cascades(self, store):
        p = store.create_provider(name="openai", type="openai")
        ep = store.create_endpoint(p.id, "https://api.openai.com/v1")
        store.create_model(
            provider_id=p.id,
            endpoint_id=ep.id,
            name="gpt-4o",
            capabilities=["text_to_text"],
        )
        assert store.delete_provider(p.id) is True
        assert store.list_models() == []


class TestEndpointCRUD:
    def test_create_endpoint(self, store):
        p = store.create_provider(name="openai", type="openai")
        ep = store.create_endpoint(p.id, "https://api.openai.com/v1")
        assert ep.base_url == "https://api.openai.com/v1"

    def test_unique_provider_base_url(self, store):
        p = store.create_provider(name="openai", type="openai")
        store.create_endpoint(p.id, "https://api.openai.com/v1")
        with pytest.raises(Exception):
            store.create_endpoint(p.id, "https://api.openai.com/v1")

    def test_list_endpoints(self, store):
        p = store.create_provider(name="openai", type="openai")
        store.create_endpoint(p.id, "https://api.openai.com/v1")
        store.create_endpoint(p.id, "https://api2.openai.com/v1")
        eps = store.list_endpoints(p.id)
        assert len(eps) == 2

    def test_delete_endpoint(self, store):
        p = store.create_provider(name="openai", type="openai")
        ep = store.create_endpoint(p.id, "https://api.openai.com/v1")
        assert store.delete_endpoint(ep.id) is True


class TestModelCRUD:
    def test_create_model(self, store):
        p = store.create_provider(name="openai", type="openai")
        ep = store.create_endpoint(p.id, "https://api.openai.com/v1")
        m = store.create_model(
            provider_id=p.id,
            endpoint_id=ep.id,
            name="gpt-4o",
            capabilities=["text_to_text", "image_to_text"],
            options={"temperature": 0.7},
        )
        assert m.name == "gpt-4o"
        assert m.capabilities == ["text_to_text", "image_to_text"]

    def test_list_models(self, store):
        p = store.create_provider(name="openai", type="openai")
        ep = store.create_endpoint(p.id, "https://api.openai.com/v1")
        store.create_model(provider_id=p.id, endpoint_id=ep.id, name="gpt-4o")
        store.create_model(provider_id=p.id, endpoint_id=ep.id, name="gpt-4o-mini")
        models = store.list_models()
        assert len(models) == 2

    def test_list_models_filter_by_capability(self, store):
        p = store.create_provider(name="openai", type="openai")
        ep = store.create_endpoint(p.id, "https://api.openai.com/v1")
        store.create_model(provider_id=p.id, endpoint_id=ep.id, name="gpt-4o",
                           capabilities=["text_to_text", "image_to_text"])
        store.create_model(provider_id=p.id, endpoint_id=ep.id, name="dall-e-3",
                           capabilities=["text_to_image"])
        text_models = store.list_models(capability="text_to_text")
        assert len(text_models) == 1
        assert text_models[0].name == "gpt-4o"

    def test_set_default_model(self, store):
        p = store.create_provider(name="openai", type="openai")
        ep = store.create_endpoint(p.id, "https://api.openai.com/v1")
        m1 = store.create_model(provider_id=p.id, endpoint_id=ep.id, name="gpt-4o")
        m2 = store.create_model(provider_id=p.id, endpoint_id=ep.id, name="gpt-4o-mini",
                                is_default=1)
        store.set_default_model(m1.id)
        m1_updated = store.get_model(m1.id)
        m2_updated = store.get_model(m2.id)
        assert m1_updated.is_default == 1
        assert m2_updated.is_default == 0

    def test_update_model(self, store):
        p = store.create_provider(name="openai", type="openai")
        ep = store.create_endpoint(p.id, "https://api.openai.com/v1")
        m = store.create_model(provider_id=p.id, endpoint_id=ep.id, name="gpt-4o")
        updated = store.update_model(m.id, options={"temperature": 0.9})
        assert updated.options == {"temperature": 0.9}

    def test_delete_model(self, store):
        p = store.create_provider(name="openai", type="openai")
        ep = store.create_endpoint(p.id, "https://api.openai.com/v1")
        m = store.create_model(provider_id=p.id, endpoint_id=ep.id, name="gpt-4o")
        assert store.delete_model(m.id) is True
        assert store.get_model(m.id) is None
