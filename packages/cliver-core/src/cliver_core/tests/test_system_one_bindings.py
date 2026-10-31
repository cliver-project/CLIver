"""The ``system_one`` purpose and its binding factory."""

from __future__ import annotations

import pytest

from cliver_core.bindings import PURPOSES, ModelBinding, ModelBindings
from cliver_core.provider.providers.system_one import SystemOneProvider

_BASE = "https://ws-h5ufhv81ynivolun.cn-beijing.maas.aliyuncs.com/compatible-mode"


def test_system_one_is_a_first_class_purpose() -> None:
    assert "system_one" in PURPOSES


def test_system_one_binding_carries_the_decision_wire() -> None:
    binding = ModelBinding.system_one(
        model="decision-model-preview",
        api_key="k",
        base_url=_BASE,
        path="/v1/systemone",
        timeout=5.0,
    )
    assert binding.purpose == "system_one"
    assert binding.protocol == "system-one"
    assert binding.provider_class is SystemOneProvider
    assert binding.base_url == _BASE
    assert binding.call_defaults() == {"path": "/v1/systemone", "timeout": 5.0}


@pytest.mark.asyncio
async def test_system_one_binding_builds_the_http_provider() -> None:
    binding = ModelBinding.system_one(model="decision-model-preview", api_key="k", base_url=_BASE)

    provider = binding.get_provider()

    assert isinstance(provider, SystemOneProvider)
    assert provider.base_url == _BASE
    assert provider.api_key == "k"
    await binding.aclose()
    assert binding._built is None


def test_model_bindings_mounts_system_one() -> None:
    binding = ModelBinding.system_one(model="m", api_key="k", base_url=_BASE)
    bindings = ModelBindings(llm=ModelBinding.llm(model="mock"), system_one=binding)

    assert bindings.get("system_one") is binding
    assert "system_one" in bindings.bound_purposes()


def test_model_bindings_rejects_a_binding_under_the_wrong_purpose() -> None:
    llm = ModelBinding.llm(model="mock")
    with pytest.raises(ValueError, match="system_one"):
        ModelBindings(system_one=llm).get("system_one")


def test_system_one_accepts_same_purpose_fallbacks() -> None:
    primary = ModelBinding.system_one(model="primary", api_key="k", base_url=_BASE)
    backup = ModelBinding.system_one(model="backup", api_key="k", base_url=_BASE)

    bindings = ModelBindings(system_one=[primary, backup])

    assert bindings.system_one is primary
    assert bindings.get_all("system_one") == [primary, backup]
