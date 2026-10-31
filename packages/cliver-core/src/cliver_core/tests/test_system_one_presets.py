"""System One presets — Jev (TypeSafe) and Aliyun Bailian decision models."""

from __future__ import annotations

import pytest

from cliver_core import presets
from cliver_core.provider.providers import API_KEY_ENV
from cliver_core.provider.providers.system_one import SystemOneProvider

_WORKSPACE = "ws-h5ufhv81ynivolun"
_BAILIAN_URL = f"https://{_WORKSPACE}.cn-beijing.maas.aliyuncs.com/compatible-mode"


def test_bailian_system_one_builds_the_workspace_host() -> None:
    binding = presets.bailian_system_one(workspace_id=_WORKSPACE, api_key="k")

    assert binding.model == "decision-model-preview"
    assert binding.purpose == "system_one"
    assert binding.protocol == "system-one"
    assert binding.base_url == _BAILIAN_URL
    assert binding.provider_class is SystemOneProvider
    assert binding.call_defaults()["path"] == "/v1/systemone"
    assert binding.api_key == "k"


def test_bailian_region_selects_the_host_suffix() -> None:
    binding = presets.bailian_system_one(workspace_id=_WORKSPACE, region="ap-southeast-1", api_key="k")
    assert binding.base_url == f"https://{_WORKSPACE}.ap-southeast-1.maas.aliyuncs.com/compatible-mode"


def test_bailian_reads_the_dashscope_key_at_call_time(monkeypatch) -> None:
    monkeypatch.setenv("DASHSCOPE_API_KEY", "env-key")
    assert presets.bailian_system_one(workspace_id=_WORKSPACE).api_key == "env-key"
    assert presets.bailian_system_one(workspace_id=_WORKSPACE, api_key="").api_key == ""


def test_bailian_base_url_override_wins() -> None:
    binding = presets.bailian_system_one(
        workspace_id=_WORKSPACE,
        api_key="k",
        base_url="https://proxy.internal/compatible-mode",
    )
    assert binding.base_url == "https://proxy.internal/compatible-mode"


def test_typesafe_system_one_defaults_to_jev_latest() -> None:
    binding = presets.typesafe_system_one(api_key="k")

    assert binding.model == "jev-latest"
    assert binding.purpose == "system_one"
    assert binding.base_url == "https://api.typesafe.ai"
    assert binding.call_defaults()["path"] == "/v1/systemone"


def test_typesafe_reads_its_own_key_variable(monkeypatch) -> None:
    assert API_KEY_ENV["typesafe"] == ("TYPESAFE_API_KEY",)
    monkeypatch.setenv("TYPESAFE_API_KEY", "env-key")
    assert presets.typesafe_system_one().api_key == "env-key"


@pytest.mark.asyncio
async def test_both_presets_build_the_http_provider() -> None:
    for binding, expected_url in (
        (presets.bailian_system_one(workspace_id=_WORKSPACE, api_key="k"), _BAILIAN_URL),
        (presets.typesafe_system_one(api_key="k"), "https://api.typesafe.ai"),
    ):
        provider = binding.get_provider()
        assert isinstance(provider, SystemOneProvider)
        assert provider.base_url == expected_url
        await binding.aclose()


def test_system_one_presets_are_exported() -> None:
    assert "bailian_system_one" in presets.__all__
    assert "typesafe_system_one" in presets.__all__
