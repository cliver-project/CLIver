"""Application-side System One configuration: binding and selector from env."""

from __future__ import annotations

import pytest
from cliver.app import build_agent, selector_from_env, system_one_binding_from_env
from cliver_core import ModelBinding
from cliver_core.local import LocalSystemOneProvider
from cliver_core.system_one.selector import SystemOneSelector


def test_no_system_one_configuration_returns_none():
    assert system_one_binding_from_env({}) is None


def test_bailian_preset_uses_workspace_id_and_dashscope_key():
    binding = system_one_binding_from_env(
        {
            "CLIVER_SYSTEM_ONE_MODEL": "decision-model-preview",
            "CLIVER_SYSTEM_ONE_PROVIDER": "bailian",
            "CLIVER_SYSTEM_ONE_WORKSPACE_ID": "ws-test-123",
            "DASHSCOPE_API_KEY": "test-dashscope-key",
        }
    )

    assert binding is not None
    assert binding.purpose == "system_one"
    assert binding.model == "decision-model-preview"
    assert binding.base_url == "https://ws-test-123.cn-beijing.maas.aliyuncs.com/compatible-mode"
    assert binding.api_key == "test-dashscope-key"
    assert binding.path == "/v1/systemone"


def test_bailian_region_override():
    binding = system_one_binding_from_env(
        {
            "CLIVER_SYSTEM_ONE_MODEL": "decision-model-preview",
            "CLIVER_SYSTEM_ONE_PROVIDER": "bailian",
            "CLIVER_SYSTEM_ONE_WORKSPACE_ID": "ws-test-123",
            "CLIVER_SYSTEM_ONE_REGION": "ap-southeast-1",
            "CLIVER_SYSTEM_ONE_API_KEY": "explicit-key",
        }
    )

    assert binding is not None
    assert binding.base_url == "https://ws-test-123.ap-southeast-1.maas.aliyuncs.com/compatible-mode"
    assert binding.api_key == "explicit-key"


def test_bailian_accepts_explicit_base_url_without_workspace_id():
    binding = system_one_binding_from_env(
        {
            "CLIVER_SYSTEM_ONE_MODEL": "decision-model-preview",
            "CLIVER_SYSTEM_ONE_PROVIDER": "bailian",
            "CLIVER_SYSTEM_ONE_BASE_URL": "https://example.test/compatible-mode",
            "CLIVER_SYSTEM_ONE_API_KEY": "explicit-key",
        }
    )

    assert binding is not None
    assert binding.base_url == "https://example.test/compatible-mode"


def test_bailian_requires_workspace_id_or_base_url():
    with pytest.raises(ValueError, match="CLIVER_SYSTEM_ONE_WORKSPACE_ID"):
        system_one_binding_from_env(
            {
                "CLIVER_SYSTEM_ONE_MODEL": "decision-model-preview",
                "CLIVER_SYSTEM_ONE_PROVIDER": "bailian",
            }
        )


def test_typesafe_preset_uses_its_key_variable():
    binding = system_one_binding_from_env(
        {
            "CLIVER_SYSTEM_ONE_MODEL": "jev-latest",
            "CLIVER_SYSTEM_ONE_PROVIDER": "typesafe",
            "TYPESAFE_API_KEY": "test-typesafe-key",
        }
    )

    assert binding is not None
    assert binding.base_url == "https://api.typesafe.ai"
    assert binding.api_key == "test-typesafe-key"


def test_custom_endpoint_uses_the_shared_system_one_wire():
    binding = system_one_binding_from_env(
        {
            "CLIVER_SYSTEM_ONE_MODEL": "my-decision-model",
            "CLIVER_SYSTEM_ONE_BASE_URL": "https://decisions.example.test/v1",
            "CLIVER_SYSTEM_ONE_API_KEY": "explicit-key",
        }
    )

    assert binding is not None
    assert binding.purpose == "system_one"
    assert binding.protocol == "system-one"
    assert binding.base_url == "https://decisions.example.test/v1"


def test_model_is_required_when_anything_is_configured():
    with pytest.raises(ValueError, match="CLIVER_SYSTEM_ONE_MODEL"):
        system_one_binding_from_env({"CLIVER_SYSTEM_ONE_PROVIDER": "bailian"})


def test_unknown_provider_is_rejected():
    with pytest.raises(ValueError, match="unknown System One provider"):
        system_one_binding_from_env(
            {
                "CLIVER_SYSTEM_ONE_MODEL": "decision-model-preview",
                "CLIVER_SYSTEM_ONE_PROVIDER": "nope",
            }
        )


def test_laya_builds_a_local_system_one_binding():
    binding = system_one_binding_from_env(
        {
            "CLIVER_SYSTEM_ONE_MODEL": "multilingual",
            "CLIVER_SYSTEM_ONE_PROVIDER": "laya",
            "CLIVER_SYSTEM_ONE_DEVICE": "cpu",
            "CLIVER_SYSTEM_ONE_MAX_LEN": "512",
        }
    )

    assert binding is not None
    assert binding.purpose == "system_one"
    assert binding.protocol == "local"
    assert binding.model == "multilingual"
    assert isinstance(binding.provider, LocalSystemOneProvider)
    assert binding.provider.backend_name == "laya"
    assert binding.provider.device == "cpu"
    # Laya takes its token limit from the provider (its model-side default),
    # not from a per-call binding option.
    assert binding.provider.max_length == 512
    assert binding.params == {}


def test_laya_rejects_a_non_positive_max_len():
    with pytest.raises(ValueError, match="CLIVER_SYSTEM_ONE_MAX_LEN"):
        system_one_binding_from_env(
            {
                "CLIVER_SYSTEM_ONE_MODEL": "multilingual",
                "CLIVER_SYSTEM_ONE_PROVIDER": "laya",
                "CLIVER_SYSTEM_ONE_MAX_LEN": "0",
            }
        )


def test_selector_is_off_by_default():
    binding = system_one_binding_from_env(
        {
            "CLIVER_SYSTEM_ONE_MODEL": "decision-model-preview",
            "CLIVER_SYSTEM_ONE_PROVIDER": "typesafe",
            "TYPESAFE_API_KEY": "test-typesafe-key",
        }
    )
    assert selector_from_env(binding, {}) is None


def test_selector_switch_without_a_system_one_model_warns_and_stays_off(capsys):
    assert selector_from_env(None, {"CLIVER_SELECTOR": "on"}) is None
    captured = capsys.readouterr()
    assert "no System One model is configured" in captured.err


def test_selector_reads_thresholds_and_tuning_from_env():
    binding = system_one_binding_from_env(
        {
            "CLIVER_SYSTEM_ONE_MODEL": "decision-model-preview",
            "CLIVER_SYSTEM_ONE_PROVIDER": "typesafe",
            "TYPESAFE_API_KEY": "test-typesafe-key",
        }
    )
    selector = selector_from_env(
        binding,
        {
            "CLIVER_SELECTOR": "true",
            "CLIVER_SELECTOR_MIN_TOOLS": "30",
            "CLIVER_SELECTOR_MIN_SKILLS": "8",
            "CLIVER_SELECTOR_TOP_K_TOOLS": "10",
            "CLIVER_SELECTOR_TOP_K_SKILLS": "4",
            "CLIVER_SELECTOR_CHUNK_SIZE": "10",
            "CLIVER_SELECTOR_SHORTLIST": "12",
            "CLIVER_SELECTOR_CLOSE_READ": "off",
            "CLIVER_SELECTOR_FIT_THRESHOLD": "0.5",
            "CLIVER_SELECTOR_TIMEOUT": "4.5",
        },
    )

    assert isinstance(selector, SystemOneSelector)
    assert selector.binding is binding
    assert selector.min_tools == 30
    assert selector.min_skills == 8
    assert selector.top_k_tools == 10
    assert selector.top_k_skills == 4
    assert selector.chunk_size == 10
    assert selector.shortlist == 12
    assert selector.close_read is False
    assert selector.fit_threshold == 0.5
    assert selector.timeout == 4.5


def test_selector_defaults_are_conservative():
    binding = system_one_binding_from_env(
        {
            "CLIVER_SYSTEM_ONE_MODEL": "decision-model-preview",
            "CLIVER_SYSTEM_ONE_PROVIDER": "typesafe",
            "TYPESAFE_API_KEY": "test-typesafe-key",
        }
    )
    selector = selector_from_env(binding, {"CLIVER_SELECTOR": "1"})

    assert selector is not None
    assert (selector.min_tools, selector.min_skills) == (20, 20)
    assert (selector.top_k_tools, selector.top_k_skills) == (10, 5)
    assert selector.timeout is None


def _laya_binding():
    return system_one_binding_from_env(
        {
            "CLIVER_SYSTEM_ONE_MODEL": "multilingual",
            "CLIVER_SYSTEM_ONE_PROVIDER": "laya",
        }
    )


def test_selector_turns_the_close_read_off_for_a_local_model_by_default():
    selector = selector_from_env(_laya_binding(), {"CLIVER_SELECTOR": "on"})

    assert selector is not None
    assert selector.close_read is False


def test_explicit_close_read_setting_wins_for_a_local_model():
    selector = selector_from_env(_laya_binding(), {"CLIVER_SELECTOR": "on", "CLIVER_SELECTOR_CLOSE_READ": "on"})

    assert selector is not None
    assert selector.close_read is True


def test_invalid_selector_boolean_is_rejected():
    binding = system_one_binding_from_env(
        {
            "CLIVER_SYSTEM_ONE_MODEL": "decision-model-preview",
            "CLIVER_SYSTEM_ONE_PROVIDER": "typesafe",
            "TYPESAFE_API_KEY": "test-typesafe-key",
        }
    )
    with pytest.raises(ValueError, match="CLIVER_SELECTOR"):
        selector_from_env(binding, {"CLIVER_SELECTOR": "maybe"})


def test_build_agent_mounts_system_one_and_the_selector(tmp_path):
    llm = ModelBinding.llm(model="chat-model", api_key="k", base_url="https://chat.example.test/v1")
    system_one = system_one_binding_from_env(
        {
            "CLIVER_SYSTEM_ONE_MODEL": "decision-model-preview",
            "CLIVER_SYSTEM_ONE_PROVIDER": "typesafe",
            "TYPESAFE_API_KEY": "test-typesafe-key",
        }
    )
    selector = selector_from_env(system_one, {"CLIVER_SELECTOR": "on"})

    agent = build_agent(
        llm,
        cwd=tmp_path,
        config_dir=tmp_path / "conf",
        home=tmp_path / "home",
        system_one_binding=system_one,
        candidate_selector=selector,
    )

    assert agent.bindings.llm is llm
    assert agent.bindings.system_one is system_one
    assert agent.candidate_selector is selector
