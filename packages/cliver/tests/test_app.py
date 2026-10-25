"""Application-side wiring: bindings, tools, skills, and permission callback."""

from __future__ import annotations

import sys

import cliver.app as app
import pytest
from cliver.app import build_agent, llm_binding_from_env, make_permission_callback
from cliver_core import DecisionAction, ModelBinding, ToolCallContext, ToolCategory

BUILTIN_TOOL_NAMES = [
    "Bash",
    "Exec",
    "Grep",
    "LS",
    "Read",
    "TodoRead",
    "TodoWrite",
    "WebFetch",
    "Write",
]


def test_llm_binding_from_env_uses_provider_defaults():
    binding = llm_binding_from_env(
        {
            "CLIVER_MODEL": "deepseek-v4-flash",
            "CLIVER_PROVIDER": "deepseek",
            "CLIVER_PROTOCOL": "openai",
            "CLIVER_API_KEY": "test-key",
        }
    )

    assert binding.model == "deepseek-v4-flash"
    assert binding.provider_name == "deepseek"
    assert binding.protocol == "openai"
    assert binding.api_key == "test-key"
    assert binding.base_url == ""


def test_llm_binding_from_env_accepts_custom_endpoint():
    binding = llm_binding_from_env(
        {
            "CLIVER_MODEL": "local-model",
            "CLIVER_BASE_URL": "http://localhost:11434/v1",
        }
    )

    assert binding.base_url == "http://localhost:11434/v1"
    assert binding.provider_name is None


def test_llm_binding_from_env_falls_back_to_the_brand_key():
    binding = llm_binding_from_env(
        {
            "CLIVER_MODEL": "deepseek-v4-flash",
            "CLIVER_PROVIDER": "deepseek",
            "DEEPSEEK_API_KEY": "brand-key",
        }
    )

    assert binding.api_key == "brand-key"


def test_llm_binding_from_env_prefers_the_cliver_key_over_the_brand_key():
    binding = llm_binding_from_env(
        {
            "CLIVER_MODEL": "deepseek-v4-flash",
            "CLIVER_PROVIDER": "deepseek",
            "CLIVER_API_KEY": "cliver-key",
            "DEEPSEEK_API_KEY": "brand-key",
        }
    )

    assert binding.api_key == "cliver-key"


def test_llm_binding_from_env_requires_model():
    with pytest.raises(ValueError, match="CLIVER_MODEL"):
        llm_binding_from_env({})


def test_llm_binding_from_env_requires_an_endpoint():
    with pytest.raises(ValueError, match="CLIVER_PROVIDER"):
        llm_binding_from_env({"CLIVER_MODEL": "some-model"})


def test_build_agent_wires_builtin_tools_and_all_skill_sources(tmp_path):
    binding = ModelBinding.llm(model="m", api_key="k", base_url="https://example.com/v1")

    agent = build_agent(
        binding,
        cwd=tmp_path,
        config_dir=tmp_path / "conf",
        home=tmp_path / "home",
    )

    assert agent.tool_registry.tool_names == BUILTIN_TOOL_NAMES
    skills = {skill.name for skill in agent.skill_manager.list_skills()}
    assert {"brainstorm", "write-plan", "execute-plan", "wbs-planner"} <= skills
    assert str(tmp_path) in agent._builtin_system_prompt


@pytest.mark.asyncio
async def test_permission_callback_assume_yes_allows():
    callback = make_permission_callback(assume_yes=True)

    decision = await callback(ToolCallContext(tool_name="Bash", args={}, server=None, category=ToolCategory.ASK))

    assert decision.action is DecisionAction.ALLOW


@pytest.mark.asyncio
async def test_permission_callback_denies_when_non_interactive(monkeypatch):
    class NonInteractive:
        def isatty(self) -> bool:
            return False

    monkeypatch.setattr(sys, "stdin", NonInteractive())

    callback = make_permission_callback(assume_yes=False)
    decision = await callback(ToolCallContext(tool_name="Write", args={}, server=None, category=ToolCategory.ASK))

    assert decision.action is DecisionAction.DENY
    assert "non-interactive" in (decision.reason or "")


@pytest.mark.asyncio
async def test_permission_callback_confirms_interactively(monkeypatch):
    class Interactive:
        def isatty(self) -> bool:
            return True

    monkeypatch.setattr(sys, "stdin", Interactive())
    monkeypatch.setattr(app, "input", lambda _prompt: "yes", raising=False)

    callback = make_permission_callback(assume_yes=False)
    decision = await callback(ToolCallContext(tool_name="Write", args={}, server=None, category=ToolCategory.ASK))

    assert decision.action is DecisionAction.ALLOW
