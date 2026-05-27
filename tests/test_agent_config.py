"""Tests for AgentConfig and AppConfig.agents."""


def test_agent_config_defaults():
    from cliver.config import AgentConfig

    cfg = AgentConfig(name="test")
    assert cfg.name == "test"
    assert cfg.description is None
    assert cfg.role is None
    assert cfg.system_prompt is None
    assert cfg.model is None
    assert cfg.skills is None
    assert cfg.toolsets is None
    assert cfg.auto_fallback is None


def test_agent_config_with_model():
    from cliver.config import AgentConfig

    cfg = AgentConfig(
        name="coder",
        model="deepseek/deepseek-chat",
        system_prompt="You are a helpful assistant",
        toolsets=["code", "web"],
    )
    assert cfg.name == "coder"
    assert cfg.model == "deepseek/deepseek-chat"
    assert cfg.system_prompt == "You are a helpful assistant"
    assert cfg.toolsets == ["code", "web"]


def test_app_config_agents_field():
    from cliver.config import AgentConfig, AppConfig

    cfg = AppConfig(
        agents={
            "researcher": AgentConfig(name="researcher", model="deepseek/deepseek-r1", toolsets=["web"]),
            "coder": AgentConfig(name="coder", model="deepseek/deepseek-chat", toolsets=["code"]),
        },
        default_agent="researcher",
    )
    assert len(cfg.agents) == 2
    assert cfg.agents["researcher"].name == "researcher"
    assert cfg.agents["coder"].name == "coder"
    assert cfg.agents["researcher"].toolsets == ["web"]
    assert cfg.default_agent == "researcher"


def test_app_config_agents_default_empty():
    from cliver.config import AppConfig

    cfg = AppConfig()
    assert cfg.agents == {}
    assert cfg.default_agent is None
