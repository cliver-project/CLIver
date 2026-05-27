"""Tests for AgentConfig and AppConfig.agents."""


def test_agent_config_defaults():
    from cliver.config import AgentConfig

    cfg = AgentConfig()
    assert cfg.type == "cliver"
    assert cfg.description is None
    assert cfg.role is None
    assert cfg.model is None
    assert cfg.command is None
    assert cfg.args is None
    assert cfg.env is None
    assert cfg.working_dir is None
    assert cfg.timeout_s == 300
    assert cfg.max_retries == 0
    assert cfg.auto_fallback is None


def test_agent_config_cli_type():
    from cliver.config import AgentConfig

    cfg = AgentConfig(
        type="claude",
        model="anthropic/claude-sonnet-4-20250514",
        working_dir="./src",
        timeout_s=600,
        max_retries=2,
    )
    assert cfg.type == "claude"
    assert cfg.timeout_s == 600
    assert cfg.max_retries == 2


def test_agent_config_custom_type():
    from cliver.config import AgentConfig

    cfg = AgentConfig(
        type="aider",
        command="aider",
        args=["--message"],
        env={"OPENAI_API_KEY": "sk-test"},
        timeout_s=600,
    )
    assert cfg.type == "aider"
    assert cfg.command == "aider"
    assert cfg.args == ["--message"]
    assert cfg.env == {"OPENAI_API_KEY": "sk-test"}


def test_app_config_agents_field():
    from cliver.config import AgentConfig, AppConfig

    cfg = AppConfig(
        agents={
            "researcher": AgentConfig(type="cliver", model="deepseek/deepseek-r1"),
            "coder": AgentConfig(type="claude", timeout_s=600),
        },
        default_agent="researcher",
    )
    assert len(cfg.agents) == 2
    assert cfg.agents["researcher"].type == "cliver"
    assert cfg.agents["coder"].type == "claude"
    assert cfg.default_agent == "researcher"


def test_app_config_agents_default_empty():
    from cliver.config import AppConfig

    cfg = AppConfig()
    assert cfg.agents == {}
    assert cfg.default_agent is None
