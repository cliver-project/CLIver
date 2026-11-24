"""The CLI's JSON configuration file: models, skills, MCP, and precedence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from cliver.config import CLIConfig, create_mcp_client, load_config

USER_KEY = {
    "DEEPSEEK_API_KEY": "deepseek-key",
    "TYPESAFE_API_KEY": "typesafe-key",
    "DASHSCOPE_API_KEY": "dashscope-key",
}


def write_config(path: Path, data: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_each_purpose_loads_as_an_ordered_list(tmp_path: Path) -> None:
    config_dir = tmp_path / "conf"
    write_config(
        config_dir / "config.json",
        {
            "models": {
                "llm": [
                    {"model": "chat-primary", "provider": "deepseek", "api_key_env": "DEEPSEEK_API_KEY"},
                    {"model": "chat-backup", "provider": "glm", "api_key": "backup-key"},
                ],
                "image": {"model": "image-model", "provider": "dashscope"},
                "system_one": [
                    {
                        "provider": "bailian",
                        "workspace_id": "ws-1",
                        "model": "decision-model-preview",
                    },
                    {"provider": "typesafe", "model": "jev-latest"},
                ],
            }
        },
    )

    config = load_config(config_dir=config_dir, cwd=tmp_path / "project", home=tmp_path / "home", env=USER_KEY)

    assert [binding.model for binding in config.models["llm"]] == ["chat-primary", "chat-backup"]
    assert [binding.model for binding in config.models["image"]] == ["image-model"]
    assert [binding.purpose for binding in config.models["image"]] == ["image"]
    assert [binding.model for binding in config.models["system_one"]] == [
        "decision-model-preview",
        "jev-latest",
    ]
    assert {binding.purpose for binding in config.models["system_one"]} == {"system_one"}
    assert config.paths == (config_dir / "config.json",)


def test_a_single_model_object_is_a_one_element_list(tmp_path: Path) -> None:
    config_dir = tmp_path / "conf"
    write_config(
        config_dir / "config.json",
        {"models": {"llm": {"model": "chat", "base_url": "https://example.test/v1", "api_key": "k"}}},
    )

    config = load_config(config_dir=config_dir, cwd=tmp_path / "project", home=tmp_path / "home", env={})

    assert len(config.models["llm"]) == 1
    assert config.models["llm"][0].model == "chat"


def test_audio_is_the_legacy_tts_slot(tmp_path: Path) -> None:
    config_dir = tmp_path / "conf"
    write_config(
        config_dir / "config.json",
        {"models": {"audio": {"model": "voice", "base_url": "https://example.test/v1", "api_key": "k"}}},
    )

    config = load_config(config_dir=config_dir, cwd=tmp_path / "project", home=tmp_path / "home", env={})

    binding = config.models["audio"][0]
    assert binding.model == "voice"
    assert binding.purpose == "tts"


def test_project_models_replace_user_models_and_sources_append(tmp_path: Path) -> None:
    config_dir = tmp_path / "conf"
    project = tmp_path / "project"
    write_config(
        config_dir / "config.json",
        {
            "models": {"llm": {"model": "user-chat", "base_url": "https://user.test/v1", "api_key": "u"}},
            "skills": {"sources": ["user-skills"]},
            "mcp": {
                "servers": {
                    "shared": {"transport": "stdio", "command": "user"},
                    "only-user": {"transport": "stdio", "command": "user-only"},
                },
                "enabled": ["shared"],
            },
        },
    )
    write_config(
        project / ".cliver" / "config.json",
        {
            "models": {"llm": {"model": "project-chat", "base_url": "https://project.test/v1", "api_key": "p"}},
            "skills": {"sources": ["project-skills"]},
            "mcp": {
                "servers": {
                    "shared": {"transport": "stdio", "command": "project"},
                    "only-project": {"transport": "stdio", "command": "project-only"},
                },
                "enabled": ["shared", "only-project"],
            },
        },
    )

    config = load_config(config_dir=config_dir, cwd=project, home=tmp_path / "home", env={})

    assert [binding.model for binding in config.models["llm"]] == ["project-chat"]
    assert config.skill_sources == (
        (config_dir / "user-skills").resolve(),
        (project / ".cliver" / "project-skills").resolve(),
    )
    assert set(config.mcp_servers) == {"shared", "only-user", "only-project"}
    assert config.mcp_servers["shared"]["command"] == "project"
    assert config.mcp_enabled == ("shared", "only-project")
    assert config.paths == (config_dir / "config.json", project / ".cliver" / "config.json")


def test_variable_expansion_and_api_key_env(tmp_path: Path) -> None:
    config_dir = tmp_path / "conf"
    write_config(
        config_dir / "config.json",
        {
            "models": {
                "llm": {
                    "model": "${CHAT_MODEL}",
                    "base_url": "${CHAT_BASE:-https://fallback.test}/v1",
                    "api_key_env": "CHAT_KEY",
                }
            }
        },
    )

    config = load_config(
        config_dir=config_dir,
        cwd=tmp_path / "project",
        home=tmp_path / "home",
        env={"CHAT_MODEL": "expanded", "CHAT_KEY": "expanded-key"},
    )

    binding = config.models["llm"][0]
    assert binding.model == "expanded"
    assert binding.base_url == "https://fallback.test/v1"
    assert binding.api_key == "expanded-key"


def test_missing_variable_or_api_key_env_is_reported(tmp_path: Path) -> None:
    config_dir = tmp_path / "conf"
    write_config(config_dir / "config.json", {"models": {"llm": {"model": "${MISSING}", "api_key": "k"}}})

    with pytest.raises(ValueError, match="MISSING"):
        load_config(config_dir=config_dir, cwd=tmp_path / "project", home=tmp_path / "home", env={})

    write_config(
        config_dir / "config.json",
        {"models": {"llm": {"model": "m", "api_key_env": "MISSING_KEY"}}},
    )
    with pytest.raises(ValueError, match="MISSING_KEY"):
        load_config(config_dir=config_dir, cwd=tmp_path / "project", home=tmp_path / "home", env={})


def test_unknown_purpose_or_binding_key_is_rejected(tmp_path: Path) -> None:
    config_dir = tmp_path / "conf"
    write_config(config_dir / "config.json", {"models": {"audioo": {"model": "m"}}})
    with pytest.raises(ValueError, match="unknown model purpose"):
        load_config(config_dir=config_dir, cwd=tmp_path / "project", home=tmp_path / "home", env={})

    write_config(
        config_dir / "config.json",
        {"models": {"llm": {"model": "m", "base_url": "https://example.test/v1", "temprature": 0.2}}},
    )
    with pytest.raises(ValueError, match="unknown key"):
        load_config(config_dir=config_dir, cwd=tmp_path / "project", home=tmp_path / "home", env={})


def test_selector_section_maps_to_the_selector_environment(tmp_path: Path) -> None:
    config_dir = tmp_path / "conf"
    write_config(
        config_dir / "config.json",
        {"selector": {"enabled": True, "min_tools": 30, "close_read": False}},
    )

    config = load_config(config_dir=config_dir, cwd=tmp_path / "project", home=tmp_path / "home", env={})

    assert config.selector_env == {
        "CLIVER_SELECTOR": "on",
        "CLIVER_SELECTOR_MIN_TOOLS": "30",
        "CLIVER_SELECTOR_CLOSE_READ": "off",
    }


def test_create_mcp_client_reads_the_cliver_user_file_and_honors_enabled(tmp_path: Path) -> None:
    config_dir = tmp_path / "conf"
    write_config(
        config_dir / "mcp.json",
        {
            "mcpServers": {
                "sqlite": {"transport": "stdio", "command": "echo"},
                "docs": {"transport": "streamable-http", "url": "https://example.test/mcp"},
            }
        },
    )
    write_config(config_dir / "config.json", {"mcp": {"enabled": ["sqlite"]}})

    config = load_config(config_dir=config_dir, cwd=tmp_path / "project", home=tmp_path / "home", env={})
    selection = create_mcp_client(
        config,
        cwd=tmp_path / "project",
        home=tmp_path / "home",
        env={"CLIVER_CONF_DIR": str(config_dir)},
    )

    assert selection.client is not None
    assert set(selection.client._adapters) == {"sqlite"}
    assert selection.problems == ()


def test_create_mcp_client_mixes_inline_and_source_servers_and_applies_disabled(tmp_path: Path) -> None:
    config_dir = tmp_path / "conf"
    write_config(
        config_dir / "extra.json",
        {"mcpServers": {"from-file": {"transport": "stdio", "command": "echo"}}},
    )
    write_config(
        config_dir / "config.json",
        {
            "mcp": {
                "sources": [str(config_dir / "extra.json")],
                "servers": {
                    "inline": {"transport": "stdio", "command": "echo"},
                    "disabled": {"transport": "stdio", "command": "echo"},
                },
                "disabled": ["disabled"],
            }
        },
    )

    config = load_config(config_dir=config_dir, cwd=tmp_path / "project", home=tmp_path / "home", env={})
    selection = create_mcp_client(
        config,
        cwd=tmp_path / "project",
        home=tmp_path / "home",
        env={"CLIVER_CONF_DIR": str(config_dir)},
    )

    assert selection.client is not None
    assert set(selection.client._adapters) == {"inline", "from-file"}


def test_shared_mcp_config_is_parsed_and_paths_resolve(tmp_path: Path) -> None:
    config_dir = tmp_path / "conf"
    write_config(
        config_dir / "config.json",
        {
            "mcp": {
                "shared": ["claude", "cursor"],
                "shared_project": True,
                "shared_sources": ["team.json"],
            }
        },
    )

    config = load_config(config_dir=config_dir, cwd=tmp_path / "project", home=tmp_path / "home", env={})

    assert config.mcp_shared == ("claude", "cursor")
    assert config.mcp_shared_project is True
    assert config.mcp_shared_sources == ((config_dir / "team.json").resolve(),)


def test_shared_mcp_selection_validation(tmp_path: Path) -> None:
    config_dir = tmp_path / "conf"
    write_config(config_dir / "config.json", {"mcp": {"shared": True}})
    config = load_config(config_dir=config_dir, cwd=tmp_path / "project", home=tmp_path / "home", env={})
    assert config.mcp_shared is None

    write_config(config_dir / "config.json", {"mcp": {"shared": ["unknown"]}})
    with pytest.raises(ValueError, match="unknown mcp.shared"):
        load_config(config_dir=config_dir, cwd=tmp_path / "project", home=tmp_path / "home", env={})

    write_config(config_dir / "config.json", {"mcp": {"shared_project": "yes"}})
    with pytest.raises(ValueError, match="shared_project"):
        load_config(config_dir=config_dir, cwd=tmp_path / "project", home=tmp_path / "home", env={})


def test_create_mcp_client_reads_explicit_shared_source(tmp_path: Path) -> None:
    config_dir = tmp_path / "conf"
    write_config(
        config_dir / "team.json",
        {"mcpServers": {"team": {"command": "echo"}}},
    )
    write_config(config_dir / "config.json", {"mcp": {"shared_sources": ["team.json"]}})

    config = load_config(config_dir=config_dir, cwd=tmp_path / "project", home=tmp_path / "home", env={})
    selection = create_mcp_client(
        config,
        cwd=tmp_path / "project",
        home=tmp_path / "home",
        env={"CLIVER_CONF_DIR": str(config_dir)},
    )

    assert selection.client is not None
    assert set(selection.client._adapters) == {"team"}
    assert selection.client._adapters["team"].config["command"] == "echo"


def test_cliver_owned_servers_shadow_shared_servers(tmp_path: Path) -> None:
    config_dir = tmp_path / "conf"
    home = tmp_path / "home"
    write_config(
        config_dir / "mcp.json",
        {"mcpServers": {"shared": {"transport": "stdio", "command": "cliver-cmd"}}},
    )
    write_config(
        home / ".cursor" / "mcp.json",
        {
            "mcpServers": {
                "shared": {"command": "cursor-cmd"},
                "cursor-only": {"command": "cursor-only-cmd"},
            }
        },
    )
    write_config(config_dir / "config.json", {"mcp": {"shared": ["cursor"]}})

    config = load_config(config_dir=config_dir, cwd=tmp_path / "project", home=home, env={})
    selection = create_mcp_client(
        config,
        cwd=tmp_path / "project",
        home=home,
        env={"CLIVER_CONF_DIR": str(config_dir)},
    )

    assert selection.client is not None
    assert set(selection.client._adapters) == {"shared", "cursor-only"}
    assert selection.client._adapters["shared"].config["command"] == "cliver-cmd"
    assert any(problem.code == "shadowed" and problem.server == "shared" for problem in selection.problems)


def test_create_mcp_client_without_servers_returns_none(tmp_path: Path) -> None:
    config = CLIConfig(config_dir=tmp_path / "conf")

    selection = create_mcp_client(
        config,
        cwd=tmp_path / "project",
        home=tmp_path / "home",
        env={"CLIVER_CONF_DIR": str(tmp_path / "conf")},
    )

    assert selection.client is None


def test_explicit_config_path_must_exist(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="configuration file not found"):
        load_config(config_path=tmp_path / "missing.json", cwd=tmp_path, home=tmp_path, env={})


def test_cliver_config_environment_variable_selects_one_file(tmp_path: Path) -> None:
    path = write_config(
        tmp_path / "custom.json",
        {"models": {"llm": {"model": "only", "base_url": "https://example.test/v1", "api_key": "k"}}},
    )

    config = load_config(
        cwd=tmp_path / "project",
        home=tmp_path / "home",
        env={"CLIVER_CONFIG": str(path)},
    )

    assert config.paths == (path,)
    assert [binding.model for binding in config.models["llm"]] == ["only"]


def test_mcp_enabled_missing_name_is_reported(tmp_path: Path) -> None:
    config_dir = tmp_path / "conf"
    write_config(
        config_dir / "mcp.json",
        {"mcpServers": {"present": {"transport": "stdio", "command": "echo"}}},
    )
    write_config(config_dir / "config.json", {"mcp": {"enabled": ["present", "missing"]}})

    config = load_config(config_dir=config_dir, cwd=tmp_path / "project", home=tmp_path / "home", env={})
    selection = create_mcp_client(
        config,
        cwd=tmp_path / "project",
        home=tmp_path / "home",
        env={"CLIVER_CONF_DIR": str(config_dir)},
    )

    assert selection.client is not None
    assert set(selection.client._adapters) == {"present"}
    assert any(problem.code == "selection-missing" and "missing" in problem.message for problem in selection.problems)
