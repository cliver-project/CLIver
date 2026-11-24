"""Shared MCP configuration readers normalize other agents' files."""

from __future__ import annotations

import json
from pathlib import Path

from cliver.mcp_shared import load_shared_mcp_servers


def write_json(path: Path, data: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def load(tmp_path: Path, *, selected, include_project=False, explicit=(), env=None):
    cwd = tmp_path / "project"
    home = tmp_path / "home"
    cwd.mkdir(parents=True, exist_ok=True)
    home.mkdir(parents=True, exist_ok=True)
    return load_shared_mcp_servers(
        selected=selected,
        include_project=include_project,
        explicit_paths=explicit,
        cwd=cwd,
        home=home,
        env=env or {},
    )


def test_claude_user_and_project_entries_are_normalized(tmp_path: Path) -> None:
    cwd = tmp_path / "project"
    home = tmp_path / "home"
    write_json(
        home / ".claude.json",
        {
            "mcpServers": {
                "stdio-srv": {
                    "type": "stdio",
                    "command": "cmd",
                    "args": ["--x"],
                    "env": {"TOKEN": "${TOKEN}"},
                },
                "sse-srv": {"type": "sse", "url": "https://example/sse"},
            },
            "projects": {
                str(cwd): {
                    "mcpServers": {
                        "project-srv": {"command": "project-cmd"},
                    }
                }
            },
        },
    )

    user_only = load(tmp_path, selected=("claude",), env={"TOKEN": "secret"})
    assert set(user_only.servers) == {"stdio-srv"}
    assert user_only.servers["stdio-srv"] == {
        "transport": "stdio",
        "command": "cmd",
        "args": ["--x"],
        "env": {"TOKEN": "secret"},
    }
    assert user_only.provenance["stdio-srv"].source == "claude"
    assert any(problem.code == "invalid-entry" and problem.server == "sse-srv" for problem in user_only.problems)

    with_project = load(tmp_path, selected=("claude",), include_project=True, env={"TOKEN": "secret"})
    assert set(with_project.servers) == {"stdio-srv", "project-srv"}
    assert with_project.servers["project-srv"]["command"] == "project-cmd"
    assert with_project.provenance["project-srv"].source == "claude-project"


def test_cursor_stdio_and_remote_are_inferred(tmp_path: Path) -> None:
    write_json(
        tmp_path / "home" / ".cursor" / "mcp.json",
        {
            "mcpServers": {
                "filesystem": {"command": "npx", "args": ["-y", "server"]},
                "docs": {"url": "https://docs.example/mcp"},
            }
        },
    )

    result = load(tmp_path, selected=("cursor",))

    assert result.servers["filesystem"]["transport"] == "stdio"
    assert result.servers["filesystem"]["command"] == "npx"
    assert result.servers["docs"]["transport"] == "streamable-http"
    assert result.servers["docs"]["url"] == "https://docs.example/mcp"


def test_gemini_settings_transport_tools_and_millisecond_timeouts(tmp_path: Path) -> None:
    cwd = tmp_path / "project"
    write_json(
        tmp_path / "home" / ".gemini" / "settings.json",
        {
            "mcpServers": {
                "http-srv": {
                    "httpUrl": "https://example/mcp",
                    "timeout": 5000,
                    "discoveryTimeoutMs": 30000,
                    "includeTools": ["a"],
                    "excludeTools": ["b"],
                },
                "sse-srv": {"url": "https://example/sse"},
                "stdio-srv": {
                    "command": "cmd",
                    "args": ["x"],
                    "cwd": "./tools",
                    "versionNegotiation": "legacy",
                },
            }
        },
    )

    result = load(tmp_path, selected=("gemini",))

    assert set(result.servers) == {"http-srv", "stdio-srv"}
    http = result.servers["http-srv"]
    assert http["transport"] == "streamable-http"
    assert http["timeout"] == 5.0
    assert http["startup_timeout"] == 30.0
    assert http["enabled_tools"] == ["a"]
    assert http["disabled_tools"] == ["b"]
    assert result.servers["stdio-srv"]["mode"] == "legacy"
    assert result.servers["stdio-srv"]["cwd"] == str((cwd / "tools").resolve())
    assert any(
        problem.code == "invalid-entry" and problem.server == "sse-srv" and "SSE" in problem.message
        for problem in result.problems
    )


def test_vscode_servers_and_input_placeholders(tmp_path: Path) -> None:
    cwd = tmp_path / "project"
    write_json(
        cwd / ".vscode" / "mcp.json",
        {
            "servers": {
                "stdio-srv": {"type": "stdio", "command": "cmd", "args": ["${workspaceFolder}/x"]},
                "http-srv": {"type": "http", "url": "https://example/mcp"},
                "prompted": {"type": "stdio", "command": "cmd", "args": ["${input:token}"]},
            }
        },
    )

    result = load(tmp_path, selected=("vscode-project",))

    assert set(result.servers) == {"stdio-srv", "http-srv"}
    assert result.servers["stdio-srv"]["args"] == [str(cwd / "x")]
    assert result.servers["http-srv"]["transport"] == "streamable-http"
    assert any(problem.code == "unsupported-input" and problem.server == "prompted" for problem in result.problems)


def test_explicit_source_and_unresolved_variable(tmp_path: Path) -> None:
    path = write_json(
        tmp_path / "team-mcp.json",
        {"mcpServers": {"team": {"command": "cmd", "env": {"TOKEN": "${MISSING}"}}}},
    )

    result = load(tmp_path, selected=(), explicit=[path])

    assert set(result.servers) == {"team"}
    assert result.servers["team"]["env"]["TOKEN"] == "${MISSING}"
    assert result.provenance["team"].source == "explicit"
    assert any(problem.code == "unresolved-variable" and problem.server == "team" for problem in result.problems)


def test_project_source_overrides_user_source(tmp_path: Path) -> None:
    write_json(
        tmp_path / "home" / ".cursor" / "mcp.json",
        {"mcpServers": {"shared": {"command": "user-cmd"}}},
    )
    write_json(
        tmp_path / "project" / ".cursor" / "mcp.json",
        {"mcpServers": {"shared": {"command": "project-cmd"}}},
    )

    result = load(tmp_path, selected=("cursor",), include_project=True)

    assert result.servers["shared"]["command"] == "project-cmd"
    assert any(problem.code == "shadowed" and "overrides" in problem.message for problem in result.problems)


def test_missing_explicit_source_is_reported(tmp_path: Path) -> None:
    result = load(tmp_path, selected=(), explicit=[tmp_path / "missing.json"])

    assert result.servers == {}
    assert any(problem.code == "missing-path" and problem.severity == "warning" for problem in result.problems)
