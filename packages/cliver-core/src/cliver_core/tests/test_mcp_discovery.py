"""MCP discovery from cliver's own places — stdlib-only, injected paths.

Discovery must stay importable (and usable) with the MCP SDK absent, so
this module imports nothing beyond ``cliver_core.mcp.discovery`` and the
standard library.  ``cwd``/``home``/``env`` are always injected so the
tests never read the real machine.
"""

from __future__ import annotations

import json
from pathlib import Path

from cliver_core.mcp.discovery import discover_mcp_servers


def _write(path: Path, data) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _stdio(command: str = "npx") -> dict:
    return {"transport": "stdio", "command": command, "args": ["-y", "some-server"]}


def _discover(tmp_path: Path, **kwargs):
    options: dict = {"cwd": tmp_path / "project", "home": tmp_path / "home", "env": {}}
    options.update(kwargs)
    return discover_mcp_servers(**options)


def test_project_user_and_dropins_read_in_precedence_order(tmp_path):
    project_root, home = tmp_path / "project", tmp_path / "home"
    _write(project_root / ".cliver" / "mcp.json", {"mcpServers": {"proj": _stdio("proj-cmd")}})
    _write(home / ".cliver" / "mcp.json", {"mcpServers": {"user-store": _stdio("user-cmd")}})
    _write(home / ".cliver" / "mcps" / "b.json", {"mcpServers": {"dropin-b": _stdio()}})
    _write(home / ".cliver" / "mcps" / "a.json", _stdio("dropin-cmd"))  # single object, name = stem

    result = _discover(tmp_path, cwd=project_root, home=home)

    assert list(result.servers) == ["proj", "user-store", "a", "dropin-b"]
    assert result.servers["a"]["command"] == "dropin-cmd"
    assert result.servers["a"]["args"] == ["-y", "some-server"]
    assert result.provenance["proj"].source == "cliver_project"
    assert result.provenance["user-store"].source == "cliver_user"
    assert result.provenance["a"].source == "cliver_user_dropin"
    assert result.provenance["a"].path == home / ".cliver" / "mcps" / "a.json"
    assert len({source for source, _ in result.sources_read}) == 3
    assert result.summary() == "4 servers from 3 sources"


def test_cliver_conf_dir_overrides_home(tmp_path):
    home, conf = tmp_path / "home", tmp_path / "conf"
    _write(conf / "mcp.json", {"mcpServers": {"from-conf": _stdio()}})
    _write(home / ".cliver" / "mcp.json", {"mcpServers": {"from-home": _stdio()}})

    result = _discover(tmp_path, home=home, env={"CLIVER_CONF_DIR": str(conf)})

    assert list(result.servers) == ["from-conf"]
    assert result.provenance["from-conf"].path == conf / "mcp.json"
    # drop-ins follow the overridden root too
    _write(conf / "mcps" / "extra.json", _stdio())
    assert list(_discover(tmp_path, home=home, env={"CLIVER_CONF_DIR": str(conf)}).servers) == ["from-conf", "extra"]


def test_dropin_directory_ignores_hidden_and_non_json(tmp_path):
    dropins = tmp_path / "home" / ".cliver" / "mcps"
    _write(dropins / "a.json", _stdio())
    _write(dropins / ".hidden.json", _stdio())
    _write(dropins / "notes.txt", _stdio())
    _write(dropins / "nested" / "c.json", _stdio())

    result = _discover(tmp_path)

    assert list(result.servers) == ["a"]


def test_extra_wins_and_shadowing_is_reported(tmp_path):
    home, team_dir = tmp_path / "home", tmp_path / "team"
    _write(home / ".cliver" / "mcp.json", {"mcpServers": {"shared": _stdio("user-cmd")}})
    _write(team_dir / "team.json", {"mcpServers": {"shared": _stdio("path-cmd"), "team-only": _stdio()}})

    result = _discover(
        tmp_path,
        home=home,
        extra={"shared": _stdio("explicit-cmd")},
        extra_paths=[team_dir],
    )

    assert list(result.servers) == ["shared", "team-only"]
    assert result.servers["shared"]["command"] == "explicit-cmd"
    assert result.provenance["shared"].source == "explicit"
    assert result.provenance["shared"].path is None
    assert result.provenance["team-only"].source == "extra_paths"

    shadowed = [problem for problem in result.problems if problem.code == "shadowed"]
    assert len(shadowed) == 2  # extra_paths vs explicit, cliver_user vs explicit
    assert {problem.server for problem in shadowed} == {"shared"}
    assert all(problem.severity == "info" and "explicit" in problem.message for problem in shadowed)


def test_project_store_shadows_user_store(tmp_path):
    project_root, home = tmp_path / "project", tmp_path / "home"
    _write(project_root / ".cliver" / "mcp.json", {"mcpServers": {"dup": _stdio("project-cmd")}})
    _write(home / ".cliver" / "mcp.json", {"mcpServers": {"dup": _stdio("home-cmd")}})

    result = _discover(tmp_path, cwd=project_root, home=home)

    assert result.servers["dup"]["command"] == "project-cmd"
    (problem,) = [p for p in result.problems if p.code == "shadowed"]
    assert "cliver_user" in problem.message


def test_extra_paths_file_directory_and_missing_path(tmp_path):
    single = _write(tmp_path / "single.json", _stdio("single-cmd"))
    bundle_dir = tmp_path / "bundle-dir"
    _write(bundle_dir / "one.json", {"mcpServers": {"one": _stdio()}})
    _write(bundle_dir / "two.json", _stdio("two-cmd"))
    missing = tmp_path / "nope.json"

    result = _discover(tmp_path, extra_paths=[single, bundle_dir, missing])

    assert list(result.servers) == ["single", "one", "two"]
    assert all(result.provenance[name].source == "extra_paths" for name in result.servers)
    (problem,) = [p for p in result.problems if p.code == "missing-path"]
    assert problem.path == missing


def test_disabled_entry_is_excluded_and_explained(tmp_path):
    home = tmp_path / "home"
    _write(
        home / ".cliver" / "mcp.json",
        {"mcpServers": {"on": _stdio(), "off": {"enabled": False, "transport": "nonsense"}}},
    )

    result = _discover(tmp_path, home=home)

    assert list(result.servers) == ["on"]
    (problem,) = result.problems  # disabled check runs before any validation
    assert (problem.code, problem.severity, problem.server) == ("disabled", "info", "off")


def test_invalid_entries_are_skipped_with_problems(tmp_path):
    home = tmp_path / "home"
    entries = {
        "bad_transport": {"transport": "sse", "url": "https://x/mcp"},
        "bad__name": _stdio(),
        "bad name": _stdio(),
        "": _stdio(),
        "not_object": "npx",
        "no_command": {"transport": "stdio", "args": []},
        "no_url": {"transport": "streamable-http"},
        "bad_args": {"transport": "stdio", "command": "npx", "args": "x"},
        "bad_env": {"transport": "stdio", "command": "npx", "env": [1]},
        "bad_cwd": {"transport": "stdio", "command": "npx", "cwd": 7},
        "ok": _stdio(),
    }
    _write(home / ".cliver" / "mcp.json", {"mcpServers": entries})

    result = _discover(tmp_path, home=home)

    assert list(result.servers) == ["ok"]
    offenders = {problem.server for problem in result.problems}
    assert offenders == set(entries) - {"ok"}
    assert all(problem.code == "invalid-entry" for problem in result.problems)


def test_lenient_fields_are_ignored_with_problems(tmp_path):
    home = tmp_path / "home"
    _write(
        home / ".cliver" / "mcp.json",
        {
            "mcpServers": {
                "s": {
                    "transport": "stdio",
                    "command": "npx",
                    "timeout": "soon",
                    "startup_timeout": -1,
                    "mode": "fast",
                    "enabled_tools": "echo",
                    "disabled_tools": [1, 2],
                }
            }
        },
    )

    result = _discover(tmp_path, home=home)

    assert result.servers["s"] == {"transport": "stdio", "command": "npx"}
    assert [problem.code for problem in result.problems] == ["invalid-entry"] * 5
    assert all(problem.server == "s" for problem in result.problems)


def test_optional_fields_are_carried(tmp_path):
    home = tmp_path / "home"
    _write(
        home / ".cliver" / "mcp.json",
        {
            "mcpServers": {
                "full": {
                    "transport": "stdio",
                    "command": "npx",
                    "args": ["-y", "pkg"],
                    "env": {"KEY": "value"},
                    "cwd": "/tmp",
                    "timeout": 5,
                    "startup_timeout": None,
                    "mode": "legacy",
                    "enabled_tools": ["echo"],
                    "disabled_tools": ["rm"],
                },
                "http": {"transport": "streamable-http", "url": "https://x/mcp", "headers": {"X": "y"}},
            }
        },
    )

    result = _discover(tmp_path, home=home)

    assert result.servers["full"] == {
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "pkg"],
        "env": {"KEY": "value"},
        "cwd": "/tmp",
        "timeout": 5.0,
        "startup_timeout": None,
        "mode": "legacy",
        "enabled_tools": ["echo"],
        "disabled_tools": ["rm"],
    }
    assert result.servers["http"] == {
        "transport": "streamable-http",
        "url": "https://x/mcp",
        "headers": {"X": "y"},
    }


def test_variable_expansion_and_unresolved_names(tmp_path):
    secret = "sk-super-secret"
    _write(
        tmp_path / "home" / ".cliver" / "mcp.json",
        {
            "mcpServers": {
                "s": {
                    "transport": "stdio",
                    "command": "${CLIVER_TEST_CMD:-npx}",
                    "args": ["${CLIVER_TEST_TOKEN}", "${MISSING:-fallback}", "${ALSO_MISSING}"],
                    "env": {"TOKEN": "${CLIVER_TEST_TOKEN}"},
                }
            }
        },
    )

    result = _discover(tmp_path, env={"CLIVER_TEST_TOKEN": secret})

    config = result.servers["s"]
    assert config["command"] == "npx"
    assert config["args"] == [secret, "fallback", "${ALSO_MISSING}"]  # unresolved stays literal
    assert config["env"] == {"TOKEN": secret}

    (problem,) = result.problems
    assert problem.code == "unresolved-variable"
    assert "ALSO_MISSING" in problem.message  # name only, never the value
    assert secret not in problem.message
    assert secret not in result.summary()


def test_redacted_masks_secret_values(tmp_path):
    _write(
        tmp_path / "home" / ".cliver" / "mcp.json",
        {
            "mcpServers": {
                "s": {"transport": "stdio", "command": "npx", "env": {"TOKEN": "abc"}},
                "h": {
                    "transport": "streamable-http",
                    "url": "https://x/mcp",
                    "headers": {"Authorization": "Bearer tok"},
                },
            }
        },
    )

    result = _discover(tmp_path)

    redacted = result.redacted()
    assert redacted["s"]["env"] == {"TOKEN": "***"}
    assert redacted["h"]["headers"] == {"Authorization": "***"}
    assert redacted["h"]["url"] == "https://x/mcp"
    assert redacted["s"]["command"] == "npx"


def test_parse_error_is_isolated(tmp_path):
    project_root, home = tmp_path / "project", tmp_path / "home"
    bad = project_root / ".cliver" / "mcp.json"
    bad.parent.mkdir(parents=True)
    bad.write_text("{ not json", encoding="utf-8")
    _write(home / ".cliver" / "mcp.json", {"mcpServers": {"ok": _stdio()}})

    result = _discover(tmp_path, cwd=project_root, home=home)

    assert list(result.servers) == ["ok"]  # other sources unaffected
    (problem,) = [p for p in result.problems if p.code == "parse-error"]
    assert problem.severity == "error"
    assert problem.path == bad
    assert ("cliver_project", bad) in result.sources_read


def test_unrecognized_files_are_reported(tmp_path):
    home = tmp_path / "home"
    _write(home / ".cliver" / "mcp.json", _stdio())  # canonical store must be a bundle
    _write(home / ".cliver" / "mcps" / "d1.json", [1, 2, 3])
    _write(home / ".cliver" / "mcps" / "d2.json", {"mcpServers": 5})

    result = _discover(tmp_path, home=home)

    assert result.servers == {}
    assert [problem.code for problem in result.problems] == ["unrecognized-file"] * 3
    assert any("mcpServers" in problem.message for problem in result.problems)
    assert any(problem.path == home / ".cliver" / "mcp.json" for problem in result.problems)


def test_extra_configs_are_not_mutated(tmp_path):
    source = {"transport": "stdio", "command": "npx", "args": ["-y", "pkg"]}
    result = _discover(tmp_path, extra={"srv": source})

    assert source == {"transport": "stdio", "command": "npx", "args": ["-y", "pkg"]}
    assert result.servers["srv"]["args"] == ["-y", "pkg"]
    assert result.servers["srv"]["args"] is not source["args"]


def test_empty_discovery(tmp_path):
    result = _discover(tmp_path)

    assert result.servers == {}
    assert result.provenance == {}
    assert result.problems == ()
    assert result.sources_read == ()
    assert result.summary() == "0 servers from 0 sources"
    assert result.redacted() == {}
