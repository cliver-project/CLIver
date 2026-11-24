"""Read MCP server definitions from other agents' configuration files.

Phase 3 of the CLI configuration.  cliver-core's discovery reads only
cliver's own schema (``transport`` required, ``stdio`` or
``streamable-http``); the files other agents write use ``type``/``url``/
``httpUrl``/``servers`` and sometimes infer the transport.  This module
normalizes those files into core's schema, in memory.

Shared sources are opt-in.  ``mcp.shared`` selects known user-level files
(``claude``, ``cursor``, ``gemini``, ``qwen``, ``windsurf``,
``claude-desktop``, ``continue``, ``zed``, ``antigravity``);
``mcp.shared_project`` adds project-level files for the selected families;
``mcp.shared_sources`` adds explicit JSON paths.  Nothing is read without
one of those settings.

Known quirks handled here:

* Claude Code: user/local definitions live in ``~/.claude.json`` (top-level
  ``mcpServers`` and ``projects["<cwd>"].mcpServers``); project ``.mcp.json``
  uses ``mcpServers``.  ``type`` is ``stdio``/``http``/``sse``.
* Cursor: ``~/.cursor/mcp.json`` and ``.cursor/mcp.json``; entries usually
  infer stdio from ``command`` or remote from ``url``.
* Gemini CLI and Qwen Code: ``settings.json`` entries use ``httpUrl`` for
  Streamable HTTP and ``url`` for SSE (unsupported — skipped with a warning),
  ``timeout``/``discoveryTimeoutMs`` in milliseconds, and
  ``includeTools``/``excludeTools``.
* VS Code/Copilot: ``.vscode/mcp.json`` uses ``servers`` and ``type``;
  ``${input:...}`` prompts cannot be answered here, so such a server is
  skipped with a warning.
* Windsurf: ``mcp_config.json`` may use ``serverUrl`` for remote servers.
* Zed: ``settings.json`` uses ``context_servers``.

Placeholders: ``${workspaceFolder}``, ``${workspaceFolderBasename}``,
``${userHome}``, ``${env:VAR}``, and ``${VAR}``/``${VAR:-default}`` are
expanded when possible; an unresolved ``${VAR}`` stays literal and is
reported by name (never with its value).  Secrets are never logged.
"""

from __future__ import annotations

import json
import os
import platform
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cliver_core.mcp import MCPDiscoveryProblem, MCPServerProvenance

#: User-level source IDs, in merge order (later sources win).
USER_SOURCE_IDS = (
    "claude",
    "cursor",
    "gemini",
    "qwen",
    "windsurf",
    "claude-desktop",
    "continue",
    "zed",
    "antigravity",
)

#: Project-level source IDs, in merge order (later sources win).
PROJECT_SOURCE_IDS = (
    "claude-project",
    "cursor-project",
    "gemini-project",
    "qwen-project",
    "vscode-project",
    "zed-project",
)

SHARED_MCP_SOURCE_IDS = frozenset((*USER_SOURCE_IDS, *PROJECT_SOURCE_IDS))

_PROJECT_BY_FAMILY = {
    "claude": "claude-project",
    "cursor": "cursor-project",
    "gemini": "gemini-project",
    "qwen": "qwen-project",
    "vscode": "vscode-project",
    "zed": "zed-project",
}

_VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")
_INPUT_RE = re.compile(r"\$\{input:[^}]*\}")

__all__ = [
    "PROJECT_SOURCE_IDS",
    "SHARED_MCP_SOURCE_IDS",
    "USER_SOURCE_IDS",
    "SharedMCPServers",
    "load_shared_mcp_servers",
]


@dataclass(frozen=True)
class SharedMCPServers:
    """Normalized servers from other agents' files, below cliver's own."""

    servers: dict[str, dict] = field(default_factory=dict)
    provenance: dict[str, MCPServerProvenance] = field(default_factory=dict)
    problems: tuple[MCPDiscoveryProblem, ...] = ()
    sources_read: tuple[tuple[str, Path], ...] = ()


@dataclass(frozen=True)
class _SharedFile:
    source: str
    path: Path
    key: str = "auto"
    scope: str = "user"
    claude_project: bool = False
    gemini_like: bool = False
    base_dir: Path | None = None


def load_shared_mcp_servers(
    *,
    selected: tuple[str, ...] | None,
    include_project: bool,
    explicit_paths: Iterable[str | os.PathLike[str]] = (),
    cwd: Path,
    home: Path,
    env: Mapping[str, str],
) -> SharedMCPServers:
    """Read the opted-in shared files and normalize their servers.

    ``selected=None`` means every user-level source; an empty tuple means
    none.  A non-empty tuple selects those source IDs (user or project).
    ``include_project`` adds project-level variants for the selected
    families, or every project source when nothing was selected.
    """
    files = _shared_files(
        selected=selected,
        include_project=include_project,
        explicit_paths=explicit_paths,
        cwd=cwd,
        home=home,
        env=env,
    )
    servers: dict[str, dict] = {}
    provenance: dict[str, MCPServerProvenance] = {}
    problems: list[MCPDiscoveryProblem] = []
    sources_read: list[tuple[str, Path]] = []

    for shared_file in files:
        _read_file(shared_file, servers, provenance, problems, sources_read, cwd=cwd, home=home, env=env)

    return SharedMCPServers(
        servers=servers,
        provenance=provenance,
        problems=tuple(problems),
        sources_read=tuple(sources_read),
    )


# ── Source registry ────────────────────────────────────────


def _shared_files(
    *,
    selected: tuple[str, ...] | None,
    include_project: bool,
    explicit_paths: Iterable[str | os.PathLike[str]],
    cwd: Path,
    home: Path,
    env: Mapping[str, str],
) -> list[_SharedFile]:
    chosen = _chosen_ids(selected, include_project)
    files: list[_SharedFile] = []

    for source_id in USER_SOURCE_IDS:
        if source_id in chosen:
            files.extend(_user_files(source_id, home=home, env=env))
    for source_id in PROJECT_SOURCE_IDS:
        if source_id in chosen:
            files.extend(_project_files(source_id, cwd=cwd, home=home, env=env))

    for entry in explicit_paths:
        path = Path(entry).expanduser()
        files.append(
            _SharedFile(
                source="explicit",
                path=path,
                key="auto",
                scope="explicit",
                base_dir=path.parent,
            )
        )
    return files


def _chosen_ids(selected: tuple[str, ...] | None, include_project: bool) -> set[str]:
    if selected is None:
        chosen = set(USER_SOURCE_IDS)
    else:
        chosen = set(selected)

    if include_project:
        if selected is None or not chosen:
            chosen.update(PROJECT_SOURCE_IDS)
        else:
            for source_id in list(chosen):
                family = source_id.split("-project", 1)[0]
                project_id = _PROJECT_BY_FAMILY.get(family)
                if project_id:
                    chosen.add(project_id)
    return chosen


def _user_files(source_id: str, *, home: Path, env: Mapping[str, str]) -> list[_SharedFile]:
    if source_id == "claude":
        return [
            _SharedFile(
                source="claude",
                path=_claude_state_path(home=home, env=env),
                key="mcpServers",
                base_dir=None,
            )
        ]
    if source_id == "cursor":
        return [_SharedFile(source="cursor", path=home / ".cursor" / "mcp.json", base_dir=None)]
    if source_id == "gemini":
        return [
            _SharedFile(
                source="gemini",
                path=home / ".gemini" / "settings.json",
                key="mcpServers",
                gemini_like=True,
            )
        ]
    if source_id == "qwen":
        return [
            _SharedFile(
                source="qwen",
                path=home / ".qwen" / "settings.json",
                key="mcpServers",
                gemini_like=True,
            )
        ]
    if source_id == "windsurf":
        return [
            _SharedFile(
                source="windsurf",
                path=home / ".codeium" / "windsurf" / "mcp_config.json",
                key="mcpServers",
            )
        ]
    if source_id == "claude-desktop":
        return [
            _SharedFile(source="claude-desktop", path=path, key="mcpServers")
            for path in _claude_desktop_paths(home=home, env=env)
        ]
    if source_id == "continue":
        directory = home / ".continue" / "mcpServers"
        if not directory.is_dir():
            return []
        return [
            _SharedFile(source="continue", path=path, key="auto", base_dir=None)
            for path in sorted(directory.glob("*.json"))
            if path.is_file() and not path.name.startswith(".")
        ]
    if source_id == "zed":
        return [
            _SharedFile(
                source="zed",
                path=home / ".config" / "zed" / "settings.json",
                key="context_servers",
            )
        ]
    if source_id == "antigravity":
        return [
            _SharedFile(source="antigravity", path=path, key="mcpServers")
            for path in (
                home / ".gemini" / "config" / "mcp_config.json",
                home / ".gemini" / "antigravity" / "mcp_config.json",
                home / ".gemini" / "antigravity-ide" / "mcp_config.json",
            )
        ]
    return []


def _project_files(source_id: str, *, cwd: Path, home: Path, env: Mapping[str, str]) -> list[_SharedFile]:
    if source_id == "claude-project":
        return [
            _SharedFile(source="claude-project", path=cwd / ".mcp.json", key="mcpServers", scope="project"),
            _SharedFile(
                source="claude-project",
                path=_claude_state_path(home=home, env=env),
                key="mcpServers",
                scope="project",
                claude_project=True,
            ),
        ]
    if source_id == "cursor-project":
        return [_SharedFile(source="cursor-project", path=cwd / ".cursor" / "mcp.json", scope="project")]
    if source_id == "gemini-project":
        return [
            _SharedFile(
                source="gemini-project",
                path=cwd / ".gemini" / "settings.json",
                key="mcpServers",
                scope="project",
                gemini_like=True,
            )
        ]
    if source_id == "qwen-project":
        return [
            _SharedFile(
                source="qwen-project",
                path=cwd / ".qwen" / "settings.json",
                key="mcpServers",
                scope="project",
                gemini_like=True,
            )
        ]
    if source_id == "vscode-project":
        return [
            _SharedFile(
                source="vscode-project",
                path=cwd / ".vscode" / "mcp.json",
                key="servers",
                scope="project",
            )
        ]
    if source_id == "zed-project":
        return [
            _SharedFile(
                source="zed-project",
                path=cwd / ".zed" / "settings.json",
                key="context_servers",
                scope="project",
            )
        ]
    return []


def _claude_state_path(*, home: Path, env: Mapping[str, str]) -> Path:
    override = env.get("CLAUDE_CONFIG_DIR")
    return (Path(override).expanduser() if override else home) / ".claude.json"


def _claude_desktop_paths(*, home: Path, env: Mapping[str, str]) -> list[Path]:
    xdg = env.get("XDG_CONFIG_HOME") or str(home / ".config")
    paths = [
        home / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json",
        Path(xdg) / "Claude" / "claude_desktop_config.json",
    ]
    appdata = env.get("APPDATA")
    if appdata:
        paths.append(Path(appdata) / "Claude" / "claude_desktop_config.json")
    elif platform.system() == "Windows":
        paths.append(home / "AppData" / "Roaming" / "Claude" / "claude_desktop_config.json")
    return paths


# ── Reading and extraction ─────────────────────────────────


def _read_file(
    shared_file: _SharedFile,
    servers: dict[str, dict],
    provenance: dict[str, MCPServerProvenance],
    problems: list[MCPDiscoveryProblem],
    sources_read: list[tuple[str, Path]],
    *,
    cwd: Path,
    home: Path,
    env: Mapping[str, str],
) -> None:
    path = shared_file.path
    if not path.is_file():
        if shared_file.scope == "explicit":
            _problem(problems, "missing-path", "file does not exist", "warning", shared_file, path=path)
        return
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        _problem(problems, "unreadable", f"unreadable: {exc}", "warning", shared_file, path=path)
        return
    try:
        raw = json.loads(text)
    except ValueError as exc:
        severity = "error" if shared_file.scope == "explicit" else "warning"
        _problem(problems, "parse-error", f"invalid JSON: {exc}", severity, shared_file, path=path)
        return
    if not isinstance(raw, Mapping):
        _problem(
            problems, "unrecognized-file", "the top level must be a JSON object", "warning", shared_file, path=path
        )
        return

    sources_read.append((shared_file.source, path))
    entries = _extract_entries(raw, shared_file, cwd=cwd, problems=problems)
    for name, entry in entries:
        normalized = _normalize_entry(
            name,
            entry,
            shared_file,
            cwd=cwd,
            home=home,
            env=env,
            problems=problems,
        )
        if normalized is None:
            continue
        server_name, config = normalized
        previous = provenance.get(server_name)
        if previous is not None:
            _problem(
                problems,
                "shadowed",
                f"'{server_name}' from {_location(shared_file, path)} overrides {previous.source} ({previous.path})",
                "info",
                shared_file,
                path=path,
                server=server_name,
            )
        servers[server_name] = config
        provenance[server_name] = MCPServerProvenance(server_name, shared_file.source, path)


def _extract_entries(
    raw: Mapping[str, Any],
    shared_file: _SharedFile,
    *,
    cwd: Path,
    problems: list[MCPDiscoveryProblem],
) -> list[tuple[str, Any]]:
    if shared_file.claude_project:
        projects = raw.get("projects")
        if not isinstance(projects, Mapping):
            return []
        for key in (str(cwd), str(cwd.resolve()), os.path.normpath(str(cwd))):
            project = projects.get(key)
            if isinstance(project, Mapping) and isinstance(project.get("mcpServers"), Mapping):
                return [(str(name), entry) for name, entry in project["mcpServers"].items()]
        return []

    if shared_file.key == "auto":
        for key in ("mcpServers", "servers", "context_servers"):
            value = raw.get(key)
            if isinstance(value, Mapping):
                return [(str(name), entry) for name, entry in value.items()]
        if _looks_like_server(raw):
            return [(shared_file.path.stem, raw)]
        return []

    value = raw.get(shared_file.key)
    if isinstance(value, Mapping):
        return [(str(name), entry) for name, entry in value.items()]
    if value is not None:
        _problem(
            problems,
            "unrecognized-file",
            f"'{shared_file.key}' must be an object",
            "warning",
            shared_file,
            path=shared_file.path,
        )
    return []


def _looks_like_server(raw: Mapping[str, Any]) -> bool:
    return any(key in raw for key in ("command", "url", "httpUrl", "serverUrl", "transport", "type"))


def _normalize_entry(
    name: str,
    raw: Any,
    shared_file: _SharedFile,
    *,
    cwd: Path,
    home: Path,
    env: Mapping[str, str],
    problems: list[MCPDiscoveryProblem],
) -> tuple[str, dict] | None:
    if not name or "__" in name or any(char.isspace() for char in name):
        _problem(
            problems,
            "invalid-entry",
            "server names must be non-empty, contain no '__', and contain no whitespace",
            "warning",
            shared_file,
            path=shared_file.path,
            server=name,
        )
        return None
    if not isinstance(raw, Mapping):
        _problem(
            problems,
            "invalid-entry",
            "server entry must be an object",
            "warning",
            shared_file,
            path=shared_file.path,
            server=name,
        )
        return None
    if raw.get("enabled", True) is False or raw.get("disabled") is True:
        _problem(
            problems,
            "disabled",
            "excluded: the entry is disabled",
            "info",
            shared_file,
            path=shared_file.path,
            server=name,
        )
        return None
    if _uses_input_placeholder(raw):
        _problem(
            problems,
            "unsupported-input",
            "uses ${input:...}, which cliver cannot prompt for; server skipped",
            "warning",
            shared_file,
            path=shared_file.path,
            server=name,
        )
        return None

    transport = _transport(raw, shared_file)
    if transport is None:
        explicit = raw.get("type", raw.get("transport"))
        if explicit is not None and str(explicit).lower() == "sse":
            reason = "SSE is not supported by cliver"
        elif explicit is not None:
            reason = f"unsupported transport {explicit!r}"
        elif shared_file.gemini_like and raw.get("url"):
            reason = "Gemini/Qwen 'url' means SSE, which cliver does not support; use 'httpUrl'"
        else:
            reason = "no usable command/url"
        _problem(
            problems,
            "invalid-entry",
            f"skipped: {reason}",
            "warning",
            shared_file,
            path=shared_file.path,
            server=name,
        )
        return None

    config: dict[str, Any] = {"transport": transport}

    def text(value: Any, field_name: str) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            _problem(
                problems,
                "invalid-entry",
                f"'{field_name}' must be a string",
                "warning",
                shared_file,
                path=shared_file.path,
                server=name,
            )
            return None
        return _prepare_text(value, shared_file, cwd=cwd, home=home, env=env, problems=problems, server=name)

    def string_map(value: Any, field_name: str) -> dict[str, str] | None:
        if value is None:
            return None
        if not isinstance(value, Mapping):
            _problem(
                problems,
                "invalid-entry",
                f"'{field_name}' must be an object",
                "warning",
                shared_file,
                path=shared_file.path,
                server=name,
            )
            return None
        result: dict[str, str] = {}
        for item_key, item_value in value.items():
            prepared = text(str(item_value), f"{field_name}.{item_key}")
            if prepared is None:
                return None
            result[str(item_key)] = prepared
        return result

    if transport == "stdio":
        command = text(raw.get("command"), "command")
        if not command:
            _problem(
                problems,
                "invalid-entry",
                "stdio transport requires a non-empty 'command'",
                "warning",
                shared_file,
                path=shared_file.path,
                server=name,
            )
            return None
        config["command"] = command
        args = raw.get("args")
        if args is not None:
            if not isinstance(args, list):
                _problem(
                    problems,
                    "invalid-entry",
                    "'args' must be a list",
                    "warning",
                    shared_file,
                    path=shared_file.path,
                    server=name,
                )
                return None
            prepared_args: list[str] = []
            for item in args:
                prepared = text(str(item), "args")
                if prepared is None:
                    return None
                prepared_args.append(prepared)
            config["args"] = prepared_args
        env_cfg = string_map(raw.get("env"), "env")
        if raw.get("env") is not None and env_cfg is None:
            return None
        if env_cfg:
            config["env"] = env_cfg
        cwd_cfg = text(raw.get("cwd"), "cwd")
        if cwd_cfg:
            cwd_path = Path(cwd_cfg).expanduser()
            base = shared_file.base_dir or cwd
            config["cwd"] = str(cwd_path if cwd_path.is_absolute() else (base / cwd_path).resolve())
    else:
        url = text(raw.get("httpUrl") or raw.get("url") or raw.get("serverUrl"), "url")
        if not url:
            _problem(
                problems,
                "invalid-entry",
                "streamable-http transport requires a non-empty 'url'",
                "warning",
                shared_file,
                path=shared_file.path,
                server=name,
            )
            return None
        config["url"] = url
        headers = string_map(raw.get("headers"), "headers")
        if raw.get("headers") is not None and headers is None:
            return None
        if headers:
            config["headers"] = headers

    _copy_filters(raw, config, shared_file, problems, name)
    _copy_timing(raw, config, shared_file, problems, name)
    return name, config


def _transport(raw: Mapping[str, Any], shared_file: _SharedFile) -> str | None:
    explicit = raw.get("transport", raw.get("type"))
    if explicit is not None:
        value = str(explicit).strip().lower()
        if value == "stdio":
            return "stdio"
        if value in {"http", "httpurl", "streamable-http", "streamable_http", "streamablehttp"}:
            return "streamable-http"
        if value == "sse":
            return None
        return None
    if raw.get("command"):
        return "stdio"
    if raw.get("httpUrl"):
        return "streamable-http"
    if raw.get("url"):
        # Gemini/Qwen use ``url`` for SSE and ``httpUrl`` for Streamable HTTP.
        return None if shared_file.gemini_like else "streamable-http"
    if raw.get("serverUrl"):
        return "streamable-http"
    return None


def _copy_filters(
    raw: Mapping[str, Any],
    config: dict[str, Any],
    shared_file: _SharedFile,
    problems: list[MCPDiscoveryProblem],
    name: str,
) -> None:
    enabled = raw.get("enabled_tools", raw.get("includeTools"))
    disabled = raw.get("disabled_tools", raw.get("excludeTools"))
    for key, value in (("enabled_tools", enabled), ("disabled_tools", disabled)):
        if value is None:
            continue
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            config[key] = list(value)
        else:
            _problem(
                problems,
                "invalid-entry",
                f"'{key}' must be a list of tool names; ignored",
                "warning",
                shared_file,
                path=shared_file.path,
                server=name,
            )


def _copy_timing(
    raw: Mapping[str, Any],
    config: dict[str, Any],
    shared_file: _SharedFile,
    problems: list[MCPDiscoveryProblem],
    name: str,
) -> None:
    # Gemini/Qwen timeouts are milliseconds; an explicit shared file is
    # assumed to use cliver's seconds.  Other agents' units are not portable,
    # so their ``timeout`` is left to cliver's default.
    timeout = raw.get("timeout")
    if timeout is not None and (shared_file.gemini_like or shared_file.scope == "explicit"):
        seconds = _seconds(timeout, milliseconds=shared_file.gemini_like)
        if seconds is None:
            _problem(
                problems,
                "invalid-entry",
                "'timeout' must be a number; ignored",
                "warning",
                shared_file,
                path=shared_file.path,
                server=name,
            )
        else:
            config["timeout"] = seconds

    discovery_timeout = raw.get("discoveryTimeoutMs")
    if discovery_timeout is not None:
        seconds = _seconds(discovery_timeout, milliseconds=True)
        if seconds is None:
            _problem(
                problems,
                "invalid-entry",
                "'discoveryTimeoutMs' must be a number; ignored",
                "warning",
                shared_file,
                path=shared_file.path,
                server=name,
            )
        else:
            config["startup_timeout"] = seconds

    startup_timeout = raw.get("startup_timeout")
    if startup_timeout is not None and shared_file.scope == "explicit":
        seconds = _seconds(startup_timeout, milliseconds=False)
        if seconds is None:
            _problem(
                problems,
                "invalid-entry",
                "'startup_timeout' must be a number; ignored",
                "warning",
                shared_file,
                path=shared_file.path,
                server=name,
            )
        else:
            config["startup_timeout"] = seconds

    mode = raw.get("mode", raw.get("versionNegotiation"))
    if mode in {"auto", "legacy"}:
        config["mode"] = mode

    pool_size = raw.get("session_pool_size")
    if isinstance(pool_size, int) and not isinstance(pool_size, bool) and pool_size >= 0:
        config["session_pool_size"] = pool_size


def _seconds(value: Any, *, milliseconds: bool) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        return None
    seconds = float(value) / 1000.0 if milliseconds else float(value)
    return seconds


# ── Placeholders ───────────────────────────────────────────


def _uses_input_placeholder(raw: Mapping[str, Any]) -> bool:
    """Only the fields that can reach a process/request are checked."""
    for key in ("command", "args", "env", "cwd", "url", "httpUrl", "serverUrl", "headers"):
        if key in raw and _contains_input_placeholder(raw[key]):
            return True
    return False


def _contains_input_placeholder(value: Any) -> bool:
    if isinstance(value, str):
        return bool(_INPUT_RE.search(value))
    if isinstance(value, Mapping):
        return any(_contains_input_placeholder(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_input_placeholder(item) for item in value)
    return False


def _prepare_text(
    value: str,
    shared_file: _SharedFile,
    *,
    cwd: Path,
    home: Path,
    env: Mapping[str, str],
    problems: list[MCPDiscoveryProblem],
    server: str,
) -> str:
    value = value.replace("${workspaceFolderBasename}", cwd.name)
    value = value.replace("${workspaceFolder}", str(cwd))
    value = value.replace("${workspaceRoot}", str(cwd))
    value = value.replace("${userHome}", str(home))
    value = _ENV_PLACEHOLDER_RE.sub(lambda match: "${" + match.group(1) + "}", value)
    return _expand_vars(value, env, shared_file, problems, server)


_ENV_PLACEHOLDER_RE = re.compile(r"\$\{env:([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand_vars(
    value: str,
    env: Mapping[str, str],
    shared_file: _SharedFile,
    problems: list[MCPDiscoveryProblem],
    server: str,
) -> str:
    missing: list[str] = []

    def replace(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        if name in env:
            return env[name]
        if default is not None:
            return default
        missing.append(name)
        return match.group(0)

    expanded = _VAR_RE.sub(replace, value)
    for name in missing:
        _problem(
            problems,
            "unresolved-variable",
            f"environment variable '{name}' is not set; the placeholder was kept literal",
            "warning",
            shared_file,
            path=shared_file.path,
            server=server,
        )
    return expanded


def _problem(
    problems: list[MCPDiscoveryProblem],
    code: str,
    message: str,
    severity: str,
    shared_file: _SharedFile,
    *,
    path: Path | None = None,
    server: str | None = None,
) -> None:
    problems.append(
        MCPDiscoveryProblem(
            code=code,
            message=message,
            severity=severity,
            source=shared_file.source,
            path=path or shared_file.path,
            server=server,
        )
    )


def _location(shared_file: _SharedFile, path: Path) -> str:
    return f"{shared_file.source} ({path})"
