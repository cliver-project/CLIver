"""MCP server discovery from cliver's own places — stdlib-only, in-memory.

Reads cliver's own MCP configuration at call time and returns ready-to-use
server configs for :class:`~cliver_core.mcp.MCPClient`.  Nothing is written
or persisted; a discovery result lives in memory only, and core never
writes config files.

Sources, in precedence order (the first definition of a name wins):

0. ``extra``               caller-supplied configs (highest priority)
1. ``extra_paths``         caller-supplied files/directories — the opt-in
                           seam for places beyond cliver's own
2. ``cliver_project``      ``<cwd>/.cliver/mcp.json``
3. ``cliver_user``         ``$CLIVER_CONF_DIR/mcp.json`` else ``~/.cliver/mcp.json``
4. ``cliver_user_dropin``  ``<root>/mcps/*.json`` (sorted, non-hidden)

Store format (strict)::

    {"mcpServers": {
      "everything": {"transport": "stdio", "command": "npx",
                     "args": ["-y", "@modelcontextprotocol/server-everything"]},
      "docs":       {"transport": "streamable-http", "url": "https://…",
                     "headers": {"Authorization": "Bearer ${DOCS_TOKEN}"}}
    }}

``transport`` is required and is exactly ``stdio`` or ``streamable-http``.
stdio servers need ``command`` (``args``/``env``/``cwd`` optional);
streamable-http servers need ``url`` (``headers`` optional).  Optional per
server: ``enabled`` (``false`` excludes it, with a problem explaining why),
``timeout`` (per-call read timeout, seconds), ``startup_timeout`` (connect
bound; default 60 s stdio / 30 s remote, explicit ``null`` disables),
``session_pool_size`` (streamable-http only: max concurrent pooled tool
calls; absent or ``0`` keeps every call ephemeral),
``mode`` (``auto`` default; ``legacy`` skips the 2026-07-28 probe),
``enabled_tools``/``disabled_tools`` (tool-name filters — every tool schema
goes to the model).  Strings expand ``${VAR}`` and ``${VAR:-default}`` from
the environment; an unresolved name stays literal and is reported by name
only (discovered values are never logged).

Canonical files (``mcp.json``) must contain a ``mcpServers`` object;
drop-ins and ``extra_paths`` files may instead be a single server object
named after the file stem.  Server names must be non-empty, contain no
``__`` (model-facing tool names are ``mcp__<server>__<tool>``) and no
whitespace.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

logger = logging.getLogger(__name__)

TRANSPORTS = ("stdio", "streamable-http")
_MODES = ("auto", "legacy")
_VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")
_SECRET_SECTIONS = ("env", "headers")


@dataclass(frozen=True)
class MCPDiscoveryProblem:
    """One thing discovery did to (or with) a config entry.

    ``severity`` is ``info`` (expected filtering, e.g. shadowed/disabled),
    ``warning`` (entry skipped or value coerced), or ``error`` (file-level
    failure, e.g. malformed JSON).
    """

    code: str
    message: str
    severity: str = "warning"
    source: str = ""
    path: Path | None = None
    server: str | None = None


@dataclass(frozen=True)
class MCPServerProvenance:
    """Where a discovered server was defined."""

    name: str
    source: str
    path: Path | None


@dataclass(frozen=True)
class MCPDiscoveryResult:
    """In-memory discovery result.  ``servers`` feeds ``MCPClient(servers)``."""

    servers: dict[str, dict]
    provenance: dict[str, MCPServerProvenance]
    problems: tuple[MCPDiscoveryProblem, ...]
    sources_read: tuple[tuple[str, Path], ...]

    def summary(self) -> str:
        count = len(self.servers)
        sources = len({source for source, _ in self.sources_read})
        text = f"{count} server{'s' if count != 1 else ''} from {sources} source{'s' if sources != 1 else ''}"
        problems = len(self.problems)
        if problems:
            text += f"; {problems} problem{'s' if problems != 1 else ''}"
        return text

    def redacted(self) -> dict[str, dict]:
        """A config copy for display — ``env``/``headers`` values masked."""
        out: dict[str, dict] = {}
        for name, config in self.servers.items():
            clean = dict(config)
            if isinstance(clean.get("args"), list):
                clean["args"] = list(clean["args"])
            for section in _SECRET_SECTIONS:
                values = clean.get(section)
                if isinstance(values, dict) and values:
                    clean[section] = dict.fromkeys(values, "***")
            out[name] = clean
        return out


def _expand(
    text: str,
    env: Mapping[str, str],
    *,
    server: str,
    source: str,
    path: Path | None,
    problems: list[MCPDiscoveryProblem],
) -> str:
    """Expand ``${VAR}`` / ``${VAR:-default}``; report unresolved names."""
    missing: list[str] = []

    def _substitute(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        value = env.get(name)
        if value is not None:
            return value
        if default is not None:
            return default
        missing.append(name)
        return match.group(0)

    expanded = _VAR_RE.sub(_substitute, text)
    for name in missing:
        problems.append(
            MCPDiscoveryProblem(
                code="unresolved-variable",
                message=f"environment variable '{name}' is not set; the placeholder was kept literal",
                source=source,
                path=path,
                server=server,
            )
        )
    return expanded


def _name_problem(name: str) -> str | None:
    if not name:
        return "empty server name"
    if "__" in name:
        return "server names may not contain '__' (model tool names are 'mcp__<server>__<tool>')"
    if any(char.isspace() for char in name):
        return "server names may not contain whitespace"
    return None


def _normalize_entry(
    name: str,
    raw: Any,
    *,
    source: str,
    path: Path | None,
    env: Mapping[str, str],
    problems: list[MCPDiscoveryProblem],
) -> tuple[str, dict] | None:
    """One raw entry → (name, MCPClient config), or None with problems recorded."""

    def problem(code: str, message: str, severity: str = "warning") -> None:
        problems.append(MCPDiscoveryProblem(code, message, severity, source, path, name))

    def expand(text: str) -> str:
        return _expand(text, env, server=name, source=source, path=path, problems=problems)

    bad_name = _name_problem(name)
    if bad_name:
        problem("invalid-entry", bad_name)
        return None
    if not isinstance(raw, Mapping):
        problem("invalid-entry", "server entry must be an object")
        return None
    if raw.get("enabled", True) is False:
        problem("disabled", "excluded: 'enabled' is false", "info")
        return None

    transport = raw.get("transport")
    if transport not in TRANSPORTS:
        problem("invalid-entry", f"'transport' must be one of {list(TRANSPORTS)} (got {transport!r})")
        return None
    config: dict[str, Any] = {"transport": transport}

    def string_map(key: str) -> tuple[dict | None, bool]:
        """(mapping or None when absent, invalid-shape flag)."""
        values = raw.get(key)
        if values is None:
            return None, False
        if not isinstance(values, Mapping):
            problem("invalid-entry", f"'{key}' must be an object")
            return None, True
        clean: dict[str, str] = {}
        for item_key, item_value in values.items():
            if not isinstance(item_value, str):
                problem("invalid-entry", f"{key} value for '{item_key}' coerced to string")
            clean[str(item_key)] = expand(str(item_value))
        return clean, False

    if transport == "stdio":
        command = raw.get("command")
        if not isinstance(command, str) or not command.strip():
            problem("invalid-entry", "stdio transport requires a non-empty 'command'")
            return None
        config["command"] = expand(command)
        args = raw.get("args")
        if args is not None:
            if not isinstance(args, list):
                problem("invalid-entry", "'args' must be a list")
                return None
            clean_args: list[str] = []
            for item in args:
                if not isinstance(item, str):
                    problem("invalid-entry", f"non-string arg {item!r} coerced to string")
                clean_args.append(expand(str(item)))
            config["args"] = clean_args
        env_cfg, invalid = string_map("env")
        if invalid:
            return None
        if env_cfg:
            config["env"] = env_cfg
        cwd = raw.get("cwd")
        if cwd is not None:
            if not isinstance(cwd, str):
                problem("invalid-entry", "'cwd' must be a string")
                return None
            config["cwd"] = expand(cwd)
    else:  # streamable-http
        url = raw.get("url")
        if not isinstance(url, str) or not url.strip():
            problem("invalid-entry", "streamable-http transport requires a non-empty 'url'")
            return None
        config["url"] = expand(url)
        headers, invalid = string_map("headers")
        if invalid:
            return None
        if headers:
            config["headers"] = headers

    if "session_pool_size" in raw:
        value = raw["session_pool_size"]
        if transport != "streamable-http":
            problem("invalid-entry", "'session_pool_size' is supported for streamable-http only; ignored")
        elif isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            config["session_pool_size"] = value
        else:
            problem("invalid-entry", "'session_pool_size' must be a non-negative integer; ignored")

    for key in ("timeout", "startup_timeout"):
        if key in raw:
            value = raw[key]
            if value is None:
                config[key] = None
            elif isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
                config[key] = float(value)
            else:
                problem("invalid-entry", f"'{key}' must be a non-negative number or null; ignored")
    if "mode" in raw:
        if raw["mode"] in _MODES:
            config["mode"] = raw["mode"]
        else:
            problem("invalid-entry", f"'mode' must be one of {list(_MODES)}; ignored")
    for key in ("enabled_tools", "disabled_tools"):
        if key in raw:
            value = raw[key]
            if isinstance(value, list) and all(isinstance(item, str) for item in value):
                config[key] = list(value)
            else:
                problem("invalid-entry", f"'{key}' must be a list of tool names; ignored")

    return name, config


def _entries_from_file(
    raw: Any,
    *,
    single_name: str | None,
    expected: str,
    source: str,
    path: Path,
    problems: list[MCPDiscoveryProblem],
) -> list[tuple[str, Any]] | None:
    if isinstance(raw, Mapping):
        if "mcpServers" in raw:
            servers = raw["mcpServers"]
            if isinstance(servers, Mapping):
                return [(str(key), value) for key, value in servers.items()]
            problems.append(
                MCPDiscoveryProblem("unrecognized-file", "'mcpServers' must be an object", source=source, path=path)
            )
            return None
        if single_name is not None:
            return [(single_name, raw)]
    problems.append(MCPDiscoveryProblem("unrecognized-file", expected, source=source, path=path))
    return None


def discover_mcp_servers(
    *,
    extra: Mapping[str, Any] | None = None,
    extra_paths: Iterable[str | os.PathLike[str]] = (),
    cwd: str | os.PathLike[str] | None = None,
    home: str | os.PathLike[str] | None = None,
    env: Mapping[str, str] | None = None,
) -> MCPDiscoveryResult:
    """Discover MCP servers from cliver's own places (in-memory, no writes).

    Args:
        extra: Caller-supplied server configs (internal schema, same as
            ``MCPClient``) — highest precedence.
        extra_paths: Files or directories to read in addition to cliver's
            own places.  A file is parsed like a drop-in (bundle or single
            server named after the stem); a directory contributes its
            ``*.json`` files in sorted order.
        cwd: Project root for the project-local ``.cliver/mcp.json``.
        home: Home directory used when ``CLIVER_CONF_DIR`` is not set.
        env: Environment used for ``CLIVER_CONF_DIR`` and ``${VAR}``
            expansion.

    Returns:
        An :class:`MCPDiscoveryResult` — feed ``result.servers`` into
        :class:`~cliver_core.mcp.MCPClient`.
    """
    cwd_path = Path(cwd) if cwd is not None else Path.cwd()
    home_path = Path(home) if home is not None else Path.home()
    environ = dict(os.environ if env is None else env)
    conf_dir = environ.get("CLIVER_CONF_DIR")
    root = Path(conf_dir).expanduser() if conf_dir else home_path / ".cliver"

    servers: dict[str, dict] = {}
    provenance: dict[str, MCPServerProvenance] = {}
    problems: list[MCPDiscoveryProblem] = []
    sources_read: list[tuple[str, Path]] = []

    def add(source: str, path: Path | None, name: str, config: dict) -> None:
        if name in servers:
            previous = provenance[name]
            where = previous.source + (f" ({previous.path})" if previous.path else "")
            location = f"{source} ({path})" if path else source
            problems.append(
                MCPDiscoveryProblem(
                    code="shadowed",
                    message=f"'{name}' from {location} ignored — already defined by {where}",
                    severity="info",
                    source=source,
                    path=path,
                    server=name,
                )
            )
            return
        servers[name] = config
        provenance[name] = MCPServerProvenance(name, source, path)

    def read_file(path: Path, *, source: str, single_from_stem: bool) -> None:
        expected = (
            "expected a single server object (or an 'mcpServers' bundle)"
            if single_from_stem
            else "expected an object with an 'mcpServers' entry"
        )
        try:
            text = path.read_text(encoding="utf-8-sig")
        except OSError as exc:
            problems.append(MCPDiscoveryProblem("unrecognized-file", f"unreadable: {exc}", source=source, path=path))
            return
        sources_read.append((source, path))
        try:
            raw = json.loads(text)
        except ValueError as exc:
            problems.append(MCPDiscoveryProblem("parse-error", f"invalid JSON: {exc}", "error", source, path))
            return
        entries = _entries_from_file(
            raw,
            single_name=path.stem if single_from_stem else None,
            expected=expected,
            source=source,
            path=path,
            problems=problems,
        )
        for name, value in entries or ():
            normalized = _normalize_entry(name, value, source=source, path=path, env=environ, problems=problems)
            if normalized is not None:
                add(source, path, *normalized)

    # 0 — caller-supplied configs
    for name, raw in (extra or {}).items():
        normalized = _normalize_entry(str(name), raw, source="explicit", path=None, env=environ, problems=problems)
        if normalized is not None:
            add("explicit", None, *normalized)

    # 1 — caller-supplied files/directories (opt-in beyond cliver's own)
    for entry in extra_paths:
        candidate = Path(entry).expanduser()
        if candidate.is_dir():
            files = sorted(
                (p for p in candidate.iterdir() if p.is_file() and p.suffix == ".json" and not p.name.startswith(".")),
                key=lambda p: p.name,
            )
            for file in files:
                read_file(file, source="extra_paths", single_from_stem=True)
        elif candidate.is_file():
            read_file(candidate, source="extra_paths", single_from_stem=True)
        else:
            problems.append(
                MCPDiscoveryProblem("missing-path", "path does not exist", "warning", "extra_paths", candidate)
            )

    # 2 — project-local
    project_file = cwd_path / ".cliver" / "mcp.json"
    if project_file.is_file():
        read_file(project_file, source="cliver_project", single_from_stem=False)

    # 3 — cliver user store
    user_file = root / "mcp.json"
    if user_file.is_file():
        read_file(user_file, source="cliver_user", single_from_stem=False)

    # 4 — user drop-ins
    dropin_dir = root / "mcps"
    if dropin_dir.is_dir():
        files = sorted(
            (p for p in dropin_dir.iterdir() if p.is_file() and p.suffix == ".json" and not p.name.startswith(".")),
            key=lambda p: p.name,
        )
        for file in files:
            read_file(file, source="cliver_user_dropin", single_from_stem=True)

    result = MCPDiscoveryResult(servers, provenance, tuple(problems), tuple(sources_read))
    logger.info("MCP discovery: %s", result.summary())
    return result
