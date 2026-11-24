"""JSON configuration for the ``cliver`` CLI.

Two optional files are read, lowest priority first:

``{config_dir}/config.json``
    The user file (``~/.cliver/config.json`` on Unix, ``%APPDATA%/cliver`` on
    Windows).  ``CLIVER_CONF_DIR`` moves the directory.
``<cwd>/.cliver/config.json``
    The project file.  A purpose's model list replaces the user file's list,
    skill and MCP source lists are appended, MCP servers merge by name, and a
    project selector entry overrides the matching user entry.

``CLIVER_CONFIG`` (or ``--config PATH``) selects one explicit file instead of
the two defaults.  ``{config_dir}/.env`` is loaded before this module reads a
file, so ``${VAR}`` and ``${VAR:-default}`` references resolve.

Schema::

    {
      "models": {
        "llm": [
          {"model": "deepseek-v4-flash", "provider": "deepseek",
           "api_key_env": "DEEPSEEK_API_KEY",
           "params": {"temperature": 0.2}},
          {"model": "glm-5.3", "provider": "glm",
           "api_key_env": "GLM_API_KEY"}
        ],
        "image": [{"model": "qwen-image-plus", "provider": "qwen-image"}],
        "video": [{"model": "...", "provider": "dashscope"}],
        "tts": [{"model": "...", "provider": "minimax"}],
        "asr": [{"model": "...", "provider": "minimax"}],
        "embed": [{"model": "...", "base_url": "https://...", "api_key": "..."}],
        "rerank": [{"model": "...", "base_url": "https://...", "api_key": "..."}],
        "system_one": [
          {"provider": "bailian", "workspace_id": "ws-...",
           "api_key_env": "DASHSCOPE_API_KEY"},
          {"provider": "typesafe", "api_key_env": "TYPESAFE_API_KEY"}
        ]
      },
      "skills": {
        "sources": ["~/.claude/skills", "~/.copilot/skills", ".agents/skills"]
      },
      "mcp": {
        "sources": ["/abs/path/team-mcp.json"],
        "shared": ["claude", "cursor", "gemini", "qwen"],
        "shared_project": false,
        "shared_sources": ["~/team-mcp.json"],
        "servers": {
          "docs": {"transport": "streamable-http", "url": "https://example/mcp"}
        },
        "enabled": ["docs", "sqlite"],
        "disabled": ["experimental"]
      },
      "selector": {"enabled": true, "min_tools": 20}
    }

Every purpose takes an ordered list.  The first entry is the primary; the
rest are same-purpose fallbacks used on retryable failures.  A single object
is accepted as a one-element list.  ``audio`` is the legacy slot and builds a
TTS binding.  ``system_one`` lives under ``models`` like every other purpose.

``mcp.servers`` uses cliver's own server schema (``transport`` is required and
is ``stdio`` or ``streamable-http``).  ``mcp.sources`` points at additional
files in the same schema; cliver-owned files are always read:
``{cwd}/.cliver/mcp.json``, ``{config_dir}/mcp.json``, and
``{config_dir}/mcps/*.json``.  ``mcp.enabled`` is the optional default
selection list — when present, only those server names are mounted.

``mcp.shared`` opts into other agents' files: ``true`` selects every known
user-level source, or list source IDs (``claude``, ``cursor``, ``gemini``,
``qwen``, ``windsurf``, ``claude-desktop``, ``continue``, ``zed``,
``antigravity``).  ``mcp.shared_project`` adds project-level files for the
selected families, and ``mcp.shared_sources`` adds explicit JSON paths.
Shared servers are normalized to cliver's schema and merged *below*
cliver-owned definitions; nothing is read without one of these settings.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cliver_core import ClientOptions, ModelBinding
from cliver_core.bindings import PURPOSES
from cliver_core.local import LocalSystemOneProvider, local_system_one
from cliver_core.mcp import MCPClient, MCPDiscoveryProblem, discover_mcp_servers
from cliver_core.presets import bailian_system_one, typesafe_system_one
from cliver_core.provider.providers import API_KEY_ENV

from cliver.mcp_shared import SHARED_MCP_SOURCE_IDS, load_shared_mcp_servers
from cliver.paths import get_config_dir

CONFIG_ENV = "CLIVER_CONFIG"
CONFIG_FILENAME = "config.json"
PROJECT_CONFIG_DIR = ".cliver"

#: The cliver-owned files core discovery always reads (in precedence order).
#: ``mcp.sources`` adds files in cliver's schema on top of these.
DEFAULT_MCP_SOURCES = (
    "{cwd}/.cliver/mcp.json",
    "{config_dir}/mcp.json",
    "{config_dir}/mcps/*.json",
)

#: Purpose → factory.  ``audio`` is the legacy spelling of the TTS slot.
_FACTORIES = {
    "llm": ModelBinding.llm,
    "embed": ModelBinding.embed,
    "rerank": ModelBinding.rerank,
    "image": ModelBinding.image,
    "video": ModelBinding.video,
    "tts": ModelBinding.tts,
    "asr": ModelBinding.asr,
    "audio": ModelBinding.tts,
}

_BINDING_FIELDS = {
    "model",
    "provider",
    "provider_name",
    "base_url",
    "protocol",
    "api_key",
    "api_key_env",
    "path",
    "wire_shape",
    "client",
    "client_timeout",
    "client_max_retries",
}
_GENERIC_KEYS = _BINDING_FIELDS | {"params"}
_SYSTEM_ONE_KEYS = {
    "model",
    "provider",
    "provider_name",
    "base_url",
    "protocol",
    "api_key",
    "api_key_env",
    "path",
    "params",
    "client",
    "client_timeout",
    "client_max_retries",
    "workspace_id",
    "region",
    "device",
    "max_len",
    "backend",
}
_TOP_LEVEL_KEYS = {"$schema", "models", "skills", "mcp", "selector"}
_VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")

#: Selector config key → the environment variable ``selector_from_env`` reads.
_SELECTOR_ENV = {
    "enabled": "CLIVER_SELECTOR",
    "min_tools": "CLIVER_SELECTOR_MIN_TOOLS",
    "min_skills": "CLIVER_SELECTOR_MIN_SKILLS",
    "top_k_tools": "CLIVER_SELECTOR_TOP_K_TOOLS",
    "top_k_skills": "CLIVER_SELECTOR_TOP_K_SKILLS",
    "chunk_size": "CLIVER_SELECTOR_CHUNK_SIZE",
    "shortlist": "CLIVER_SELECTOR_SHORTLIST",
    "close_read": "CLIVER_SELECTOR_CLOSE_READ",
    "fit_threshold": "CLIVER_SELECTOR_FIT_THRESHOLD",
    "timeout": "CLIVER_SELECTOR_TIMEOUT",
}

__all__ = [
    "CLIConfig",
    "CONFIG_ENV",
    "CONFIG_FILENAME",
    "DEFAULT_MCP_SOURCES",
    "MCPSelection",
    "create_mcp_client",
    "load_config",
]


@dataclass(frozen=True)
class CLIConfig:
    """The merged configuration the CLI runs with."""

    models: dict[str, list[ModelBinding]] = field(default_factory=dict)
    skill_sources: tuple[Path, ...] = ()
    mcp_sources: tuple[Path, ...] = ()
    mcp_servers: dict[str, dict] = field(default_factory=dict)
    mcp_enabled: tuple[str, ...] | None = None
    mcp_disabled: tuple[str, ...] = ()
    mcp_shared: tuple[str, ...] | None = ()
    mcp_shared_project: bool = False
    mcp_shared_sources: tuple[Path, ...] = ()
    selector_env: dict[str, str] = field(default_factory=dict)
    paths: tuple[Path, ...] = ()
    config_dir: Path | None = None


@dataclass(frozen=True)
class MCPSelection:
    """A built MCP client plus discovery diagnostics for the CLI to report."""

    client: MCPClient | None
    problems: tuple[MCPDiscoveryProblem, ...] = ()
    summary: str = ""


def load_config(
    *,
    config_path: str | os.PathLike[str] | None = None,
    config_dir: str | os.PathLike[str] | None = None,
    cwd: str | os.PathLike[str] | None = None,
    home: str | os.PathLike[str] | None = None,
    env: Mapping[str, str] | None = None,
) -> CLIConfig:
    """Read and merge the CLI's JSON configuration.

    ``config_path`` (or ``CLIVER_CONFIG``) selects one explicit file.  Without
    it the user file and the project file are merged.  A missing default file
    is ignored; an explicit file that does not exist is an error.
    """
    values = dict(os.environ if env is None else env)
    working = Path.cwd() if cwd is None else Path(cwd).expanduser()
    home_dir = Path.home() if home is None else Path(home).expanduser()
    config_root = get_config_dir(home=home_dir, env=values) if config_dir is None else Path(config_dir).expanduser()

    explicit = config_path or values.get(CONFIG_ENV)
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise ValueError(f"configuration file not found: {path}")
        raw = _read_config_file(path, values)
        merged = raw
        paths = (path,)
    else:
        merged: dict[str, Any] = {}
        paths_list: list[Path] = []
        for path in (config_root / CONFIG_FILENAME, working / PROJECT_CONFIG_DIR / CONFIG_FILENAME):
            if not path.is_file():
                continue
            merged = _merge_configs(merged, _read_config_file(path, values))
            paths_list.append(path)
        paths = tuple(paths_list)

    return _build_config(merged, values, paths=paths, config_dir=config_root)


def create_mcp_client(
    config: CLIConfig,
    *,
    cwd: str | os.PathLike[str] | None = None,
    home: str | os.PathLike[str] | None = None,
    env: Mapping[str, str] | None = None,
) -> MCPSelection:
    """Discover MCP servers and build a client for the selected subset.

    cliver-owned files are read by core's discovery.  ``mcp.sources`` adds
    more cliver-schema files, shared sources add other agents' files
    (normalized here), ``mcp.enabled`` narrows the result to a default
    selection list, and ``mcp.disabled`` removes names.  No server is started.
    """
    values = dict(os.environ if env is None else env)
    if config.config_dir is not None:
        values["CLIVER_CONF_DIR"] = str(config.config_dir)
    cwd_path = Path.cwd() if cwd is None else Path(cwd).expanduser()
    home_path = Path.home() if home is None else Path(home).expanduser()

    result = discover_mcp_servers(
        extra=config.mcp_servers or None,
        extra_paths=config.mcp_sources,
        cwd=cwd_path,
        home=home_path,
        env=values,
    )
    shared = load_shared_mcp_servers(
        selected=config.mcp_shared,
        include_project=config.mcp_shared_project,
        explicit_paths=config.mcp_shared_sources,
        cwd=cwd_path,
        home=home_path,
        env=values,
    )
    servers = dict(result.servers)
    problems = list(result.problems)

    for name, server in shared.servers.items():
        if name in servers:
            previous = result.provenance.get(name)
            where = previous.source if previous is not None else "another source"
            if previous is not None and previous.path is not None:
                where += f" ({previous.path})"
            current = shared.provenance[name]
            problems.append(
                MCPDiscoveryProblem(
                    code="shadowed",
                    message=f"'{name}' from {current.source} ({current.path}) ignored — already defined by {where}",
                    severity="info",
                    source=current.source,
                    path=current.path,
                    server=name,
                )
            )
            continue
        servers[name] = server
    problems.extend(shared.problems)

    if config.mcp_enabled is not None:
        for name in config.mcp_enabled:
            if name not in servers:
                problems.append(_selection_problem("selection-missing", f"selected server '{name}' was not discovered"))
        servers = {name: servers[name] for name in config.mcp_enabled if name in servers}

    if config.mcp_disabled:
        disabled = set(config.mcp_disabled)
        servers = {name: server for name, server in servers.items() if name not in disabled}

    client = MCPClient(servers) if servers else None
    return MCPSelection(
        client=client,
        problems=tuple(problems),
        summary=_summary(servers, result, shared, config, problems),
    )


def _summary(
    servers: Mapping[str, Any],
    result: Any,
    shared: Any,
    config: CLIConfig,
    problems: list[MCPDiscoveryProblem],
) -> str:
    """A short startup line; inline definitions count as one more source."""
    count = len(servers)
    sources = {source for source, _ in result.sources_read}
    sources |= {source for source, _ in shared.sources_read}
    if config.mcp_servers:
        sources.add("config")
    text = f"{count} server{'s' if count != 1 else ''} from {len(sources)} source{'s' if len(sources) != 1 else ''}"
    visible = [problem for problem in problems if problem.severity != "info"]
    if visible:
        text += f"; {len(visible)} problem{'s' if len(visible) != 1 else ''}"
    return text


def _selection_problem(code: str, message: str) -> MCPDiscoveryProblem:
    return MCPDiscoveryProblem(code=code, message=message, severity="warning", source="selection")


# ── File loading and merging ────────────────────────────────


def _read_config_file(path: Path, env: Mapping[str, str]) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise ValueError(f"cannot read configuration file {path}: {exc}") from exc
    try:
        raw = json.loads(text)
    except ValueError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise ValueError(f"{path}: the top level must be a JSON object")
    return _normalize(raw, path, env)


def _normalize(raw: Mapping[str, Any], path: Path, env: Mapping[str, str]) -> dict[str, Any]:
    """Expand ``${VAR}`` and resolve path lists against the file's directory."""
    expanded = _expand(raw, env, path)
    result = dict(expanded)
    for section, key in (("skills", "sources"), ("mcp", "sources"), ("mcp", "shared_sources")):
        value = result.get(section)
        if not isinstance(value, Mapping) or key not in value:
            continue
        sources = value[key]
        if not isinstance(sources, list) or not all(isinstance(item, str) for item in sources):
            raise ValueError(f"{path}: {section}.{key} must be a list of strings")
        resolved = []
        for item in sources:
            if not item.strip():
                raise ValueError(f"{path}: {section}.{key} may not contain empty paths")
            candidate = Path(item).expanduser()
            resolved.append(str(candidate if candidate.is_absolute() else (path.parent / candidate).resolve()))
        result[section] = {**value, key: resolved}
    return result


def _expand(value: Any, env: Mapping[str, str], path: Path) -> Any:
    if isinstance(value, str):

        def replace(match: re.Match[str]) -> str:
            name, default = match.group(1), match.group(2)
            if name in env:
                return env[name]
            if default is not None:
                return default
            raise ValueError(f"{path}: environment variable '{name}' is not set (referenced as ${{{name}}})")

        return _VAR_RE.sub(replace, value)
    if isinstance(value, list):
        return [_expand(item, env, path) for item in value]
    if isinstance(value, Mapping):
        return {key: _expand(item, env, path) for key, item in value.items()}
    return value


def _merge_configs(user: Mapping[str, Any], project: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(user)
    for key, value in project.items():
        if key == "models" and isinstance(value, Mapping):
            current = merged.get("models")
            models = dict(current) if isinstance(current, Mapping) else {}
            models.update(value)
            merged["models"] = models
        elif key == "skills" and isinstance(value, Mapping):
            current = merged.get("skills")
            current = current if isinstance(current, Mapping) else {}
            sources = list(current.get("sources") or []) + list(value.get("sources") or [])
            merged["skills"] = {**current, **value, "sources": _dedupe(sources)}
        elif key == "mcp" and isinstance(value, Mapping):
            current = merged.get("mcp")
            current = current if isinstance(current, Mapping) else {}
            sources = _dedupe(list(current.get("sources") or []) + list(value.get("sources") or []))
            shared_sources = _dedupe(
                list(current.get("shared_sources") or []) + list(value.get("shared_sources") or [])
            )
            merged_mcp = {**current, **value, "sources": sources, "shared_sources": shared_sources}
            current_servers = current.get("servers")
            project_servers = value.get("servers")
            if isinstance(current_servers, Mapping) and isinstance(project_servers, Mapping):
                merged_mcp["servers"] = {**current_servers, **project_servers}
            merged["mcp"] = merged_mcp
        elif key == "selector" and isinstance(value, Mapping):
            current = merged.get("selector")
            current = current if isinstance(current, Mapping) else {}
            merged["selector"] = {**current, **value}
        else:
            merged[key] = value
    return merged


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        key = str(Path(value).expanduser().resolve())
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


# ── Building the merged configuration ──────────────────────


def _build_config(
    raw: Mapping[str, Any],
    env: Mapping[str, str],
    *,
    paths: tuple[Path, ...],
    config_dir: Path,
) -> CLIConfig:
    origin = ", ".join(str(path) for path in paths) or "configuration"
    unknown = sorted(set(raw) - _TOP_LEVEL_KEYS)
    if unknown:
        raise ValueError(f"{origin}: unknown top-level key(s) {unknown}; allowed: {sorted(_TOP_LEVEL_KEYS)}")

    models = _build_models(raw.get("models"), env, origin)
    skills_raw = _mapping(raw.get("skills"), "skills", origin)
    skill_sources = tuple(_path_list(skills_raw.get("sources", []), "skills.sources", origin))

    mcp_raw = _mapping(raw.get("mcp"), "mcp", origin)
    unknown_mcp = sorted(
        set(mcp_raw) - {"sources", "shared_sources", "servers", "enabled", "disabled", "shared", "shared_project"}
    )
    if unknown_mcp:
        raise ValueError(f"{origin}: unknown mcp key(s) {unknown_mcp}")
    mcp_sources = tuple(_path_list(mcp_raw.get("sources", []), "mcp.sources", origin))
    mcp_shared_sources = tuple(_path_list(mcp_raw.get("shared_sources", []), "mcp.shared_sources", origin))
    servers_raw = _mapping(mcp_raw.get("servers"), "mcp.servers", origin)
    mcp_servers: dict[str, dict] = {}
    for name, server in servers_raw.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{origin}: mcp.servers names must be non-empty strings")
        if not isinstance(server, Mapping):
            raise ValueError(f"{origin}: mcp.servers.{name} must be an object")
        mcp_servers[name] = dict(server)
    mcp_enabled = _name_list(mcp_raw.get("enabled"), "mcp.enabled", origin, none=True)
    mcp_disabled = _name_list(mcp_raw.get("disabled"), "mcp.disabled", origin) or ()
    mcp_shared = _shared_selection(mcp_raw.get("shared", False), origin)
    mcp_shared_project = mcp_raw.get("shared_project", False)
    if not isinstance(mcp_shared_project, bool):
        raise ValueError(f"{origin}: mcp.shared_project must be a boolean")

    selector_raw = _mapping(raw.get("selector"), "selector", origin)
    selector_env = _selector_env(selector_raw, origin)

    return CLIConfig(
        models=models,
        skill_sources=skill_sources,
        mcp_sources=mcp_sources,
        mcp_servers=mcp_servers,
        mcp_enabled=mcp_enabled,
        mcp_disabled=mcp_disabled,
        mcp_shared=mcp_shared,
        mcp_shared_project=mcp_shared_project,
        mcp_shared_sources=mcp_shared_sources,
        selector_env=selector_env,
        paths=paths,
        config_dir=config_dir,
    )


def _build_models(raw: Any, env: Mapping[str, str], origin: str) -> dict[str, list[ModelBinding]]:
    models_raw = _mapping(raw, "models", origin)
    models: dict[str, list[ModelBinding]] = {}
    for purpose, entries in models_raw.items():
        if purpose not in PURPOSES:
            raise ValueError(f"{origin}: unknown model purpose {purpose!r}; supported: {list(PURPOSES)}")
        models[purpose] = _binding_list(purpose, entries, env, origin)
    return models


def _binding_list(purpose: str, entries: Any, env: Mapping[str, str], origin: str) -> list[ModelBinding]:
    if isinstance(entries, Mapping):
        entries = [entries]
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        raise ValueError(f"{origin}: models.{purpose} must be an object or a list of objects")
    if purpose == "llm" and not entries:
        raise ValueError(f"{origin}: models.llm needs at least one binding")
    return [
        _build_binding(purpose, entry, env, f"{origin}: models.{purpose}[{index}]")
        for index, entry in enumerate(entries)
    ]


def _build_binding(purpose: str, entry: Any, env: Mapping[str, str], where: str) -> ModelBinding:
    if not isinstance(entry, Mapping):
        raise ValueError(f"{where} must be an object")
    data = dict(entry)
    if purpose == "system_one":
        return _build_system_one(data, env, where)

    unknown = sorted(set(data) - _GENERIC_KEYS)
    if unknown:
        raise ValueError(f"{where}: unknown key(s) {unknown}; request options belong under 'params'")
    if purpose == "llm" and ({"path", "wire_shape"} & set(data)):
        raise ValueError(f"{where}: llm bindings take no 'path' or 'wire_shape'")
    if purpose in {"embed", "rerank", "asr"} and "wire_shape" in data:
        raise ValueError(f"{where}: {purpose} bindings take no 'wire_shape'")

    model = _required_text(data, "model", where)
    provider = _provider_name(data, where)
    api_key = _resolve_api_key(data, env, provider, where)
    kwargs: dict[str, Any] = {"model": model, "api_key": api_key}
    if provider:
        kwargs["provider_name"] = provider
    for key in ("base_url", "protocol"):
        if key in data:
            value = _optional_text(data, key, where)
            if value:
                kwargs[key] = value
    for key in ("path", "wire_shape"):
        if key in data:
            kwargs[key] = _optional_text_or_none(data, key, where)
    if "client" in data:
        kwargs["client"] = _client_options(data["client"], where)
    for key in ("client_timeout", "client_max_retries"):
        if key in data:
            kwargs[key] = data[key]
    kwargs.update(_params(data, where))
    return _FACTORIES[purpose](**kwargs)


def _build_system_one(data: dict[str, Any], env: Mapping[str, str], where: str) -> ModelBinding:
    unknown = sorted(set(data) - _SYSTEM_ONE_KEYS)
    if unknown:
        raise ValueError(f"{where}: unknown key(s) {unknown}; request options belong under 'params'")
    provider = _provider_name(data, where)
    kind = (provider or "").lower()
    params = _params(data, where)

    if kind in {"bailian", "typesafe"}:
        preset_reject = {"device", "max_len", "backend", "protocol"}
        if kind == "typesafe":
            preset_reject |= {"workspace_id", "region"}
        _reject_keys(data, preset_reject, where)
        model = _optional_text(data, "model", where)
        kwargs: dict[str, Any] = {
            "api_key": _resolve_api_key(data, env, "dashscope" if kind == "bailian" else kind, where),
            "model": model or ("decision-model-preview" if kind == "bailian" else "jev-latest"),
        }
        if "base_url" in data:
            kwargs["base_url"] = _optional_text(data, "base_url", where)
        if kind == "bailian":
            kwargs["workspace_id"] = _optional_text(data, "workspace_id", where) or ""
            kwargs["region"] = _optional_text(data, "region", where) or "cn-beijing"
        if "path" in data:
            kwargs["path"] = _optional_text_or_none(data, "path", where)
        if "client" in data:
            kwargs["client"] = _client_options(data["client"], where)
        for key in ("client_timeout", "client_max_retries"):
            if key in data:
                kwargs[key] = data[key]
        kwargs.update(params)
        return bailian_system_one(**kwargs) if kind == "bailian" else typesafe_system_one(**kwargs)

    if kind == "laya":
        _reject_keys(
            data,
            {
                "base_url",
                "path",
                "protocol",
                "client",
                "client_timeout",
                "client_max_retries",
                "workspace_id",
                "region",
                "api_key",
                "api_key_env",
            },
            where,
        )
        model = _required_text(data, "model", where)
        backend = _optional_text(data, "backend", where) or "laya"
        if backend != "laya":
            raise ValueError(f"{where}: the laya provider requires backend 'laya' (got {backend!r})")
        if params:
            raise ValueError(f"{where}: laya does not accept 'params'; use 'max_len' and 'device'")
        local = LocalSystemOneProvider(
            model=model,
            backend="laya",
            device=_optional_text(data, "device", where) or "auto",
            max_length=_optional_int(data.get("max_len"), "max_len", where),
        )
        return local_system_one(provider=local)

    if kind:
        raise ValueError(
            f"{where}: unknown system_one provider {kind!r}; use bailian, typesafe, laya, "
            "or omit 'provider' for a custom base_url"
        )

    _reject_keys(data, {"workspace_id", "region", "device", "max_len", "backend"}, where)
    model = _required_text(data, "model", where)
    base_url = _optional_text(data, "base_url", where)
    if not base_url:
        raise ValueError(f"{where}: a custom system_one binding needs 'base_url'")
    protocol = _optional_text(data, "protocol", where) or "system-one"
    if protocol != "system-one":
        raise ValueError(f"{where}: custom system_one bindings support protocol 'system-one' only")
    kwargs = {"model": model, "api_key": _resolve_api_key(data, env, None, where), "base_url": base_url}
    if "path" in data:
        kwargs["path"] = _optional_text_or_none(data, "path", where)
    if "client" in data:
        kwargs["client"] = _client_options(data["client"], where)
    for key in ("client_timeout", "client_max_retries"):
        if key in data:
            kwargs[key] = data[key]
    kwargs.update(params)
    return ModelBinding.system_one(**kwargs)


# ── Field helpers ──────────────────────────────────────────


def _mapping(value: Any, name: str, origin: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{origin}: {name} must be an object")
    return value


def _path_list(value: Any, name: str, origin: str) -> list[Path]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{origin}: {name} must be a list of strings")
    return [Path(item).expanduser() for item in value]


def _name_list(value: Any, name: str, origin: str, *, none: bool = False) -> tuple[str, ...] | None:
    if value is None:
        if none:
            return None
        return ()
    if not isinstance(value, list) or not all(isinstance(item, str) and item.strip() for item in value):
        raise ValueError(f"{origin}: {name} must be a list of non-empty server names")
    return tuple(value)


def _shared_selection(value: Any, origin: str) -> tuple[str, ...] | None:
    """``mcp.shared``: false/absent → none, true → every user source, list → IDs."""
    if value is None or value is False:
        return ()
    if value is True:
        return None
    if not isinstance(value, list) or not all(isinstance(item, str) and item.strip() for item in value):
        raise ValueError(f"{origin}: mcp.shared must be a boolean or a list of source IDs")
    unknown = sorted(set(value) - SHARED_MCP_SOURCE_IDS)
    if unknown:
        raise ValueError(
            f"{origin}: unknown mcp.shared source(s) {unknown}; supported: {sorted(SHARED_MCP_SOURCE_IDS)}"
        )
    return tuple(value)


def _provider_name(data: Mapping[str, Any], where: str) -> str | None:
    short, long = data.get("provider"), data.get("provider_name")
    if short is not None and long is not None and short != long:
        raise ValueError(f"{where}: 'provider' and 'provider_name' disagree")
    value = short if short is not None else long
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where}: provider must be a non-empty string")
    return value.strip()


def _resolve_api_key(
    data: Mapping[str, Any],
    env: Mapping[str, str],
    brand: str | None,
    where: str,
) -> str:
    if "api_key" in data:
        value = data["api_key"]
        if not isinstance(value, str):
            raise ValueError(f"{where}: api_key must be a string")
        return value
    name = data.get("api_key_env")
    if name is not None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{where}: api_key_env must be a non-empty environment variable name")
        if name not in env:
            raise ValueError(f"{where}: api_key_env '{name}' is not set")
        return env[name]
    if brand:
        for candidate in API_KEY_ENV.get(brand.lower(), ()):
            if env.get(candidate):
                return env[candidate]
    return ""


def _required_text(data: Mapping[str, Any], key: str, where: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where}: '{key}' is required and must be a non-empty string")
    return value.strip()


def _optional_text(data: Mapping[str, Any], key: str, where: str) -> str:
    """A string field; absent/null becomes ``""``, non-strings are rejected."""
    value = data.get(key)
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"{where}: '{key}' must be a string")
    return value.strip()


def _optional_text_or_none(data: Mapping[str, Any], key: str, where: str) -> str | None:
    """A nullable endpoint field; absent/null/empty becomes ``None``."""
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{where}: '{key}' must be a string or null")
    return value.strip() or None


def _optional_int(value: Any, key: str, where: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{where}: '{key}' must be a positive integer")
    return value


def _params(data: Mapping[str, Any], where: str) -> dict[str, Any]:
    value = data.get("params", {})
    if not isinstance(value, Mapping):
        raise ValueError(f"{where}: 'params' must be an object")
    reserved = sorted(set(value) & _BINDING_FIELDS)
    if reserved:
        raise ValueError(f"{where}: params may not contain binding field(s) {reserved}")
    return dict(value)


def _client_options(value: Any, where: str) -> ClientOptions:
    if not isinstance(value, Mapping):
        raise ValueError(f"{where}: 'client' must be an object")
    try:
        return ClientOptions(**value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{where}: invalid 'client' options: {exc}") from exc


def _reject_keys(data: Mapping[str, Any], keys: set[str], where: str) -> None:
    present = sorted(keys & set(data))
    if present:
        raise ValueError(f"{where}: key(s) {present} do not apply to this system_one provider")


def _selector_env(raw: Mapping[str, Any], origin: str) -> dict[str, str]:
    unknown = sorted(set(raw) - set(_SELECTOR_ENV))
    if unknown:
        raise ValueError(f"{origin}: unknown selector key(s) {unknown}; supported: {sorted(_SELECTOR_ENV)}")
    values: dict[str, str] = {}
    for key, value in raw.items():
        if isinstance(value, bool):
            values[_SELECTOR_ENV[key]] = "on" if value else "off"
        elif isinstance(value, (int, float)):
            values[_SELECTOR_ENV[key]] = str(value)
        else:
            raise ValueError(f"{origin}: selector.{key} must be a boolean or a number")
    return values
