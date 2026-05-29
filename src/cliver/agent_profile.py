"""
CliverProfile — manages instance-scoped resources for CLIver.

Single instance per host. Resources stored under {config_dir}/:

  Files (human-editable):
    identity.md     — YAML frontmatter (name, role, preferences) + free-form body
    memory.md       — persistent knowledge, append-only
    tasks/          — scheduled task definitions (YAML)
    audit_logs/     — token usage logs (JSONL, one file per month)

  SQLite (structured data):
    cliver.db       — sessions, turns, keys, labs, golden tests, task runs
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

import yaml

from cliver.util import get_config_dir

if TYPE_CHECKING:
    from cliver.llm.agent_core import AgentCore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level singletons — set at startup, accessed by builtin tools.
# ---------------------------------------------------------------------------

_current_profile: Optional["CliverProfile"] = None
_agent_core: Optional["AgentCore"] = None
_input_fn: Callable[..., str] = input
_output_fn: Callable[..., None] = print
_cli_instance: Any = None


def set_current_profile(profile: Optional["CliverProfile"]) -> None:
    global _current_profile
    _current_profile = profile


def get_current_profile() -> Optional["CliverProfile"]:
    return _current_profile


def set_agent_core(executor: "AgentCore") -> None:
    global _agent_core
    _agent_core = executor


def get_agent_core() -> Optional["AgentCore"]:
    return _agent_core


def set_input_fn(fn: Callable[..., str]) -> None:
    global _input_fn
    _input_fn = fn


def get_input_fn() -> Callable[..., str]:
    return _input_fn


def set_output_fn(fn: Callable[..., None]) -> None:
    global _output_fn
    _output_fn = fn


def get_output_fn() -> Callable[..., None]:
    return _output_fn


def set_cli_instance(instance: Any) -> None:
    global _cli_instance
    _cli_instance = instance


def get_cli_instance() -> Any:
    return _cli_instance


MAX_MEMORY_CHARS = 3000
MAX_IDENTITY_CHARS = 1500

_DEFAULT_IDENTITY = """\
---
name: CLIver
role: General-purpose AI assistant
preferences:
  style: concise
---
"""


class CliverProfile:
    """Manages instance-scoped resources under {config_dir}/.

    Layout:
        {config_dir}/
        ├── identity.md          # YAML frontmatter + free-form body
        ├── memory.md            # persistent knowledge
        ├── tasks/               # scheduled task definitions (YAML)
        ├── audit_logs/          # token usage logs (JSONL per month)
        └── cliver.db            # unified SQLite database (all tables)
    """

    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = config_dir or get_config_dir()

        self.memory_file = self.config_dir / "memory.md"
        self.identity_file = self.config_dir / "identity.md"
        self.tasks_dir = self.config_dir / "tasks"

    @property
    def db_path(self) -> Path:
        """Unified SQLite database path shared by all stores."""
        return self.config_dir / "cliver.db"

    def ensure_dirs(self) -> None:
        self.config_dir.mkdir(parents=True, exist_ok=True)

    # -- Profile (YAML frontmatter in identity.md) ----------------------------

    def load_profile(self) -> dict:
        """Parse YAML frontmatter from identity.md and return as dict."""
        content = self._read_file(self.identity_file)
        if not content:
            return {}
        frontmatter, _ = _parse_frontmatter(content)
        return frontmatter

    def set_profile_field(self, key: str, value: Any) -> None:
        """Set a field in identity.md YAML frontmatter.

        Supports dot-notation for nested keys (e.g., 'preferences.language').
        """
        content = self._read_file(self.identity_file)
        if not content:
            content = _DEFAULT_IDENTITY

        frontmatter, body = _parse_frontmatter(content)

        _set_nested(frontmatter, key, value)

        new_content = _render_frontmatter(frontmatter, body)
        self.identity_file.parent.mkdir(parents=True, exist_ok=True)
        self.identity_file.write_text(new_content, encoding="utf-8")

    @property
    def profile_name(self) -> str:
        """Get the profile name from identity frontmatter, defaulting to 'CLIver'."""
        return self.load_profile().get("name", "CLIver")

    # -- Memory ----------------------------------------------------------------

    def load_memory(self) -> str:
        """Load memory for system prompt injection.

        Truncated to MAX_MEMORY_CHARS from the end (keeping recent entries).
        """
        content = self._read_file(self.memory_file)
        if not content:
            return ""
        if len(content) > MAX_MEMORY_CHARS:
            content = "...(truncated)\n" + content[-MAX_MEMORY_CHARS:]
        return content

    def append_memory(self, entry: str, comment: str = "") -> None:
        """Append a timestamped memory entry."""
        from cliver.util import format_datetime

        timestamp = format_datetime()
        comment_line = f" — {comment}" if comment else ""
        formatted = f"\n## {timestamp}{comment_line}\n{entry}\n"

        if not self.memory_file.exists():
            self.memory_file.parent.mkdir(parents=True, exist_ok=True)
            self.memory_file.write_text(f"# Memory\n{formatted}", encoding="utf-8")
        else:
            with open(self.memory_file, "a", encoding="utf-8") as f:
                f.write(formatted)

        logger.info(f"Memory appended: {entry[:80]}...")

    def save_memory(self, content: str) -> None:
        """Replace the entire memory document."""
        self.memory_file.parent.mkdir(parents=True, exist_ok=True)
        self.memory_file.write_text(content, encoding="utf-8")
        logger.info(f"Memory rewritten: {len(content)} chars")

    # -- Identity --------------------------------------------------------------

    def load_identity(self) -> str:
        """Load identity markdown for system prompt injection.

        Returns the full identity.md content (frontmatter + body).
        Truncated to MAX_IDENTITY_CHARS.
        """
        content = self._read_file(self.identity_file)
        if content and len(content) > MAX_IDENTITY_CHARS:
            content = content[:MAX_IDENTITY_CHARS] + "\n...(truncated)"
        return content

    def save_identity(self, content: str) -> None:
        """Save identity markdown, replacing the entire document."""
        self.ensure_dirs()
        self.identity_file.write_text(content, encoding="utf-8")

    # -- Helpers ---------------------------------------------------------------

    @staticmethod
    def _read_file(path: Path) -> str:
        if not path.exists():
            return ""
        try:
            return path.read_text(encoding="utf-8").strip()
        except Exception as e:
            logger.warning(f"Could not read {path}: {e}")
            return ""


# Backward compatibility alias
AgentProfile = CliverProfile


# ---------------------------------------------------------------------------
# YAML frontmatter helpers
# ---------------------------------------------------------------------------


def _parse_frontmatter(content: str) -> tuple[dict, str]:
    """Parse YAML frontmatter and body from a markdown string.

    Returns (frontmatter_dict, body_string).
    """
    if not content.startswith("---"):
        return {}, content

    end = content.find("\n---", 3)
    if end == -1:
        return {}, content

    yaml_str = content[4:end]
    body = content[end + 4 :].strip()

    try:
        frontmatter = yaml.safe_load(yaml_str) or {}
    except yaml.YAMLError:
        return {}, content

    return frontmatter, body


def _render_frontmatter(frontmatter: dict, body: str) -> str:
    """Render YAML frontmatter + body back to markdown string."""
    yaml_str = yaml.dump(frontmatter, default_flow_style=False, allow_unicode=True).strip()
    parts = [f"---\n{yaml_str}\n---"]
    if body:
        parts.append(body)
    return "\n\n".join(parts) + "\n"


def _set_nested(d: dict, key: str, value: Any) -> None:
    """Set a value in a nested dict using dot-notation key."""
    keys = key.split(".")
    for k in keys[:-1]:
        if k not in d or not isinstance(d[k], dict):
            d[k] = {}
        d = d[k]
    d[keys[-1]] = value
