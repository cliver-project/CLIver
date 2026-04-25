"""
CliverProfile — manages instance-scoped resources for a CLIver agent.

Each agent instance (identified by `agent_name`) has its own:
- memory.md   — persistent knowledge, append-only
- identity.md — living document about agent persona and user profile

These are stored under {config_dir}/agents/{agent_name}/.

Global resources (config, skills, workflows, tools) are NOT managed here.
There is also a global memory at {config_dir}/memory.md shared by all agents.
"""

import logging
from pathlib import Path
from typing import Optional

from cliver.util import get_config_dir

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Current profile registry — single source of truth for the active profile.
# Set by AgentCore at init, accessed by builtin tools (memory, etc.).
# ---------------------------------------------------------------------------

_current_profile: Optional["CliverProfile"] = None

# Global AgentCore reference — set by AgentCore at init.
# Tools like parallel_tasks need this to spawn concurrent LLM calls.
_agent_core = None

# Global input function — set by CLI layer to support TUI-safe input.
# Tools should call get_input_fn()("prompt") instead of raw input().
_input_fn = input  # default: standard input

# Global output function — set by CLI layer to support TUI-safe output.
# Tools should call get_output_fn()("text") instead of raw print().
_output_fn = print  # default: standard print

# Global CLI instance — set by CLI layer for TUI dialog support.
_cli_instance = None


def set_current_profile(profile: Optional["CliverProfile"]) -> None:
    """Set the active CliverProfile. Called by AgentCore at init."""
    global _current_profile
    _current_profile = profile


def get_current_profile() -> Optional["CliverProfile"]:
    """Get the active CliverProfile. Used by builtin tools that need it."""
    return _current_profile


def set_agent_core(executor) -> None:
    """Set the active AgentCore. Called by AgentCore at init."""
    global _agent_core
    _agent_core = executor


def get_agent_core():
    """Get the active AgentCore. Used by tools that need LLM calls."""
    return _agent_core


def set_input_fn(fn) -> None:
    """Set the global input function. Called by CLI layer for TUI support."""
    global _input_fn
    _input_fn = fn


def get_input_fn():
    """Get the current input function. Tools use this instead of raw input()."""
    return _input_fn


def set_output_fn(fn) -> None:
    """Set the global output function. Called by CLI layer for TUI support."""
    global _output_fn
    _output_fn = fn


def get_output_fn():
    """Get the current output function. Tools use this instead of raw print()."""
    return _output_fn


def set_cli_instance(instance) -> None:
    """Set the global CLI instance. Called by CLI layer at TUI start."""
    global _cli_instance
    _cli_instance = instance


def get_cli_instance():
    """Get the CLI instance for TUI dialog support. Returns None in non-TUI mode."""
    return _cli_instance


MAX_MEMORY_CHARS = 3000
MAX_IDENTITY_CHARS = 1500


class CliverProfile:
    """Manages instance-scoped resources for a CLIver agent.

    Directory layout:
        {config_dir}/
        ├── memory.md                          # global memory (shared)
        └── agents/{agent_name}/
            ├── memory.md                      # agent instance memory
            ├── identity.md                    # living doc: agent persona + user profile
            ├── tasks/                         # scheduled tasks (future)
            └── sessions/                      # chat sessions (future)
    """

    def __init__(self, agent_name: str, config_dir: Optional[Path] = None):
        self.agent_name = agent_name
        self.config_dir = config_dir or get_config_dir()

        # Instance-scoped paths
        self.agent_dir = self.config_dir / "agents" / agent_name
        self.memory_file = self.agent_dir / "memory.md"
        self.identity_file = self.agent_dir / "identity.md"
        self.tasks_dir = self.agent_dir / "tasks"
        self.sessions_dir = self.agent_dir / "sessions"
        self.workflows_dir = self.agent_dir / "workflows"

        # Global-scoped paths
        self.global_memory_file = self.config_dir / "memory.md"

    def ensure_dirs(self) -> None:
        """Create the agent directory structure on first use."""
        self.agent_dir.mkdir(parents=True, exist_ok=True)
        self.workflows_dir.mkdir(parents=True, exist_ok=True)

    # -- Memory ----------------------------------------------------------------

    def load_memory(self) -> str:
        """Load combined memory (global + agent instance) for system prompt injection.

        Returns global memory first, then agent-specific memory, separated
        by a heading. Truncated to MAX_MEMORY_CHARS from the end (keeping
        the most recent entries).
        """
        parts = []

        # Global memory
        global_content = self._read_file(self.global_memory_file)
        if global_content:
            parts.append(f"## Global Memory\n\n{global_content}")

        # Agent instance memory
        agent_content = self._read_file(self.memory_file)
        if agent_content:
            parts.append(f"## Agent Memory: {self.agent_name}\n\n{agent_content}")

        if not parts:
            return ""

        combined = "\n\n".join(parts)

        # Truncate from the front to keep the most recent entries
        if len(combined) > MAX_MEMORY_CHARS:
            combined = "...(truncated)\n" + combined[-MAX_MEMORY_CHARS:]

        return combined

    def append_memory(self, entry: str, scope: str = "agent", comment: str = "") -> None:
        """Append a timestamped memory entry.

        Args:
            entry: The text to remember.
            scope: "agent" for instance memory, "global" for shared memory.
            comment: Optional context about why this is being saved.
        """
        target = self._memory_target(scope)
        from cliver.util import format_datetime

        timestamp = format_datetime()

        comment_line = f" — {comment}" if comment else ""
        formatted = f"\n## {timestamp}{comment_line}\n{entry}\n"

        # Create file with header if it doesn't exist
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            header = f"# Memory: {self.agent_name}\n" if scope == "agent" else "# Global Memory\n"
            target.write_text(header + formatted, encoding="utf-8")
        else:
            with open(target, "a", encoding="utf-8") as f:
                f.write(formatted)

        logger.info(f"Memory appended ({scope}): {entry[:80]}...")

    def save_memory(self, content: str, scope: str = "agent") -> None:
        """Replace the entire memory document.

        Use when the LLM wants to reorganize, correct, or consolidate
        memory rather than just appending.

        Args:
            content: The full markdown content to write.
            scope: "agent" for instance memory, "global" for shared memory.
        """
        target = self._memory_target(scope)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        logger.info(f"Memory rewritten ({scope}): {len(content)} chars")

    def _memory_target(self, scope: str) -> Path:
        """Get the file path for the given memory scope."""
        if scope == "global":
            return self.global_memory_file
        self.ensure_dirs()
        return self.memory_file

    # -- Identity --------------------------------------------------------------

    def load_identity(self) -> str:
        """Load identity markdown for system prompt injection.

        Identity is a living document describing the agent persona and
        user profile (name, preferences, environment, etc.). Unlike memory
        (append-only), identity is rewritten as a whole when updated.

        Truncated to MAX_IDENTITY_CHARS to avoid bloating the system prompt.
        Returns empty string if no identity file exists.
        """
        content = self._read_file(self.identity_file)
        if content and len(content) > MAX_IDENTITY_CHARS:
            content = content[:MAX_IDENTITY_CHARS] + "\n...(truncated)"
        return content

    def save_identity(self, content: str) -> None:
        """Save identity markdown, replacing the entire document.

        The LLM rewrites the full identity when it learns new information
        about the user or when the agent persona changes.
        """
        self.ensure_dirs()
        self.identity_file.write_text(content, encoding="utf-8")

    # -- Rename ----------------------------------------------------------------

    def rename(self, new_name: str) -> "CliverProfile":
        """Rename this agent, moving all instance-scoped resources.

        Moves the agent directory from {agents}/{old_name} to {agents}/{new_name}.
        Returns a new CliverProfile pointing to the new location.

        Raises:
            FileNotFoundError: If the current agent directory doesn't exist.
            FileExistsError: If the target name already has a directory.
        """
        import shutil

        new_dir = self.config_dir / "agents" / new_name

        if new_dir.exists():
            raise FileExistsError(
                f"Cannot rename '{self.agent_name}' to '{new_name}': directory '{new_dir}' already exists."
            )

        if not self.agent_dir.exists():
            # No data to move — just return a fresh profile
            logger.info(f"No existing data for '{self.agent_name}', creating fresh profile for '{new_name}'")
            return CliverProfile(new_name, self.config_dir)

        shutil.move(str(self.agent_dir), str(new_dir))
        logger.info(f"Renamed agent '{self.agent_name}' to '{new_name}' (moved {self.agent_dir} → {new_dir})")

        return CliverProfile(new_name, self.config_dir)

    # -- Class Methods ---------------------------------------------------------

    @staticmethod
    def list_agents(config_dir: Path) -> list[str]:
        """List all agent names that have a directory under agents/."""
        agents_dir = config_dir / "agents"
        if not agents_dir.exists():
            return []
        return sorted(d.name for d in agents_dir.iterdir() if d.is_dir())

    # -- Helpers ---------------------------------------------------------------

    @staticmethod
    def _read_file(path: Path) -> str:
        """Read a file, returning empty string if it doesn't exist."""
        if not path.exists():
            return ""
        try:
            return path.read_text(encoding="utf-8").strip()
        except Exception as e:
            logger.warning(f"Could not read {path}: {e}")
            return ""


# Backward compatibility alias
AgentProfile = CliverProfile
