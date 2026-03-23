"""
AgentProfile — manages instance-scoped resources for a CLIver agent.

Each agent instance (identified by `agent_name`) has its own:
- memory.md   — persistent knowledge, append-only
- identity.md — living document about agent persona and user profile

These are stored under {config_dir}/agents/{agent_name}/.

Global resources (config, skills, workflows, tools) are NOT managed here.
There is also a global memory at {config_dir}/memory.md shared by all agents.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from cliver.util import get_config_dir

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Current profile registry — single source of truth for the active profile.
# Set by TaskExecutor at init, accessed by builtin tools (memory, etc.).
# ---------------------------------------------------------------------------

_current_profile: Optional["AgentProfile"] = None

# Global input function — set by CLI layer to support TUI-safe input.
# Tools should call get_input_fn()("prompt") instead of raw input().
_input_fn = input  # default: standard input


def set_current_profile(profile: Optional["AgentProfile"]) -> None:
    """Set the active AgentProfile. Called by TaskExecutor at init."""
    global _current_profile
    _current_profile = profile


def get_current_profile() -> Optional["AgentProfile"]:
    """Get the active AgentProfile. Used by builtin tools that need it."""
    return _current_profile


def set_input_fn(fn) -> None:
    """Set the global input function. Called by CLI layer for TUI support."""
    global _input_fn
    _input_fn = fn


def get_input_fn():
    """Get the current input function. Tools use this instead of raw input()."""
    return _input_fn


# Maximum characters of memory to inject into the system prompt.
# This prevents memory from consuming too much of the context window.
MAX_MEMORY_CHARS = 4000


class AgentProfile:
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

        # Global-scoped paths
        self.global_memory_file = self.config_dir / "memory.md"

    def ensure_dirs(self) -> None:
        """Create the agent directory structure on first use."""
        self.agent_dir.mkdir(parents=True, exist_ok=True)

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
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

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

        Returns empty string if no identity file exists.
        """
        return self._read_file(self.identity_file)

    def save_identity(self, content: str) -> None:
        """Save identity markdown, replacing the entire document.

        The LLM rewrites the full identity when it learns new information
        about the user or when the agent persona changes.
        """
        self.ensure_dirs()
        self.identity_file.write_text(content, encoding="utf-8")

    # -- Rename ----------------------------------------------------------------

    def rename(self, new_name: str) -> "AgentProfile":
        """Rename this agent, moving all instance-scoped resources.

        Moves the agent directory from {agents}/{old_name} to {agents}/{new_name}.
        Returns a new AgentProfile pointing to the new location.

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
            return AgentProfile(new_name, self.config_dir)

        shutil.move(str(self.agent_dir), str(new_dir))
        logger.info(f"Renamed agent '{self.agent_name}' to '{new_name}' (moved {self.agent_dir} → {new_dir})")

        return AgentProfile(new_name, self.config_dir)

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
