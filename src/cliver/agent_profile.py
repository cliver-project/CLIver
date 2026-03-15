"""
AgentProfile — manages instance-scoped resources for a CLIver agent.

Each agent instance (identified by `agent_name`) has its own:
- memory.md   — persistent knowledge, append-only
- identity.yaml — persona and preferences

These are stored under {config_dir}/agents/{agent_name}/.

Global resources (config, skills, workflows, tools) are NOT managed here.
There is also a global memory at {config_dir}/memory.md shared by all agents.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml

from cliver.util import get_config_dir

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Current profile registry — single source of truth for the active profile.
# Set by TaskExecutor at init, accessed by builtin tools (memory, etc.).
# ---------------------------------------------------------------------------

_current_profile: Optional["AgentProfile"] = None


def set_current_profile(profile: Optional["AgentProfile"]) -> None:
    """Set the active AgentProfile. Called by TaskExecutor at init."""
    global _current_profile
    _current_profile = profile


def get_current_profile() -> Optional["AgentProfile"]:
    """Get the active AgentProfile. Used by builtin tools that need it."""
    return _current_profile


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
            ├── identity.yaml                  # agent persona
            ├── tasks/                         # scheduled tasks (future)
            └── sessions/                      # chat sessions (future)
    """

    def __init__(self, agent_name: str, config_dir: Optional[Path] = None):
        self.agent_name = agent_name
        self.config_dir = config_dir or get_config_dir()

        # Instance-scoped paths
        self.agent_dir = self.config_dir / "agents" / agent_name
        self.memory_file = self.agent_dir / "memory.md"
        self.identity_file = self.agent_dir / "identity.yaml"
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

    def append_memory(self, entry: str, scope: str = "agent") -> None:
        """Append a timestamped memory entry.

        Args:
            entry: The text to remember.
            scope: "agent" for instance memory, "global" for shared memory.
        """
        if scope == "global":
            target = self.global_memory_file
        else:
            self.ensure_dirs()
            target = self.memory_file

        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        formatted = f"\n## {timestamp}\n{entry}\n"

        # Create file with header if it doesn't exist
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            header = f"# Memory: {self.agent_name}\n" if scope == "agent" else "# Global Memory\n"
            target.write_text(header + formatted, encoding="utf-8")
        else:
            with open(target, "a", encoding="utf-8") as f:
                f.write(formatted)

        logger.info(f"Memory entry appended to {scope} memory: {entry[:80]}...")

    # -- Identity --------------------------------------------------------------

    def load_identity(self) -> dict:
        """Load agent identity (persona, preferences) from identity.yaml.

        Returns an empty dict if no identity file exists.
        """
        if not self.identity_file.exists():
            return {}

        try:
            content = self.identity_file.read_text(encoding="utf-8")
            return yaml.safe_load(content) or {}
        except Exception as e:
            logger.warning(f"Could not load identity for '{self.agent_name}': {e}")
            return {}

    def save_identity(self, identity: dict) -> None:
        """Save agent identity to identity.yaml."""
        self.ensure_dirs()
        self.identity_file.write_text(
            yaml.dump(identity, default_flow_style=False, allow_unicode=True),
            encoding="utf-8",
        )

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
                f"Cannot rename '{self.agent_name}' to '{new_name}': "
                f"directory '{new_dir}' already exists."
            )

        if not self.agent_dir.exists():
            # No data to move — just return a fresh profile
            logger.info(f"No existing data for '{self.agent_name}', creating fresh profile for '{new_name}'")
            return AgentProfile(new_name, self.config_dir)

        shutil.move(str(self.agent_dir), str(new_dir))
        logger.info(f"Renamed agent '{self.agent_name}' to '{new_name}' (moved {self.agent_dir} → {new_dir})")

        return AgentProfile(new_name, self.config_dir)

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
