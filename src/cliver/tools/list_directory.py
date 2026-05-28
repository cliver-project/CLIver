"""Built-in list_directory tool for listing files and directories."""

import logging
import os
from fnmatch import fnmatch
from typing import Optional

from cliver.tool import tool

logger = logging.getLogger(__name__)


def _load_gitignore(dir_path: str) -> list:
    """Load patterns from .gitignore if present."""
    gitignore_path = os.path.join(dir_path, ".gitignore")
    patterns = []
    if os.path.isfile(gitignore_path):
        try:
            with open(gitignore_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        patterns.append(line)
        except Exception:
            pass
    return patterns


def _format_size(size: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.0f}{unit}" if unit == "B" else f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"


@tool(
    name="LS",
    description=(
        "Lists the names of files and subdirectories directly within a specified directory path. "
        "Can optionally ignore entries matching provided glob patterns. "
        "Respects .gitignore patterns by default."
    ),
)
def list_directory(path: str = ".", ignore: Optional[list[str]] = None) -> list[dict]:
    """Lists files and subdirectories within a specified directory.

    Can optionally ignore entries matching provided glob patterns.
    Respects .gitignore patterns by default.

    Args:
        path: Directory path to list. Defaults to current working directory.
            Use relative paths when possible.
        ignore: List of glob patterns to ignore (e.g., ["*.pyc", "__pycache__"]).
    """
    try:
        abs_path = os.path.abspath(path)
        if not os.path.exists(abs_path):
            return [{"error": f"Path not found: {abs_path}"}]
        if not os.path.isdir(abs_path):
            return [{"error": f"Path is not a directory: {abs_path}"}]

        entries = sorted(os.listdir(abs_path))

        # Load .gitignore patterns if present
        gitignore_patterns = _load_gitignore(abs_path)
        all_ignore = list(gitignore_patterns)
        if ignore:
            all_ignore.extend(ignore)

        results = []
        for entry in entries:
            # Skip entries matching ignore patterns
            if any(fnmatch(entry, pat) for pat in all_ignore):
                continue

            full_path = os.path.join(abs_path, entry)
            if os.path.isdir(full_path):
                results.append(f"  {entry}/")
            else:
                size = os.path.getsize(full_path)
                results.append(f"  {entry} ({_format_size(size)})")

        if not results:
            return [{"text": f"Directory is empty: {abs_path}"}]

        header = f"Directory: {abs_path} ({len(results)} entries)\n"
        return [{"text": header + "\n".join(results)}]

    except PermissionError:
        return [{"error": f"Permission denied: {path}"}]
    except Exception as e:
        logger.error(f"Error listing directory {path}: {e}")
        return [{"error": f"Error listing directory: {e}"}]
