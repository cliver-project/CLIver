"""Built-in list_directory tool for listing files and directories."""

import logging
import os
from fnmatch import fnmatch
from typing import List, Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ListDirectoryInput(BaseModel):
    """Input schema for the list_directory tool."""

    path: str = Field(description="The absolute path to the directory to list (must be absolute, not relative).")
    ignore: Optional[List[str]] = Field(
        default=None,
        description='List of glob patterns to ignore (e.g., ["*.pyc", "__pycache__"]).',
    )


class ListDirectoryTool(BaseTool):
    """Lists files and subdirectories within a specified directory."""

    name: str = "list_directory"
    description: str = (
        "Lists the names of files and subdirectories directly within a specified directory path. "
        "Can optionally ignore entries matching provided glob patterns. "
        "Respects .gitignore patterns by default."
    )
    args_schema: Type[BaseModel] = ListDirectoryInput
    tags: list = ["search", "file", "list"]

    def _run(self, path: str, ignore: Optional[List[str]] = None) -> str:
        try:
            abs_path = os.path.abspath(path)
            if not os.path.exists(abs_path):
                return f"Error: Path not found: {abs_path}"
            if not os.path.isdir(abs_path):
                return f"Error: Path is not a directory: {abs_path}"

            entries = sorted(os.listdir(abs_path))

            # Load .gitignore patterns if present
            gitignore_patterns = self._load_gitignore(abs_path)
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
                    results.append(f"  {entry} ({self._format_size(size)})")

            if not results:
                return f"Directory is empty: {abs_path}"

            header = f"Directory: {abs_path} ({len(results)} entries)\n"
            return header + "\n".join(results)

        except PermissionError:
            return f"Error: Permission denied: {path}"
        except Exception as e:
            logger.error(f"Error listing directory {path}: {e}")
            return f"Error listing directory: {e}"

    def _load_gitignore(self, dir_path: str) -> list:
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

    @staticmethod
    def _format_size(size: int) -> str:
        for unit in ("B", "KB", "MB", "GB"):
            if size < 1024:
                return f"{size:.0f}{unit}" if unit == "B" else f"{size:.1f}{unit}"
            size /= 1024
        return f"{size:.1f}TB"


list_directory = ListDirectoryTool()
