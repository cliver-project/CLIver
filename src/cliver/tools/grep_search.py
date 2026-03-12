"""Built-in grep_search tool for searching file contents with regex patterns."""

import logging
import os
import re
import subprocess
from typing import Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

DEFAULT_MATCH_LIMIT = 100


class GrepSearchInput(BaseModel):
    """Input schema for the grep_search tool."""

    pattern: str = Field(description="The regular expression pattern to search for in file contents.")
    path: str = Field(
        default=".",
        description="File or directory to search in. Defaults to current working directory.",
    )
    glob: Optional[str] = Field(
        default=None,
        description='Glob pattern to filter files (e.g. "*.py", "*.{ts,tsx}").',
    )
    limit: Optional[int] = Field(
        default=None,
        description=f"Limit output to first N matching lines. Defaults to {DEFAULT_MATCH_LIMIT}.",
    )


class GrepSearchTool(BaseTool):
    """Searches file contents using regex patterns."""

    name: str = "grep_search"
    description: str = (
        "A powerful search tool for finding patterns in file contents. "
        "Supports full regex syntax (e.g., 'log.*Error', 'function\\s+\\w+'). "
        "Filter files with glob parameter (e.g., '*.py', '**/*.tsx'). "
        "Case-insensitive by default. "
        "Returns matching lines with file paths and line numbers."
    )
    args_schema: Type[BaseModel] = GrepSearchInput
    tags: list = ["search", "grep", "find"]

    def _run(
        self,
        pattern: str,
        path: str = ".",
        glob: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> str:
        max_results = limit if limit is not None else DEFAULT_MATCH_LIMIT
        search_path = os.path.abspath(path)

        if not os.path.exists(search_path):
            return f"Error: Path not found: {search_path}"

        # Try ripgrep first, fall back to Python regex search
        try:
            return self._rg_search(pattern, search_path, glob, max_results)
        except FileNotFoundError:
            logger.debug("ripgrep (rg) not found, using Python fallback")
            return self._python_search(pattern, search_path, glob, max_results)

    def _rg_search(self, pattern: str, path: str, glob: Optional[str], limit: int) -> str:
        cmd = ["rg", "--no-heading", "--line-number", "--ignore-case", "--max-count", str(limit)]
        if glob:
            cmd.extend(["--glob", glob])
        cmd.extend(["--", pattern, path])

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            if len(lines) >= limit:
                return "\n".join(lines[:limit]) + f"\n\n--- Results limited to {limit} matches ---"
            return "\n".join(lines) if lines[0] else "No matches found."
        elif result.returncode == 1:
            return "No matches found."
        else:
            raise FileNotFoundError("rg failed")

    def _python_search(self, pattern: str, path: str, glob_pattern: Optional[str], limit: int) -> str:
        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            return f"Error: Invalid regex pattern: {e}"

        matches = []
        search_files = []

        if os.path.isfile(path):
            search_files = [path]
        else:
            import fnmatch

            for root, _, files in os.walk(path):
                # Skip hidden and common non-text directories
                if any(part.startswith(".") for part in root.split(os.sep)):
                    continue
                for fname in files:
                    if glob_pattern and not fnmatch.fnmatch(fname, glob_pattern):
                        continue
                    search_files.append(os.path.join(root, fname))

        for file_path in search_files:
            if len(matches) >= limit:
                break
            try:
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    for line_no, line in enumerate(f, 1):
                        if regex.search(line):
                            matches.append(f"{file_path}:{line_no}:{line.rstrip()}")
                            if len(matches) >= limit:
                                break
            except (OSError, UnicodeDecodeError):
                continue

        if not matches:
            return "No matches found."

        result = "\n".join(matches)
        if len(matches) >= limit:
            result += f"\n\n--- Results limited to {limit} matches ---"
        return result


grep_search = GrepSearchTool()
