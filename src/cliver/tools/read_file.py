"""Built-in read_file tool for reading file contents."""

import logging
import os
from typing import Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Maximum lines to return by default to avoid overwhelming the LLM context
DEFAULT_MAX_LINES = 2000
# Maximum characters per line before truncation
MAX_LINE_LENGTH = 2000


class ReadFileInput(BaseModel):
    """Input schema for the read_file tool."""

    file_path: str = Field(
        description="The absolute path to the file to read (e.g., '/home/user/project/file.txt'). "
        "Relative paths are not supported. You must provide an absolute path."
    )
    offset: Optional[int] = Field(
        default=None,
        description="Optional: The 0-based line number to start reading from. "
        "Use with 'limit' to paginate through large files.",
    )
    limit: Optional[int] = Field(
        default=None,
        description="Optional: Maximum number of lines to read. "
        "Use with 'offset' to paginate through large files. "
        f"If omitted, reads up to {DEFAULT_MAX_LINES} lines.",
    )


class ReadFileTool(BaseTool):
    """Reads and returns the content of a specified file."""

    name: str = "read_file"
    description: str = (
        "Reads and returns the content of a specified file. "
        "If the file is large, the content will be truncated. "
        "The tool's response will clearly indicate if truncation has occurred "
        "and will provide details on how to read more of the file using the 'offset' and 'limit' parameters. "
        "For text files, it can read specific line ranges."
    )
    args_schema: Type[BaseModel] = ReadFileInput
    tags: list = ["read", "file"]

    def _run(self, file_path: str, offset: Optional[int] = None, limit: Optional[int] = None) -> str:
        try:
            abs_path = os.path.abspath(file_path)
            if not os.path.exists(abs_path):
                return f"Error: File not found: {abs_path}"
            if not os.path.isfile(abs_path):
                return f"Error: Path is not a file: {abs_path}"

            with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                all_lines = f.readlines()

            total_lines = len(all_lines)
            start = offset if offset is not None else 0
            max_lines = limit if limit is not None else DEFAULT_MAX_LINES
            end = min(start + max_lines, total_lines)
            selected = all_lines[start:end]

            # Truncate long lines
            result_lines = []
            for i, line in enumerate(selected):
                line_no = start + i + 1  # 1-based line numbers
                content = line.rstrip("\n\r")
                if len(content) > MAX_LINE_LENGTH:
                    content = content[:MAX_LINE_LENGTH] + "... (truncated)"
                result_lines.append(f"{line_no:>6}\t{content}")

            result = "\n".join(result_lines)

            # Add truncation notice if applicable
            if end < total_lines:
                result += (
                    f"\n\n--- File truncated ---\n"
                    f"Showing lines {start + 1}-{end} of {total_lines} total lines.\n"
                    f"Use offset={end} to read more."
                )

            return result

        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return f"Error reading file: {e}"


read_file = ReadFileTool()
