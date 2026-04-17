"""Built-in write_file tool for writing content to files."""

import logging
import os
from typing import Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class WriteFileInput(BaseModel):
    """Input schema for the write_file tool."""

    file_path: str = Field(
        description="The absolute path to the file to write to "
        "(e.g., '/home/user/project/file.txt'). Relative paths are not supported."
    )
    content: str = Field(description="The content to write to the file.")


class WriteFileTool(BaseTool):
    """Writes content to a specified file in the local filesystem."""

    name: str = "Write"
    description: str = (
        "Writes content to a specified file in the local filesystem. "
        "Creates the file and any parent directories if they don't exist. "
        "If the file already exists, it will be overwritten."
    )
    args_schema: Type[BaseModel] = WriteFileInput
    tags: list = ["write", "file", "edit"]

    def _run(self, file_path: str, content: str) -> str:
        try:
            abs_path = os.path.abspath(file_path)

            # Create parent directories if they don't exist
            parent_dir = os.path.dirname(abs_path)
            if parent_dir and not os.path.exists(parent_dir):
                os.makedirs(parent_dir, exist_ok=True)

            existed = os.path.exists(abs_path)
            with open(abs_path, "w", encoding="utf-8") as f:
                f.write(content)

            line_count = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
            action = "Updated" if existed else "Created"
            return f"{action} file: {abs_path} ({line_count} lines written)"

        except Exception as e:
            logger.error(f"Error writing file {file_path}: {e}")
            return f"Error writing file: {e}"


write_file = WriteFileTool()
