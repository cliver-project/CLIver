"""Built-in write_file tool for writing content to files."""

import logging
import os

from cliver.tool import tool

logger = logging.getLogger(__name__)


@tool(
    name="Write",
    description=(
        "Writes content to a specified file in the local filesystem. "
        "Creates the file and any parent directories if they don't exist. "
        "If the file already exists, it will be overwritten. "
        "There is no length limit on the content. For very large files, "
        "write the complete content in a single call — do not truncate."
    ),
)
def write_file(file_path: str, content: str) -> list[dict]:
    """Writes content to a specified file in the local filesystem.

    Creates the file and any parent directories if they don't exist.
    If the file already exists, it will be overwritten.
    There is no length limit on the content.

    Args:
        file_path: The absolute path to the file to write to
            (e.g., '/home/user/project/file.txt'). Relative paths are not supported.
        content: The content to write to the file.
    """
    try:
        # Convert non-string content to JSON (safety check)
        if not isinstance(content, str):
            import json

            content = json.dumps(content, ensure_ascii=False, indent=2)

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
        return [{"text": f"{action} file: {abs_path} ({line_count} lines written)"}]

    except Exception as e:
        logger.error(f"Error writing file {file_path}: {e}")
        return [{"error": f"Error writing file: {e}"}]
