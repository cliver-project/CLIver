"""Built-in read_file tool for reading file contents."""

import logging
import os
import re

from cliver.tool import tool

logger = logging.getLogger(__name__)

DEFAULT_MAX_LINES = 2000
MAX_LINE_LENGTH = 2000

_SENSITIVE_PATTERNS = [
    r"\.env$",
    r"\.env\..+$",
    r"credentials\.json$",
    r"credentials\.yaml$",
    r"credentials\.yml$",
    r"\.secret$",
    r"\.secrets$",
    r"secrets\.json$",
    r"secrets\.yaml$",
    r"secrets\.yml$",
    r"\.pem$",
    r"\.key$",
    r"id_rsa$",
    r"id_ed25519$",
    r"id_dsa$",
    r"id_ecdsa$",
    r"\.p12$",
    r"\.pfx$",
    r"\.keystore$",
    r"\.jks$",
    r"htpasswd$",
    r"shadow$",
    r"\.netrc$",
    r"\.pgpass$",
    r"token\.json$",
    r"service[-_]?account.*\.json$",
    r"kubeconfig$",
]
_SENSITIVE_RE = re.compile("|".join(f"(?:{p})" for p in _SENSITIVE_PATTERNS), re.IGNORECASE)


def _is_sensitive_file(file_path: str) -> bool:
    basename = os.path.basename(file_path)
    return bool(_SENSITIVE_RE.search(basename))


@tool(name="Read", description="Reads and returns the content of a specified file.")
def read_file(
    file_path: str,
    offset: int | None = None,
    limit: int | None = None,
) -> list[dict]:
    """Read a file from the filesystem.

    If the file is large, the content will be truncated. The response
    indicates if truncation occurred and provides details on how to
    read more using offset/limit.
    """
    try:
        abs_path = os.path.abspath(file_path)

        if _is_sensitive_file(abs_path):
            return [{"error": f"Access denied: '{os.path.basename(abs_path)}' appears to be a sensitive file."}]

        if not os.path.exists(abs_path):
            return [{"error": f"File not found: {abs_path}"}]
        if not os.path.isfile(abs_path):
            return [{"error": f"Path is not a file: {abs_path}"}]

        with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()

        total_lines = len(all_lines)
        start = offset if offset is not None else 0
        max_lines = limit if limit is not None else DEFAULT_MAX_LINES
        end = min(start + max_lines, total_lines)
        selected = all_lines[start:end]

        result_lines = []
        for i, line in enumerate(selected):
            line_no = start + i + 1
            content = line.rstrip("\n\r")
            if len(content) > MAX_LINE_LENGTH:
                content = content[:MAX_LINE_LENGTH] + "... (truncated)"
            result_lines.append(f"{line_no:>6}\t{content}")

        result = "\n".join(result_lines)
        if end < total_lines:
            result += (
                f"\n\n--- File truncated ---\n"
                f"Showing lines {start + 1}-{end} of {total_lines} total lines.\n"
                f"Use offset={end} to read more."
            )

        return [{"text": result}]
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return [{"error": f"Error reading file: {e}"}]
