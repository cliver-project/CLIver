"""Built-in run_shell_command tool for executing shell commands."""

import logging
import platform
import subprocess
from typing import Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_MS = 120000  # 2 minutes
MAX_TIMEOUT_MS = 600000  # 10 minutes
MAX_OUTPUT_LENGTH = 50000  # characters


class RunShellCommandInput(BaseModel):
    """Input schema for the run_shell_command tool."""

    command: str = Field(description="The shell command to execute.")
    description: Optional[str] = Field(
        default=None,
        description="Brief description of what this command does (5-10 words). "
        "Helps the user understand the purpose of the command.",
    )
    timeout: Optional[int] = Field(
        default=None,
        description=f"Optional timeout in milliseconds (max {MAX_TIMEOUT_MS}ms / 10 minutes). "
        f"Defaults to {DEFAULT_TIMEOUT_MS}ms (2 minutes).",
    )
    directory: Optional[str] = Field(
        default=None,
        description="Optional: The absolute path of the directory to run the command in. "
        "If not provided, the current working directory is used.",
    )


class RunShellCommandTool(BaseTool):
    """Executes shell commands with timeout and output capture."""

    name: str = "run_shell_command"
    description: str = (
        "Executes a given shell command and returns its output. "
        "Use this for terminal operations like git, npm, docker, pip, make, etc. "
        "Do NOT use this for file operations (reading, writing, searching) - "
        "use the specialized tools (read_file, write_file, grep_search, list_directory) instead.\n\n"
        "Usage notes:\n"
        "- The command is required.\n"
        f"- Timeout defaults to {DEFAULT_TIMEOUT_MS // 1000}s, max {MAX_TIMEOUT_MS // 1000}s.\n"
        "- Write a clear description of what the command does.\n"
        "- Avoid using this tool with find, grep, cat, head, tail, sed, awk - "
        "use the dedicated builtin tools instead."
    )
    args_schema: Type[BaseModel] = RunShellCommandInput
    tags: list = ["execute", "shell", "command"]

    def _run(
        self,
        command: str,
        description: Optional[str] = None,
        timeout: Optional[int] = None,
        directory: Optional[str] = None,
    ) -> str:
        timeout_ms = min(timeout or DEFAULT_TIMEOUT_MS, MAX_TIMEOUT_MS)
        timeout_s = timeout_ms / 1000.0

        system = platform.system()
        if system == "Windows":
            shell_cmd = ["cmd.exe", "/c", command]
        else:
            shell_cmd = ["bash", "-c", command]

        try:
            result = subprocess.run(
                shell_cmd,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                cwd=directory,
            )

            output_parts = []
            if result.stdout:
                stdout = result.stdout
                if len(stdout) > MAX_OUTPUT_LENGTH:
                    stdout = stdout[:MAX_OUTPUT_LENGTH] + "\n... (output truncated)"
                output_parts.append(stdout)
            if result.stderr:
                stderr = result.stderr
                if len(stderr) > MAX_OUTPUT_LENGTH:
                    stderr = stderr[:MAX_OUTPUT_LENGTH] + "\n... (stderr truncated)"
                output_parts.append(f"STDERR:\n{stderr}")

            output = "\n".join(output_parts) if output_parts else "(no output)"

            if result.returncode != 0:
                return f"Command exited with code {result.returncode}\n{output}"
            return output

        except subprocess.TimeoutExpired:
            return f"Error: Command timed out after {timeout_s:.0f} seconds."
        except FileNotFoundError:
            return f"Error: Command not found or shell not available."
        except Exception as e:
            logger.error(f"Error executing command: {e}")
            return f"Error executing command: {e}"


run_shell_command = RunShellCommandTool()
