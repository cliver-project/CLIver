"""Built-in run_shell_command tool for executing shell commands."""

import logging
import os
import platform
import subprocess
from pathlib import Path
from typing import Optional

from cliver_core.policy import current_policy
from cliver_core.tool import tool
from cliver_core.util import _safe_int

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_MS = 120000  # 2 minutes
MAX_TIMEOUT_MS = 600000  # 10 minutes
MAX_OUTPUT_LENGTH = 50000  # characters


@tool(
    name="Bash",
    description=(
        "Executes a given shell command and returns its output. "
        "Use this for terminal operations like git, npm, docker, pip, make, etc. "
        "Do NOT use this for file operations (reading, writing, searching) - "
        "use the specialized tools (Read, Write, Grep, LS) instead. "
        "Do NOT use this to run Python code - use Exec instead.\n\n"
        "Usage notes:\n"
        f"- timeout is in milliseconds (default {DEFAULT_TIMEOUT_MS}, max {MAX_TIMEOUT_MS}).\n"
        "- directory sets the working directory (defaults to the current directory).\n"
        "- Avoid using this tool with find, grep, cat, head, tail, sed, awk - "
        "use the dedicated builtin tools instead."
    ),
)
def run_shell_command(
    command: str,
    description: Optional[str] = None,
    timeout: Optional[int] = None,
    directory: Optional[str] = None,
) -> list[dict]:
    """Execute a shell command with timeout and output capture.

    Args:
        command: The shell command to execute.
        description: Brief description of what this command does (5-10 words).
        timeout: Optional timeout in milliseconds (max 600000ms / 10 minutes).
            Defaults to 120000ms (2 minutes).
        directory: Optional absolute path to run the command in.
            If not provided, the current working directory is used.
    """
    timeout_ms = _safe_int(timeout, DEFAULT_TIMEOUT_MS)
    timeout_ms = min(timeout_ms, MAX_TIMEOUT_MS)
    timeout_s = timeout_ms / 1000.0

    # Ambient execution policy (AgentCore chat/stream call): a policy env
    # replaces the inherited environment; a pinned cwd wins over the
    # ``directory`` param (logged so the override is never silent).
    policy = current_policy()
    env = None
    run_cwd = directory
    if policy is not None:
        env = policy.env.build(os.environ)
        if policy.cwd is not None:
            if directory and Path(directory).resolve() != Path(policy.cwd).resolve():
                logger.info("Bash: execution policy pins cwd to %s, ignoring directory=%s", policy.cwd, directory)
            run_cwd = str(policy.cwd)

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
            cwd=run_cwd,
            env=env,
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
            return [{"text": f"Command exited with code {result.returncode}\n{output}"}]
        return [{"text": output}]

    except subprocess.TimeoutExpired:
        return [{"error": f"Command timed out after {timeout_s:.0f} seconds."}]
    except FileNotFoundError:
        return [{"error": "Command not found or shell not available."}]
    except Exception as e:
        logger.error(f"Error executing command: {e}")
        return [{"error": f"Error executing command: {e}"}]
