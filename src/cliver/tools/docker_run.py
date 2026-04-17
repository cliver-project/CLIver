"""Built-in docker_run tool for running one-time containers."""

import logging
import shutil
import subprocess
from typing import Dict, List, Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S = 300  # 5 minutes
MAX_TIMEOUT_S = 600  # 10 minutes
MAX_OUTPUT_LENGTH = 50000


class DockerRunInput(BaseModel):
    """Input schema for the docker_run tool."""

    image: str = Field(
        description="The container image to run (e.g., 'python:3.12-slim', 'node:20-alpine', 'ubuntu:22.04')."
    )
    command: Optional[str] = Field(
        default=None,
        description="The command to execute inside the container. "
        "If not provided, the image's default entrypoint is used.",
    )
    volumes: Optional[List[str]] = Field(
        default=None,
        description="List of volume mounts in 'host_path:container_path' format "
        "(e.g., ['/tmp/data:/data', '/home/user/code:/app:ro']).",
    )
    env: Optional[Dict[str, str]] = Field(
        default=None,
        description='Environment variables to set in the container (e.g., {"MY_VAR": "value", "DEBUG": "true"}).',
    )
    workdir: Optional[str] = Field(
        default=None,
        description="Working directory inside the container.",
    )
    runtime: Optional[str] = Field(
        default=None,
        description="Container runtime to use: 'docker', 'podman', or 'nerdctl'. "
        "If not specified, auto-detects the available runtime.",
    )
    timeout: Optional[int] = Field(
        default=None,
        description=f"Timeout in seconds (max {MAX_TIMEOUT_S}). Defaults to {DEFAULT_TIMEOUT_S}s.",
    )


class DockerRunTool(BaseTool):
    """Runs a one-time container for a specific purpose."""

    name: str = "Docker"
    description: str = (
        "Runs a one-time (ephemeral) container using Docker, Podman, or nerdctl. "
        "The container is automatically removed after execution (--rm flag).\n\n"
        "Use this tool when you need to:\n"
        "1. Run a command in an isolated environment\n"
        "2. Execute tools that are only available as container images\n"
        "3. Test code in a specific runtime environment\n"
        "4. Run database migrations, linters, formatters in isolated containers\n\n"
        "The container runs in the foreground and returns stdout/stderr output. "
        "The runtime is auto-detected if not specified."
    )
    args_schema: Type[BaseModel] = DockerRunInput
    tags: list = ["execute", "docker", "container"]

    def _run(
        self,
        image: str,
        command: Optional[str] = None,
        volumes: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
        workdir: Optional[str] = None,
        runtime: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> str:
        # Detect runtime
        rt = self._detect_runtime(runtime)
        if not rt:
            return "Error: No container runtime found. Install Docker, Podman, or nerdctl."

        timeout_s = min(timeout or DEFAULT_TIMEOUT_S, MAX_TIMEOUT_S)

        # Build the command
        cmd = [rt, "run", "--rm"]

        # Add volume mounts
        if volumes:
            for vol in volumes:
                cmd.extend(["-v", vol])

        # Add environment variables
        if env:
            for key, value in env.items():
                cmd.extend(["-e", f"{key}={value}"])

        # Add working directory
        if workdir:
            cmd.extend(["-w", workdir])

        # Add image
        cmd.append(image)

        # Add command (split if it's a string)
        if command:
            cmd.extend(["sh", "-c", command])

        try:
            logger.info(f"Running container: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_s,
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
                return f"Container exited with code {result.returncode}\nImage: {image}\nRuntime: {rt}\n\n{output}"

            return f"Container completed successfully.\nImage: {image}\nRuntime: {rt}\n\n{output}"

        except subprocess.TimeoutExpired:
            return f"Error: Container timed out after {timeout_s} seconds."
        except Exception as e:
            logger.error(f"Error running container: {e}")
            return f"Error running container: {e}"

    def _detect_runtime(self, preferred: Optional[str] = None) -> Optional[str]:
        """Detect available container runtime."""
        if preferred:
            if shutil.which(preferred):
                return preferred
            return None

        for rt in ["docker", "podman", "nerdctl"]:
            if shutil.which(rt):
                return rt
        return None


docker_run = DockerRunTool()
