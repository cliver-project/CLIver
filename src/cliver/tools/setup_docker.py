"""Built-in setup_docker tool for detecting and configuring container runtimes."""

import logging
import shutil
import subprocess
from typing import Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Supported container runtimes in preference order
RUNTIMES = ["docker", "podman", "nerdctl"]


class SetupDockerInput(BaseModel):
    """Input schema for the setup_docker tool. No parameters needed."""

    pass


class SetupDockerTool(BaseTool):
    """Detects and verifies the container runtime available on the system."""

    name: str = "setup_docker"
    description: str = (
        "Detects and verifies the container runtime (Docker, Podman, or nerdctl) "
        "available on the current system. Returns the runtime name, version, "
        "and whether it is ready to run containers.\n\n"
        "Use this tool before running containers to ensure the environment is set up. "
        "It checks:\n"
        "1. Which container runtime is installed\n"
        "2. Whether the runtime daemon is running\n"
        "3. Whether the current user has permission to use it"
    )
    args_schema: Type[BaseModel] = SetupDockerInput
    tags: list = ["execute", "docker", "container"]

    def _run(self) -> str:
        results = []

        for runtime in RUNTIMES:
            path = shutil.which(runtime)
            if not path:
                continue

            # Get version
            version = self._get_version(runtime)
            if not version:
                results.append(f"{runtime}: found at {path} but could not determine version")
                continue

            # Check if daemon is accessible
            ready, msg = self._check_ready(runtime)
            status = "ready" if ready else f"not ready ({msg})"
            results.append(f"{runtime}: {version} at {path} - {status}")

            if ready:
                return (
                    f"Container runtime detected and ready.\n"
                    f"  Runtime: {runtime}\n"
                    f"  Version: {version}\n"
                    f"  Path: {path}\n"
                    f"  Status: Ready to run containers\n\n"
                    f"Use the 'docker_run' tool with runtime='{runtime}' to run containers."
                )

        if results:
            return (
                "Container runtime(s) found but not ready:\n"
                + "\n".join(f"  - {r}" for r in results)
                + "\n\nPlease ensure the container daemon is running. "
                "For Docker: 'systemctl start docker' or start Docker Desktop. "
                "For Podman: it runs daemonless, check 'podman info' for issues."
            )

        return (
            "No container runtime found. Please install one of:\n"
            "  - Docker: https://docs.docker.com/get-docker/\n"
            "  - Podman: https://podman.io/getting-started/installation\n"
            "  - nerdctl: https://github.com/containerd/nerdctl"
        )

    def _get_version(self, runtime: str) -> str:
        try:
            result = subprocess.run(
                [runtime, "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.stdout.strip() if result.returncode == 0 else None
        except Exception:
            return None

    def _check_ready(self, runtime: str) -> tuple:
        """Check if the runtime can actually run containers."""
        try:
            # Try a lightweight info/ping command
            cmd = [runtime, "info"]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=15,
            )
            if result.returncode == 0:
                return True, "ok"
            return False, result.stderr.strip().split("\n")[0] if result.stderr else "unknown error"
        except subprocess.TimeoutExpired:
            return False, "timed out checking daemon"
        except Exception as e:
            return False, str(e)


setup_docker = SetupDockerTool()
