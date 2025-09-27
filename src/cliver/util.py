import logging
import os
import platform
import select
import stat
import sys
import time
from pathlib import Path
from typing import Any, Callable

from cliver.constants import *


def get_config_dir() -> Path:
    """
    Returns the config directory for CLiver.
    It can be overridden by the environment of 'CLIVER_CONF_DIR'
    """
    config_dir = os.getenv(CONFIG_DIR)
    if config_dir is not None:
        return Path(config_dir)
    system = platform.system()
    if system == "Windows":
        return Path(os.getenv("APPDATA")) / APP_NAME
    elif system == "Darwin":
        return Path.home() / "Library" / "Application Support" / APP_NAME
    else:
        return Path(os.getenv("XDG_CONFIG_HOME", Path.home() / ".config")) / APP_NAME


def stdin_is_piped():
    try:
        fd = sys.stdin.fileno()
        mode = os.fstat(fd).st_mode
        # True if stdin is a FIFO (pipe)
        return not os.isatty(fd) and stat.S_ISFIFO(mode)
    except Exception:
        return True


def read_piped_input(timeout=5.0):
    """Non-blocking read from stdin if data is available."""
    if select.select([sys.stdin], [], [], timeout)[0]:
        return sys.stdin.read()
    return None


def retry_with_confirmation(
    func: Callable[..., Any],
    *args,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    confirm_on_retry: bool = True,
    confirmation_prompt: str = "Operation failed. Retry?",
    **kwargs,
) -> Any:
    """
    Retry a function until it succeeds or reaches max retries.

    Args:
        func: The function to retry
        *args: Positional arguments to pass to the function
        max_retries: Maximum number of retry attempts (default: 3)
        retry_delay: Delay between retries in seconds (default: 1.0)
        confirm_on_retry: Whether to ask for confirmation before
            retrying (default: True)
        confirmation_prompt: Prompt to show when asking for
            confirmation (default: "Operation failed. Retry?")
        **kwargs: Keyword arguments to pass to the function

    Returns:
        The result of the function call if successful

    Raises:
        The last exception raised by the function if all
        retries are exhausted
    """
    import asyncio

    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            logging.debug(f"Attempt {attempt + 1} failed with error: {str(e)}")

            # If this was the last attempt, don't ask to retry
            if attempt == max_retries:
                break

            # Ask for confirmation before retrying (if enabled)
            if confirm_on_retry:
                if not _confirm_tool_execution(f"{confirmation_prompt} (y/n): "):
                    raise e

            # Wait before retrying
            if retry_delay > 0:
                time.sleep(retry_delay)

    # If we get here, all retries were exhausted
    raise last_exception


async def retry_with_confirmation_async(
    func: Callable[..., Any],
    *args,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    confirm_on_retry: bool = True,
    confirmation_prompt: str = "Operation failed. Retry?",
    **kwargs,
) -> Any:
    """
    Retry an async function until it succeeds or reaches max retries.

    Args:
        func: The async function to retry
        *args: Positional arguments to pass to the function
        max_retries: Maximum number of retry attempts (default: 3)
        retry_delay: Delay between retries in seconds (default: 1.0)
        confirm_on_retry: Whether to ask for confirmation before
            retrying (default: True)
        confirmation_prompt: Prompt to show when asking for
            confirmation (default: "Operation failed. Retry?")
        **kwargs: Keyword arguments to pass to the function

    Returns:
        The result of the function call if successful

    Raises:
        The last exception raised by the function if all
        retries are exhausted
    """
    import asyncio

    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            logging.debug(f"Attempt {attempt + 1} failed with error: {str(e)}")

            # If this was the last attempt, don't ask to retry
            if attempt == max_retries:
                break

            # Ask for confirmation before retrying (if enabled)
            if confirm_on_retry:
                if not _confirm_tool_execution(f"{confirmation_prompt} (y/n): "):
                    raise e

            # Wait before retrying
            if retry_delay > 0:
                await asyncio.sleep(retry_delay)

    # If we get here, all retries were exhausted
    raise last_exception


def _confirm_tool_execution(prompt="Are you sure? (y/n): ") -> bool:
    """Helper function to get user confirmation."""
    while True:
        response = input(prompt).strip().lower()
        if response in ["y", "yes"]:
            return True
        elif response in ["n", "no"]:
            return False


def read_context_files(base_path: str = ".", file_filter: list[str] = None) -> str:
    """
    Read context from markdown files if they exist.

    Args:
        base_path: The base path to look for context files (default: current directory)
        file_filter: List of filenames to look for (default: ["Cliver.md"])

    Returns:
        A string containing the context from the files, or empty string if none found
    """
    import os

    context = ""

    # Files to look for in order of priority
    if file_filter is None:
        context_files = ["Cliver.md"]
    else:
        context_files = file_filter

    for filename in context_files:
        file_path = os.path.join(base_path, filename)
        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    if content.strip():
                        context += f"\n# Content from {filename}:\n{content}\n"
            except Exception as e:
                # Log error but continue with other files
                import logging

                logging.warning(f"Could not read {filename}: {e}")

    return context.strip()
