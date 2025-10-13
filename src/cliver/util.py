import logging
import os
import platform
import re
import select
import stat
import sys
import time
from pathlib import Path
from typing import Any, List, Optional, Callable, Awaitable

from cliver.constants import *

from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)

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
    except Exception as ignored:
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

async def create_tools_filter(pattern: Optional[str] = None) -> Optional[Callable[[str, List[BaseTool]], Awaitable[List[BaseTool]]]]:
    """
    Create a filter function that can be used to filter tools based on a pattern,
    while ensuring builtin tools are always included.

    Args:
        pattern: A regex pattern to match against tool names, it is in format: `mcp_server_name#tool_name`

    Returns:
        A filter function or None if no filtering is needed
    """
    if not pattern:
        # No filtering needed
        return None

    async def filter_fn(user_input: str, tools: List[BaseTool]) -> List[BaseTool]:
        if not pattern or pattern.strip() == "":
            # If no pattern is specified, return all tools
            return tools
        # TODO: we may introduce logic to relate the user_input to filter the tools
        # Convert wildcard patterns to regex patterns
        # Replace * with .* and ? with .
        regex_pattern = pattern.replace("*", ".*").replace("?", ".")
        # Add anchors to match the entire string
        regex_pattern = f"^{regex_pattern}$"

        try:
            compiled_pattern = re.compile(regex_pattern, re.IGNORECASE)
            filtered_tools = []

            for tool in tools:
                # Tool names in MCP tools follow the format "server_name#tool_name"
                tool_name = tool.name
                if compiled_pattern.search(tool_name):
                    filtered_tools.append(tool)

            logger.debug(f"Filtering tools with pattern '{pattern}', matched {len(filtered_tools)}/{len(tools)} tools")
            return filtered_tools
        except re.error as e:
            logger.error(f"Invalid regex pattern '{pattern}': {e}")
            # Return all tools if pattern is invalid
            return tools

    return filter_fn


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


def parse_key_value_options(option_list: tuple, console=None) -> dict:
    """
    Parse a list of key=value strings into a dictionary with appropriate type conversion.

    Args:
        option_list: Tuple of strings in key=value format
        console: Optional console object for printing warnings

    Returns:
        Dictionary with parsed key-value pairs
    """
    options_dict = {}

    if not option_list:
        return options_dict

    for opt in option_list:
        if "=" in opt:
            key, value = opt.split("=", 1)
            # Try to convert value to appropriate type
            try:
                # Try integer first
                if "." not in value and value.isdigit():
                    options_dict[key] = int(value)
                # Try float
                elif "." in value and all(c.isdigit() or c == "." for c in value):
                    options_dict[key] = float(value)
                # Keep as string
                else:
                    options_dict[key] = value
            except ValueError:
                options_dict[key] = value
        else:
            # Print warning if console is provided
            if console:
                console.print(
                    f"Warning: Invalid option format '{opt}', expected key=value"
                )

    return options_dict


def read_file_content(file_path: str, max_size: int = 100000) -> str:
    """
    Read the content of a file and return it as a string.

    Args:
        file_path: Path to the file to read
        max_size: Maximum file size to read in bytes (default: 100KB)

    Returns:
        String content of the file

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file is too large or cannot be read as text
    """
    import os
    import re

    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Check file size
    file_size = os.path.getsize(file_path)
    if file_size > max_size:
        raise ValueError(f"File is too large ({file_size} bytes). Maximum allowed size is {max_size} bytes.")

    # Try to read as text with common encodings
    encodings = ['utf-8', 'utf-16', 'latin-1']

    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
                return content
        except UnicodeDecodeError:
            continue
        except Exception as e:
            raise ValueError(f"Error reading file {file_path}: {str(e)}")

    raise ValueError(f"Unable to read file {file_path} with any of the supported encodings: {encodings}")