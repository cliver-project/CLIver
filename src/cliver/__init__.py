import importlib.metadata
import pathlib
import sys

from dotenv import load_dotenv

from cliver.llm import TaskExecutor
from cliver.media_handler import MultimediaResponse, MultimediaResponseHandler

load_dotenv()

# Export for public API
TaskExecutor = TaskExecutor
MultimediaResponse = MultimediaResponse
MultimediaResponseHandler = MultimediaResponseHandler

# For Python 3.11+, use built-in tomllib
if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


# noinspection PyBroadException
def get_version() -> str:
    """Get version from package metadata or pyproject.toml."""
    try:
        return importlib.metadata.version("cliver")
    except importlib.metadata.PackageNotFoundError:
        # Running from source — read pyproject.toml
        root = pathlib.Path(__file__).resolve().parents[1]
        pyproject = root / "pyproject.toml"
        try:
            with open(pyproject, "rb") as f:
                data = tomllib.load(f)
            return data["project"]["version"]
        except Exception:
            return "0.0.1+dev"


__version__ = get_version()
