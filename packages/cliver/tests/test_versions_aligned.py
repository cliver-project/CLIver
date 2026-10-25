"""The app and the core version move in lockstep.

Both packages ship from one repository and one tag, and the app pins the
core exactly, so a version skew would silently pin a stale core.
"""

from __future__ import annotations

import re
from pathlib import Path

PACKAGES = Path(__file__).resolve().parents[2]


def _pyproject(package: str) -> str:
    return (PACKAGES / package / "pyproject.toml").read_text(encoding="utf-8")


def _version(package: str) -> str:
    match = re.search(r'^version = "([^"]+)"', _pyproject(package), re.MULTILINE)
    assert match, f"no version declared in {package}/pyproject.toml"
    return match.group(1)


def test_app_and_core_share_one_version():
    assert _version("cliver") == _version("cliver-core")


def test_app_pins_the_core_at_the_shared_version():
    assert f"cliver-core=={_version('cliver-core')}" in _pyproject("cliver")
