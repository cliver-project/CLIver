"""All workspace packages share one version (release requirement)."""

import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[5]
_PACKAGES = ("cliver-core", "cliver")


def test_workspace_packages_share_one_version() -> None:
    versions: dict[str, str] = {}
    for name in _PACKAGES:
        pyproject = _ROOT / "packages" / name / "pyproject.toml"
        match = re.search(r'^version = "([^"]+)"', pyproject.read_text(encoding="utf-8"), re.MULTILINE)
        assert match is not None, f"missing version in {pyproject}"
        versions[name] = match.group(1)

    assert len(set(versions.values())) == 1, versions
