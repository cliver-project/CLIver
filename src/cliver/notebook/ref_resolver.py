"""Reference resolver for ${cell_id.outputs.field} syntax."""

from __future__ import annotations

import re
from typing import Any, Dict, List

_REF_PATTERN = re.compile(r"\$\{([^}]+)\}")


def extract_ref_paths(template: str) -> List[str]:
    """Extract all reference paths from a template string."""
    return _REF_PATTERN.findall(template)


def resolve_value(path: str, variables: Dict[str, Any]) -> Any:
    """Resolve a dot-path to a Python value.

    path: "cell_id.outputs.field" or "cell_id.outputs.data.0.title"
    variables: {"cell_id": {"outputs": {...}}, ...}

    Returns the actual Python value (dict, list, int, str, etc.).
    Raises ValueError if path cannot be resolved.
    """
    parts = path.split(".")
    current = variables

    for i, part in enumerate(parts):
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, list):
            try:
                idx = int(part)
                current = current[idx]
            except (ValueError, IndexError):
                traversed = ".".join(parts[:i])
                raise ValueError(
                    f"Reference '${{{path}}}' not found. '{part}' is not a valid index in list at '{traversed}'"
                ) from None
        else:
            traversed = ".".join(parts[:i])
            available = list(current.keys()) if isinstance(current, dict) else []
            hint = f" Available keys: {available}" if available else ""
            raise ValueError(f"Reference '${{{path}}}' not found. Key '{part}' does not exist at '{traversed}'.{hint}")

    return current


def resolve_refs(template: str, variables: Dict[str, Any]) -> str:
    """Resolve all ${...} references in a template string.

    Each reference is resolved to its string representation.
    For Python-native access, use resolve_value() directly.
    """
    if not template or "${" not in template:
        return template

    def replacer(match: re.Match) -> str:
        path = match.group(1)
        value = resolve_value(path, variables)
        return str(value)

    return _REF_PATTERN.sub(replacer, template)
