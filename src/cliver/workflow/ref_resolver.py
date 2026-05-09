"""${ref} substitution and auto-inject context for workflow prompts."""

import json
import re
from typing import Any, Dict, List

_REF_PATTERN = re.compile(r"\$\{([^}]+)\}")


def _traverse(path: str, data: Dict[str, Any]) -> Any:
    """Walk a dot-path (with optional [N] indexing) into nested dicts/lists."""
    parts = re.split(r"\.|\[(\d+)\]", path)
    parts = [p for p in parts if p is not None and p != ""]
    current: Any = data
    for part in parts:
        if current is None:
            return None
        if isinstance(current, dict):
            current = current.get(part)
        elif isinstance(current, (list, tuple)):
            try:
                current = current[int(part)]
            except (ValueError, IndexError):
                return None
        else:
            return None
    return current


def _resolve_single_ref(ref_path: str, state: Dict[str, Any]) -> str:
    """Resolve a single ${ref.path} against state."""
    if ref_path.startswith("inputs."):
        value = _traverse(ref_path[len("inputs.") :], state.get("inputs", {}))
    else:
        value = _traverse(ref_path, state.get("steps", {}))
    if value is None:
        return ""
    return str(value)


def resolve_refs(text: str, state: Dict[str, Any]) -> str:
    """Replace all ${ref} patterns in text with values from state."""
    if "${" not in text:
        return text
    return _REF_PATTERN.sub(lambda m: _resolve_single_ref(m.group(1), state), text)


def build_auto_context(depends_on: List[str], state: Dict[str, Any]) -> str:
    """Build auto-inject context block from dependency outputs."""
    steps_data = state.get("steps", {})
    blocks = []
    for dep_id in depends_on:
        dep_output = steps_data.get(dep_id)
        if dep_output is None:
            continue
        blocks.append(f"[Output from step '{dep_id}']:\n{json.dumps(dep_output, ensure_ascii=False, indent=2)}")
    return "\n\n".join(blocks)
