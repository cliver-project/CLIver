"""Chat template store — reads/writes templates from YAML files.

Templates are loaded from two sources (higher priority overrides lower):
1. Project-level: .cliver/templates.yaml
2. User-level: ~/.cliver/templates/templates.yaml
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

DEFAULT_TEMPLATES: list[dict] = [
    {
        "id": "make-slides",
        "label": "Make slides",
        "system_prompt": "You are a presentation design expert. Create well-structured slide content.",
        "skills": ["brainstorm"],
    },
    {
        "id": "create-website",
        "label": "Create a website",
        "system_prompt": "You are a senior full-stack web developer.",
        "skills": ["write-plan", "execute-plan"],
    },
    {
        "id": "write-novel",
        "label": "Write a novel",
        "system_prompt": "You are a creative fiction writer.",
        "skills": ["brainstorm"],
    },
    {
        "id": "create-video",
        "label": "Create a video",
        "system_prompt": "You are a video production expert. Plan and script video content.",
        "skills": ["brainstorm"],
    },
    {
        "id": "analyze-data",
        "label": "Analyze data",
        "system_prompt": "You are a data scientist. Analyze data and provide insights.",
        "skills": [],
    },
]


class ChatTemplate(BaseModel):
    """A reusable chat configuration template."""

    id: str = Field(description="Unique template identifier (i18n key: templates.{id})")
    label: str = Field(default="", description="Display label — fallback when i18n key is missing")
    system_prompt: str = ""
    skills: List[str] = Field(default_factory=list)
    model: Optional[str] = None
    knowledge_base: Optional[str] = None
    description: Optional[str] = None

    def model_dump(self, **kwargs):
        data = super().model_dump(**kwargs)
        # Remove None values for cleaner YAML
        return {k: v for k, v in data.items() if v is not None}


class ChatTemplateStore:
    """Loads and saves chat templates from YAML files."""

    def __init__(self, config_dir: Path):
        self._config_dir = Path(config_dir)
        self._user_file = self._config_dir / "templates" / "templates.yaml"
        self._project_file: Optional[Path] = None

    def set_project_root(self, path: Optional[Path]) -> None:
        if path:
            self._project_file = Path(path) / ".cliver" / "templates.yaml"
        else:
            self._project_file = None

    def list_all(self) -> List[ChatTemplate]:
        """Return merged templates: project overrides user, user overrides defaults."""
        merged: Dict[str, dict] = {}

        # Layer 1: built-in defaults
        for item in DEFAULT_TEMPLATES:
            merged[item["id"]] = dict(item)

        # Layer 2: user-level
        user_templates = self._load_yaml(self._user_file)
        for item in user_templates:
            merged[item["id"]] = {**merged.get(item["id"], {}), **item}

        # Layer 3: project-level
        if self._project_file:
            project_templates = self._load_yaml(self._project_file)
            for item in project_templates:
                merged[item["id"]] = {**merged.get(item["id"], {}), **item}

        return [ChatTemplate(**v) for v in merged.values()]

    def get(self, template_id: str) -> Optional[ChatTemplate]:
        for t in self.list_all():
            if t.id == template_id:
                return t
        return None

    def save_to_user(self, template: ChatTemplate) -> None:
        """Save or update a template in the user-level file."""
        templates = self._load_yaml(self._user_file)
        existing = {t["id"]: t for t in templates}
        existing[template.id] = template.model_dump(exclude_none=True)
        self._write_yaml(self._user_file, list(existing.values()))

    def delete_from_user(self, template_id: str) -> bool:
        """Delete a template from the user-level file. Cannot delete built-in defaults."""
        templates = self._load_yaml(self._user_file)
        new_list = [t for t in templates if t.get("id") != template_id]
        if len(new_list) == len(templates):
            return False
        self._write_yaml(self._user_file, new_list)
        return True

    def _load_yaml(self, path: Path) -> list[dict]:
        if not path.exists():
            return []
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            items = data.get("templates", []) if isinstance(data, dict) else []
            if not isinstance(items, list):
                return []
            return [i for i in items if isinstance(i, dict) and i.get("id")]
        except Exception as e:
            logger.warning("Failed to load templates from %s: %s", path, e)
            return []

    def _write_yaml(self, path: Path, templates: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        content = yaml.dump({"templates": templates}, allow_unicode=True, default_flow_style=False, sort_keys=False)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(content, encoding="utf-8")
        tmp.rename(path)
