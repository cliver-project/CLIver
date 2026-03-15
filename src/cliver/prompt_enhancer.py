"""
Prompt enhancement utilities for CLIver.

This module provides functionality for loading and applying templates
to enhance prompts with predefined structures and parameters.

Templates
---------
Templates are text or markdown files that can contain placeholders for dynamic content.
They support parameter substitution using Jinja2 templating.

Syntax for Templates:
- Simple placeholder: `{{ placeholder_name }}`
- Placeholder with default: `{{ placeholder_name | default('default_value') }}`

Example Template (.txt or .md):
```
You are asked to analyze the following code:
{{ user_input }}

Please provide a detailed review focusing on:
1. Code quality and best practices
2. Potential bugs or issues
3. Performance considerations

Analysis:
```
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.messages import BaseMessage

from cliver.template_utils import render_template_if_needed
from cliver.util import get_config_dir

logger = logging.getLogger(__name__)


class Template:
    """
    Represents a template with placeholders.

    A Template is a text or markdown file that can contain placeholders for dynamic content.
    It supports both simple placeholders and placeholders with default values.

    Attributes:
        name (str): The name of the template
        content (str): The template content with placeholders
    """

    def __init__(self, name: str, content: str):
        self.name = name
        self.content = content

    def apply(self, params: Dict[str, str] = None) -> str:
        """Apply parameters to the template using Jinja2 templating."""
        if not self.content:
            return self.content
        return render_template_if_needed(self.content, params)


def load_template(template_name: str) -> Optional[Template]:
    """
    Load a template by name.

    Searches in the following directories in order of priority:
    1. .cliver directory in the current working directory
    2. Current working directory
    3. Global config directory

    Supports both .md and .txt file extensions.

    Args:
        template_name: Name of the template to load (without extension)

    Returns:
        Template object or None if not found
    """
    extensions = [".md", ".txt"]
    content = _load_with_extensions_of_dirs(template_name, extensions)
    if content:
        return Template(template_name, content)
    logger.warning(f"Template {template_name} not found")
    return None


def _load_with_extensions_of_dirs(
    file_name: str = None, extensions: List[str] = None, dirs: List[Path] = None
) -> Optional[str]:
    """Load file content with extension and directory search."""
    if not file_name or len(file_name) == 0:
        return None
    if not extensions or len(extensions) == 0:
        extensions = ["md", "txt"]
    if not dirs or len(dirs) == 0:
        dirs = [Path.cwd() / ".cliver", Path.cwd(), get_config_dir()]
    for ext in extensions:
        for _dir in dirs:
            file_path = _dir / f"{file_name}{ext}"
            if file_path.exists():
                try:
                    with open(file_path, "r") as f:
                        content = f.read()
                    return content
                except Exception as e:
                    logger.error(f"Error loading file {file_path}: {e}")
                    raise e
    return None


def apply_template(
    user_input: str,
    messages: List[BaseMessage],
    template_name: Optional[str] = None,
    params: Dict[str, str] = None,
) -> List[BaseMessage]:
    """
    Apply a template to enhance the messages.

    Args:
        user_input: Original user input to enhance
        messages: List of BaseMessage objects to enhance
        template_name: Name of template to apply
        params: Parameters for substitution in templates

    Returns:
        Enhanced messages list
    """
    from langchain_core.messages import HumanMessage

    if template_name and messages:
        template = load_template(template_name)
        if template:
            template_params = params.copy() if params else {}
            template_params["user_input"] = user_input
            template_params["input"] = user_input

            enhanced_content = template.apply(template_params)
            messages.append(HumanMessage(content=enhanced_content))
        else:
            logger.warning(f"Template {template_name} not found")

    return messages


# Backward compatibility alias
def apply_skill_sets_and_template(
    user_input: str,
    messages: List[BaseMessage],
    skill_set_names: List[str] = None,
    template_name: Optional[str] = None,
    params: Dict[str, str] = None,
) -> tuple[List[BaseMessage], List[Dict[str, Any]]]:
    """Deprecated: use apply_template() instead. Skill sets are now LLM-driven via the skill tool."""
    if skill_set_names:
        logger.warning(
            "YAML skill sets are deprecated. Skills are now activated by the LLM "
            "via the builtin 'skill' tool using SKILL.md files."
        )
    messages = apply_template(user_input, messages, template_name, params)
    return messages, []
