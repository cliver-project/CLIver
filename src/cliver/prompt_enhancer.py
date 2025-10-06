"""
Prompt enhancement utilities for CLIver.

This module provides functionality for loading and applying skill sets and templates
to enhance prompts with predefined structures and parameters.

Skill Sets
----------
Skill sets are YAML files that define a set of tools and system messages that can be
applied to enhance the behavior of the AI agent. They support parameter substitution
to make them reusable across different contexts.

Syntax for SkillSet YAML files:
```yaml
description: A brief description of what this skill set does
system_message: |
  System message that will be prepended to the conversation.
  Can include parameter placeholders like ${param_name}.
tools:
  - name: tool_name
    mcp_server: server_name
    description: Description of what this tool does
    parameters:
      param1: ${param_value}  # Parameter placeholder
      param2: static_value    # Static value
parameters:
  param_name: default_value   # Default values for parameter substitution
```

Example SkillSet:
```yaml
description: File system operations
system_message: |
  You are an expert file system assistant. You have access to tools for
  file operations. When asked to perform operations, be precise and
  confirm destructive actions.
tools:
  - name: read_file
    mcp_server: file_system
    description: Read the contents of a file
    parameters:
      path: ${file_path}
  - name: write_file
    mcp_server: file_system
    description: Write content to a file
    parameters:
      path: ${file_path}
      content: ${content}
parameters:
  file_path: /default/path.txt
  content: Default content
```

Templates
---------
Templates are text or markdown files that can contain placeholders for dynamic content.
They support both simple placeholders and placeholders with default values.

Syntax for Templates:
- Simple placeholder: `{placeholder_name}`
- Placeholder with default: `{placeholder_name:default_value}`

Example Template (.txt or .md):
```
You are asked to analyze the following code:
{user_input}

Please provide a detailed review focusing on:
1. Code quality and best practices
2. Potential bugs or issues
3. Performance considerations

Analysis:
```

Template with defaults:
```
Task: {task:Code Review}
Context: {context:No additional context provided}

Instructions:
{user_input}

Response:
```
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from langchain_core.messages import BaseMessage

from cliver.util import get_config_dir

logger = logging.getLogger(__name__)


class SkillSet:
    """
    Represents a skill set with tools and parameters.

    A SkillSet encapsulates a collection of tools and a system message that can be
    applied to enhance the behavior of the AI agent. It supports parameter substitution
    to make skill sets reusable across different contexts.

    Attributes:
        name (str): The name of the skill set
        description (str): A brief description of what this skill set does
        system_message (str): System message that will be prepended to the conversation
        tools (List[Dict]): List of tools available in this skill set
        parameters (Dict): Default parameter values for substitution
    """

    def __init__(self, name: str, data: Dict[str, Any]):
        """
        Initialize a SkillSet.

        Args:
            name: The name of the skill set
            data: Dictionary containing the skill set data loaded from YAML
        """
        self.name = name
        self.data = data
        self.description = data.get("description", "")
        self.system_message = data.get("system_message", "")
        self.tools = data.get("tools", [])
        self.parameters = data.get("parameters", {})

    def get_system_message(self, params: Dict[str, str] = None) -> str:
        """
        Get the system message with parameter substitution.

        This method performs parameter substitution in the system message, replacing
        placeholders like ${param_name} with actual values from the provided params
        or default parameters defined in the skill set.

        Args:
            params: Dictionary of parameter values to substitute

        Returns:
            The system message with all parameter placeholders replaced
        """
        message = self.system_message
        if params:
            for key, value in params.items():
                message = message.replace(f"${{{key}}}", value)
        # Also substitute with skill set default parameters
        for key, value in self.parameters.items():
            message = message.replace(f"${{{key}}}", str(value))
        return message

    def get_tools_with_params(
        self, params: Dict[str, str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get tools with parameter substitution.

        This method returns the tools defined in the skill set with parameter
        substitution applied to tool parameters. Placeholders like ${param_name}
        are replaced with actual values from the provided params or default
        parameters defined in the skill set.

        Each tool will have an mcp_server field, and the name will be formatted
        as "mcp_server#tool_name" when sent to the LLM.

        Args:
            params: Dictionary of parameter values to substitute

        Returns:
            List of tools with parameter substitution applied
        """
        tools = []
        for tool in self.tools:
            # Create a copy of the tool definition
            tool_copy = tool.copy()
            # Substitute parameters in tool parameters
            if "parameters" in tool_copy:
                for param_key, param_value in tool_copy["parameters"].items():
                    if (
                        isinstance(param_value, str)
                        and param_value.startswith("${")
                        and param_value.endswith("}")
                    ):
                        param_name = param_value[2:-1]  # Remove ${ and }
                        if params and param_name in params:
                            tool_copy["parameters"][param_key] = params[param_name]
                        elif param_name in self.parameters:
                            tool_copy["parameters"][param_key] = self.parameters[
                                param_name
                            ]
            tools.append(tool_copy)
        return tools


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
        """
        Initialize a Template.

        Args:
            name: The name of the template
            content: The template content with placeholders
        """
        self.name = name
        self.content = content

    def apply(self, params: Dict[str, str] = None) -> str:
        """
        Apply parameters to the template.

        This method performs parameter substitution in the template content, replacing
        placeholders like {param_name} with actual values from the provided params.
        It also handles placeholders with default values like {param_name:default_value}.

        Args:
            params: Dictionary of parameter values to substitute

        Returns:
            The template content with all parameter placeholders replaced
        """
        result = self.content
        if params:
            for key, value in params.items():
                result = result.replace(f"{{{key}}}", value)
        # Handle default values: {key:default_value}
        import re

        pattern = r"\{([^}:]+):([^}]+)\}"
        matches = re.findall(pattern, result)
        for key, default_value in matches:
            if not params or key not in params:
                result = result.replace(f"{{{key}:{default_value}}}", default_value)
            else:
                result = result.replace(f"{{{key}:{default_value}}}", params[key])
        return result


def load_skill_set(skill_set_name: str) -> Optional[SkillSet]:
    """
    Load a skill set by name.

    This function loads a skill set from a YAML file, searching in the following directories
    in order of priority:
    1. .cliver directory in the current working directory
    2. Current working directory
    3. Global config directory

    The function supports both .yaml and .yml file extensions.

    Args:
        skill_set_name: Name of the skill set to load (without extension)

    Returns:
        SkillSet object or None if not found

    Example:
        >>> skill_set = load_skill_set("file_operations")
        >>> if skill_set:
        ...     print(skill_set.description)
    """
    extensions = ["yaml", "yml"]
    content = _load_with_extensions_of_dirs(skill_set_name, extensions)
    if content:
        data = yaml.safe_load(content)
        return SkillSet(skill_set_name, data)
    logger.warning(f"Skill set {skill_set_name} not found")
    return None


def load_template(template_name: str) -> Optional[Template]:
    """
    Load a template by name.

    This function loads a template from a text or markdown file, searching in the following
    directories in order of priority:
    1. .cliver directory in the current working directory
    2. Current working directory
    3. Global config directory

    The function supports both .md and .txt file extensions, with .md files taking precedence
    when both extensions exist in the same directory.

    Args:
        template_name: Name of the template to load (without extension)

    Returns:
        Template object or None if not found

    Example:
        >>> template = load_template("code_review")
        >>> if template:
        ...     content = template.apply({"user_input": "def hello(): pass"})
    """
    # Supported template file extensions
    extensions = [".md", ".txt"]
    content = _load_with_extensions_of_dirs(template_name, extensions)
    if content:
        return Template(template_name, content)
    logger.warning(f"Template {template_name} not found")
    return None


def _load_with_extensions_of_dirs(
    file_name: str = None, extensions: List[str] = None, dirs: List[Path] = None
) -> Optional[str]:
    """
    Load file content with extension and directory search.

    This helper function searches for a file with the given name across multiple directories
    and extensions, returning the content of the first matching file found.

    Args:
        file_name: Name of the file to load (without extension)
        extensions: List of file extensions to try (e.g., [".md", ".txt"])
        dirs: List of directories to search in order of priority

    Returns:
        File content as string or None if not found
    """
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
    return None


def apply_skill_sets_and_template(
    user_input: str,
    messages: List[BaseMessage],
    skill_set_names: List[str] = None,
    template_name: Optional[str] = None,
    params: Dict[str, str] = None,
) -> tuple[List[BaseMessage], List[Dict[str, Any]]]:
    """
    Apply skill sets and template to enhance the messages.

    This function enhances the provided messages by:
    1. Appending system messages from skill sets to the beginning of the messages
    2. Applying templates to the last human message in the messages
    3. Collecting tools from skill sets for use in tool calling

    Args:
        user_input: Original user input to enhance
        messages: List of BaseMessage objects to enhance
        skill_set_names: List of skill set names to apply
        template_name: Name of template to apply
        params: Parameters for substitution in skill sets and templates

    Returns:
        Tuple of (enhanced_messages, tools_to_include)

    Example:
        >>> from langchain_core.messages import HumanMessage
        >>> messages = [HumanMessage(content="Analyze this code")]
        >>> enhanced_messages, tools = apply_skill_sets_and_template(
        ...     "Analyze this code",
        ...     messages,
        ...     skill_set_names=["code_analysis"],
        ...     template_name="code_review",
        ...     params={"language": "python"}
        ... )
    """
    from langchain_core.messages import SystemMessage, HumanMessage

    tools_to_include = []

    # Apply skill sets if specified
    if skill_set_names:
        for skill_set_name in skill_set_names:
            skill_set = load_skill_set(skill_set_name)
            if skill_set:
                # Add system message to enhanced messages
                system_message = skill_set.get_system_message(params)
                if system_message:
                    # we append to the messages because we want the default system messages always be the first.
                    messages.append(SystemMessage(content=system_message))

                # Collect tools
                tools = skill_set.get_tools_with_params(params)
                tools_to_include.extend(tools)
            else:
                logger.warning(f"Skill set {skill_set_name} not found")

    # Apply template to the last human message if specified
    if template_name and messages:
        template = load_template(template_name)
        if template:
            enhanced_content = template.apply(params)
            # Replace placeholders in template with original content if needed
            if "{user_input}" in enhanced_content:
                enhanced_content = enhanced_content.replace("{user_input}", user_input)
            elif "{input}" in enhanced_content:
                enhanced_content = enhanced_content.replace("{input}", user_input)
            messages.append(HumanMessage(content=enhanced_content))
        else:
            logger.warning(f"Template {template_name} not found")

    return messages, tools_to_include
