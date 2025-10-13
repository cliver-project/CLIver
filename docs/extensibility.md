---
title: Extensibility
description: How to extend CLIver functionality and use it as a Python library
---

# Extensibility Guide

CLIver is designed with extensibility in mind, allowing you to customize its functionality, add new features, and integrate it seamlessly into your own Python applications. This guide covers how to extend CLIver and use it as a Python library.

## Using CLIver as a Python Library

CLIver provides a rich Python API that allows you to integrate LLM functionality directly into your applications.

### Basic LLM Client

```python
from cliver import LLMClient

# Initialize the client with your preferred model
client = LLMClient(
    model="gpt-4-turbo",
    api_key="your-api-key",  # Or set via environment variable
    temperature=0.7,
    max_tokens=1000
)

# Send a chat message
response = client.chat("Hello, how can I use CLIver in my Python project?")
print(response)
```

### Streaming Responses

For real-time applications, you can stream responses from the LLM:

```python
from cliver import LLMClient

client = LLMClient(model="gpt-4-turbo")

# Stream the response
for chunk in client.stream_chat([
    {"role": "user", "content": "Write a poem about programming"}
]):
    print(chunk, end='', flush=True)
```

### Batch Processing

Process multiple requests efficiently:

```python
from cliver import LLMClient

client = LLMClient(model="gpt-4-turbo")

prompts = [
    "Summarize the benefits of Python",
    "Explain the concept of neural networks",
    "Describe best practices for code review"
]

# Process all prompts in batch
responses = client.batch_chat(prompts)
for i, response in enumerate(responses):
    print(f"Response {i+1}: {response}")
```

## Custom Commands

You can add custom commands to CLIver's CLI interface by creating command modules.

### Creating a Custom Command

Create a Python file in your project (e.g., `my_custom_command.py`):

```python
import click
from cliver.cli.base import cli

@cli.command()
@click.argument('text')
@click.option('--style', default='formal', help='Style of response (formal, casual, humorous)')
def custom_assistant(text, style):
    """A custom assistant command that responds in different styles."""
    from cliver import LLMClient
    
    styles = {
        'formal': 'Respond in a formal, professional tone.',
        'casual': 'Respond in a casual, friendly tone.',
        'humorous': 'Respond with humor and wit where appropriate.'
    }
    
    client = LLMClient(model='gpt-4-turbo')
    system_prompt = styles.get(style, styles['formal'])
    
    response = client.chat([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text}
    ])
    
    click.echo(response)
```

To register this command with CLIver, you need to set up a plugin system:

```python
# setup.py or pyproject.toml entry points
entry_points={
    'cliver.plugins': [
        'my_custom_command = my_custom_command',
    ]
}
```

## Custom LLM Adapters

Create custom adapters for LLM providers not natively supported by CLIver:

```python
from cliver.llms.base import LLMAdapter
from typing import List, Dict, Any

class CustomLLMAdapter(LLMAdapter):
    def __init__(self, api_key: str, model: str, **kwargs):
        super().__init__(model, **kwargs)
        self.api_key = api_key
        self.base_url = kwargs.get('base_url', 'https://api.custom-llm.com/v1')
    
    async def chat_async(self, messages: List[Dict[str, str]], **kwargs) -> str:
        import aiohttp
        
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            'model': self.model,
            'messages': messages,
            **kwargs
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(f'{self.base_url}/chat/completions', 
                                   json=payload, headers=headers) as response:
                data = await response.json()
                return data['choices'][0]['message']['content']
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        # Synchronous implementation
        import requests
        
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            'model': self.model,
            'messages': messages,
            **kwargs
        }
        
        response = requests.post(f'{self.base_url}/chat/completions', 
                                json=payload, headers=headers)
        return response.json()['choices'][0]['message']['content']

# Register the custom adapter
from cliver.llms.registry import register_adapter

register_adapter('custom-llm', CustomLLMAdapter)
```

## Custom Workflow Actions

Extend the workflow system with custom actions:

```python
from cliver.workflows.base import WorkflowAction
from cliver.workflows.registry import register_action

class CustomFileAction(WorkflowAction):
    name = "file.custom_process"
    description = "Perform custom processing on a file"

    async def execute_async(self, parameters: Dict[str, Any]) -> Any:
        import os
        
        file_path = parameters.get('path')
        operation = parameters.get('operation', 'read')
        
        if operation == 'read':
            with open(file_path, 'r') as f:
                return f.read()
        elif operation == 'word_count':
            with open(file_path, 'r') as f:
                content = f.read()
                words = len(content.split())
                return {'word_count': words, 'file_path': file_path}
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    def execute(self, parameters: Dict[str, Any]) -> Any:
        # Synchronous implementation
        return self.execute_async(parameters)

# Register the custom action
register_action(CustomFileAction())
```

The custom action can then be used in workflows:

```yaml
name: Custom File Processor
description: Uses custom file processing action
inputs:
  file_path:
    type: string
    required: true

steps:
  - name: count_words
    action: file.custom_process
    parameters:
      path: "{{ inputs.file_path }}"
      operation: "word_count"

outputs:
  word_count: "{{ steps.count_words.output.word_count }}"
```

## MCP Server Integration

You can create custom MCP servers and integrate them with CLIver:

```python
# custom_mcp_server.py
from mcp.server import Server
from mcp.types import Prompt
import asyncio

server = Server("my-custom-mcp-server")

@server.collect_prompts()
async def handle_prompts() -> list[Prompt]:
    return [
        Prompt(
            name="code-review-prompt",
            description="A prompt for code review",
            template="Please review this code for security vulnerabilities and best practices:\n{{code}}",
            arguments=[
                {
                    "name": "code",
                    "description": "The code to review",
                    "required": True
                }
            ]
        )
    ]

async def main():
    # Use uvicorn or similar to serve the MCP server
    # This is a simplified example
    from mcp.server.stdio import run_server
    await run_server(server)

if __name__ == "__main__":
    asyncio.run(main())
```

## Configuration Extensions

Create custom configuration extensions to support additional features:

```python
from cliver.config.schema import ConfigSchema, register_config_schema

class CustomConfigSchema(ConfigSchema):
    def __init__(self):
        super().__init__()
        self.schema = {
            "type": "object",
            "properties": {
                "custom_feature": {
                    "type": "object",
                    "properties": {
                        "enabled": {"type": "boolean", "default": False},
                        "api_key": {"type": "string"},
                        "endpoint": {"type": "string", "format": "uri"}
                    },
                    "required": ["enabled"]
                }
            }
        }

# Register the custom schema
register_config_schema("custom", CustomConfigSchema())
```

## Plugin Development

Develop plugins to package and distribute your extensions:

```python
# cliver_plugin_example/__init__.py
from . import custom_commands, custom_adapters, custom_actions

def init_plugin():
    """Initialize the plugin and register all extensions"""
    from cliver.llms.registry import register_adapter
    from cliver.workflows.registry import register_action
    from cliver.cli.base import cli
    
    # Register custom adapters
    register_adapter('my-custom-llm', custom_adapters.MyCustomAdapter)
    
    # Register custom actions
    register_action(custom_actions.MyCustomAction())
    
    # The CLI commands are automatically registered via click decorators

__all__ = ['init_plugin']
```

Package your plugin as a Python package:

```toml
# pyproject.toml
[project]
name = "cliver-plugin-example"
version = "0.1.0"
dependencies = ["cliver"]

[project.entry-points."cliver.plugins"]
example = "cliver_plugin_example:init_plugin"
```

## Integration Examples

### Integration with Data Science Workflows

```python
from cliver import LLMClient
import pandas as pd

class DataAnalyzer:
    def __init__(self, model="gpt-4-turbo"):
        self.client = LLMClient(model=model)
    
    def analyze_dataset(self, df: pd.DataFrame, analysis_type: str = "overview"):
        """Analyze a pandas DataFrame using LLM"""
        # Convert DataFrame to a readable format
        head = df.head().to_string()
        info = df.info(buf=None)
        
        prompt = f"""
        Perform {analysis_type} analysis on the following dataset:
        
        Dataset Head:
        {head}
        
        Dataset Info:
        {info}
        
        Provide insights about the data, including patterns, anomalies, and suggestions for further analysis.
        """
        
        response = self.client.chat(prompt)
        return response

# Usage
df = pd.read_csv("data.csv")
analyzer = DataAnalyzer()
insights = analyzer.analyze_dataset(df, "statistical")
print(insights)
```

### Integration with DevOps Tools

```python
from cliver import LLMClient
import subprocess

class DevOpsAssistant:
    def __init__(self):
        self.client = LLMClient(model="gpt-4-turbo")
    
    def explain_command(self, command: str) -> str:
        """Explain what a command does"""
        response = self.client.chat([
            {
                "role": "system", 
                "content": "Explain Unix/Linux commands in a clear, helpful way."
            },
            {
                "role": "user", 
                "content": f"Explain what this command does: {command}"
            }
        ])
        return response
    
    def suggest_command(self, task: str) -> str:
        """Suggest a command for a given task"""
        response = self.client.chat([
            {
                "role": "system", 
                "content": "Suggest appropriate Unix/Linux commands for the given task."
            },
            {
                "role": "user", 
                "content": f"What command should I use to: {task}"
            }
        ])
        return response

# Usage
assistant = DevOpsAssistant()
explanation = assistant.explain_command("find /var/log -name '*.log' -mtime +7 -delete")
print(explanation)
```

## Testing Custom Extensions

When developing custom extensions, it's important to test them properly:

```python
# test_custom_extensions.py
import unittest
from cliver import LLMClient

class TestCustomExtensions(unittest.TestCase):
    def setUp(self):
        self.client = LLMClient(model="gpt-4-turbo")
    
    def test_custom_workflow_action(self):
        # Test your custom workflow action
        from my_custom_action import CustomFileAction
        
        action = CustomFileAction()
        result = action.execute({
            'path': '/path/to/test/file.txt',
            'operation': 'word_count'
        })
        
        self.assertIn('word_count', result)
        self.assertIsInstance(result['word_count'], int)
    
    def test_custom_llm_adapter(self):
        # Test your custom LLM adapter
        from my_custom_adapter import CustomLLMAdapter
        
        adapter = CustomLLMAdapter(api_key="test-key", model="test-model")
        response = adapter.chat([{"role": "user", "content": "Hello"}])
        
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)

if __name__ == '__main__':
    unittest.main()
```

## Next Steps

Now that you understand how to extend CLIver, see our [Roadmap](roadmap.md) for upcoming features and how you can contribute to the project.