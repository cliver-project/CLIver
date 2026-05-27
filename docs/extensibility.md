---
title: Extensibility
description: How to extend CLIver functionality and use it as a Python library
---

# Extensibility Guide

CLIver is designed with extensibility in mind, allowing you to customize its functionality, add new features, and integrate it seamlessly into your own Python applications. This guide covers how to extend CLIver and use it as a Python library.

## Using CLIver as a Python Library

CLIver's core engine (`AgentCore`) is designed to be **independent of the CLI layer**. It has no dependencies on terminal I/O, prompt_toolkit, or Rich — making it suitable for embedding in web services, automation scripts, or other applications.

All features — LLM inference, tool calling, permissions, skills, and memory — work identically whether invoked from the CLI or from your own Python code.

### Basic LLM Inference

The simplest way to use CLIver is to create an `AgentCore` instance and call `process_user_input()`:

```python
import asyncio
from cliver import AgentCore
from cliver.config import ModelConfig, ModelOptions

# Configure at least one LLM model
llm_models = {
    "openai/gpt-4": ModelConfig(
        name="openai/gpt-4",
        provider="openai",
        options=ModelOptions(
            temperature=0.7,
            max_tokens=4096
        )
    )
}

# Create the agent core (no MCP servers for this simple example)
agent = AgentCore(
    llm_models=llm_models,
    mcp_servers={},
    default_model="openai/gpt-4"
)

# Run a query
async def main():
    result = await agent.process_user_input(
        user_input="What is the capital of France?",
        max_iterations=10
    )
    print(result.content)

asyncio.run(main())
```

### Streaming Responses

For real-time applications, you can stream responses from the LLM:

```python
import asyncio
from cliver import AgentCore
from cliver.config import ModelConfig, ModelOptions

llm_models = {
    "openai/gpt-4": ModelConfig(
        name="openai/gpt-4",
        provider="openai"
    )
}

agent = AgentCore(
    llm_models=llm_models,
    mcp_servers={},
    default_model="openai/gpt-4"
)

async def stream_example():
    async for chunk in agent.stream_user_input(
        user_input="Write a short poem about Python",
        max_iterations=10
    ):
        if chunk.content:
            print(chunk.content, end="", flush=True)
    print()  # newline at end

asyncio.run(stream_example())
```

### Working with Images, Audio, and Video

AgentCore supports multimodal inputs through the `images`, `audio_files`, and `video_files` parameters:

```python
import asyncio
from cliver import AgentCore
from cliver.config import ModelConfig

llm_models = {
    "openai/gpt-4o": ModelConfig(
        name="openai/gpt-4o",
        provider="openai"
    )
}

agent = AgentCore(
    llm_models=llm_models,
    mcp_servers={},
    default_model="openai/gpt-4o"
)

async def analyze_image():
    result = await agent.process_user_input(
        user_input="What's in this image?",
        images=["/path/to/image.png"]
    )
    print(result.content)

asyncio.run(analyze_image())
```

### Conversation History

Maintain multi-turn conversations by passing conversation history:

```python
import asyncio
from langchain_core.messages import HumanMessage, AIMessage
from cliver import AgentCore
from cliver.config import ModelConfig

llm_models = {
    "openai/gpt-4": ModelConfig(
        name="openai/gpt-4",
        provider="openai"
    )
}

agent = AgentCore(
    llm_models=llm_models,
    mcp_servers={},
    default_model="openai/gpt-4"
)

async def conversation():
    # Start with empty history
    history = []
    
    # First turn
    result = await agent.process_user_input(
        user_input="My name is Alice",
        conversation_history=history
    )
    print(f"Assistant: {result.content}")
    
    # Add to history
    history.append(HumanMessage(content="My name is Alice"))
    history.append(result)
    
    # Second turn - agent should remember the name
    result = await agent.process_user_input(
        user_input="What's my name?",
        conversation_history=history
    )
    print(f"Assistant: {result.content}")

asyncio.run(conversation())
```

## AgentCore API Reference

### Constructor Parameters

```python
AgentCore(
    llm_models: Dict[str, ModelConfig],
    mcp_servers: Dict[str, Dict],
    default_model: Optional[str] = None,
    user_agent: Optional[str] = None,
    agent_name: str = "CLIver",
    on_tool_event: Optional[ToolEventHandler] = None,
    agent_profile: Optional[CliverProfile] = None,
    token_tracker: Optional[TokenTracker] = None,
    permission_manager: Optional[PermissionManager] = None,
    on_permission_prompt: Optional[Callable] = None,
    enabled_toolsets: Optional[List[str]] = None,
    skill_auto_learn: bool = False,
    model_auto_fallback: bool = True
)
```

**Key Parameters:**

- `llm_models`: Dictionary of model configurations (model name → ModelConfig)
- `mcp_servers`: Dictionary of MCP server configurations
- `default_model`: Default model to use when none specified
- `on_tool_event`: Callback for tool execution events (start, end, error)
- `enabled_toolsets`: List of tool groups to enable (e.g., `["core", "web", "browser"]`)
- `model_auto_fallback`: Automatically fall back to other models on errors

### Core Methods

#### `process_user_input()`

Process a user query and return the final response.

```python
async def process_user_input(
    user_input: str,
    images: List[str] = None,
    audio_files: List[str] = None,
    video_files: List[str] = None,
    files: List[str] = None,
    max_iterations: int = 50,
    confirm_tool_exec: Optional[Callable[[str], bool]] = None,
    model: str = None,
    system_message_appender: Optional[Callable[[], str]] = None,
    filter_tools: Optional[Callable] = None,
    enhance_prompt: Optional[Callable] = None,
    template: Optional[str] = None,
    params: dict = None,
    conversation_history: Optional[List[BaseMessage]] = None,
    timeout_s: Optional[int] = None,
    outputs_dir: Optional[str] = None
) -> BaseMessage
```

#### `stream_user_input()`

Stream the response in real-time (same parameters as `process_user_input()`):

```python
async for chunk in agent.stream_user_input(user_input="...", ...):
    # Process each chunk as it arrives
    print(chunk.content, end="", flush=True)
```

#### `process_skill()`

Execute a skill by name:

```python
result = await agent.process_skill(
    skill_name="code-review",
    user_input="Review the changes in main.py",
    model="openai/gpt-4"
)
```

## Custom Tool Registration

You can register custom tools by creating LangChain `BaseTool` instances and adding them to the tool registry.

### Creating a Custom Tool

```python
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

class WeatherInput(BaseModel):
    city: str = Field(description="City name to get weather for")
    units: str = Field(default="celsius", description="Temperature units")

class WeatherTool(BaseTool):
    name: str = "GetWeather"
    description: str = "Get current weather for a city"
    args_schema: type[BaseModel] = WeatherInput
    
    def _run(self, city: str, units: str = "celsius") -> str:
        # Your implementation here
        return f"Weather in {city}: 22°{units[0].upper()}, sunny"
    
    async def _arun(self, city: str, units: str = "celsius") -> str:
        # Async version (optional)
        return self._run(city, units)

# Register the tool
from cliver.tool_registry import tool_registry

weather_tool = WeatherTool()
tool_registry._tools.append(weather_tool)
tool_registry._tools_by_name[weather_tool.name] = weather_tool
```

### Using Custom Tools with AgentCore

```python
import asyncio
from cliver import AgentCore
from cliver.config import ModelConfig

async def main():
    llm_models = {
        "openai/gpt-4": ModelConfig(
            name="openai/gpt-4",
            provider="openai"
        )
    }
    
    agent = AgentCore(
        llm_models=llm_models,
        mcp_servers={},
        default_model="openai/gpt-4"
    )
    
    # The custom tool is now available to the agent
    result = await agent.process_user_input(
        user_input="What's the weather in Paris?"
    )
    print(result.content)

asyncio.run(main())
```

## Custom Commands

You can add custom commands to CLIver's CLI interface by creating command modules.

### Creating a Custom Command

Create a Python file in `~/.cliver/commands/my_command.py`:

```python
"""
Custom command example.
"""

import click
from cliver.cli import Cliver, pass_cliver
from cliver.commands import wants_help

@click.command("greet")
@click.argument("name", required=False)
@pass_cliver
def greet_command(cliver: Cliver, name: str = None):
    """Greet a user by name."""
    
    # Check if user wants help
    if wants_help([name]):
        cliver.output("Usage: /greet <name>")
        cliver.output("Example: /greet Alice")
        return
    
    if not name:
        cliver.output("Please provide a name: /greet <name>")
        return
    
    cliver.output(f"Hello, {name}!")

# Register the command
def register_commands(cliver_cli):
    """Called by CLIver to register this command."""
    cliver_cli.add_command(greet_command)
```

The command will be loaded automatically by CLIver and available as `/greet <name>`.

### Advanced Command Example

Commands can interact with the agent core and use async operations:

```python
"""
Advanced command that uses the agent.
"""

import asyncio
import click
from cliver.cli import Cliver, pass_cliver

@click.command("summarize")
@click.argument("file_path")
@pass_cliver
def summarize_command(cliver: Cliver, file_path: str):
    """Summarize a file using the LLM."""
    
    # Read file content
    try:
        with open(file_path, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        cliver.output(f"File not found: {file_path}")
        return
    except Exception as e:
        cliver.output(f"Error reading file: {e}")
        return
    
    # Use the agent to summarize
    async def run_summary():
        result = await cliver.agent_core.process_user_input(
            user_input=f"Summarize this text:\n\n{content}",
            max_iterations=5
        )
        return result.content
    
    cliver.output("Generating summary...")
    summary = asyncio.run(run_summary())
    cliver.output(f"\nSummary:\n{summary}")

def register_commands(cliver_cli):
    cliver_cli.add_command(summarize_command)
```

## MCP Server Integration

AgentCore provides programmatic access to MCP (Model Context Protocol) servers.

### Configuring MCP Servers

```python
from cliver import AgentCore
from cliver.config import ModelConfig

llm_models = {
    "openai/gpt-4": ModelConfig(
        name="openai/gpt-4",
        provider="openai"
    )
}

mcp_servers = {
    "filesystem": {
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
    },
    "github": {
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-github"],
        "env": {
            "GITHUB_TOKEN": "your-token-here"
        }
    }
}

agent = AgentCore(
    llm_models=llm_models,
    mcp_servers=mcp_servers,
    default_model="openai/gpt-4"
)
```

### Using MCP Tools

MCP tools are automatically discovered and made available to the agent:

```python
import asyncio

async def main():
    # MCP tools are automatically available
    result = await agent.process_user_input(
        user_input="List files in the /tmp directory"
    )
    print(result.content)

asyncio.run(main())
```

### Direct MCP Server Access

You can also call MCP servers directly:

```python
import asyncio
from cliver import AgentCore
from cliver.config import ModelConfig

async def main():
    agent = AgentCore(
        llm_models={"openai/gpt-4": ModelConfig(name="openai/gpt-4", provider="openai")},
        mcp_servers={
            "filesystem": {
                "transport": "stdio",
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
            }
        }
    )
    
    # Get the MCP caller
    mcp_caller = agent.get_mcp_caller()
    
    # Get available tools from a specific server
    tools = await mcp_caller.get_mcp_tools(server="filesystem")
    print(f"Available tools: {[tool.name for tool in tools]}")
    
    # Call a tool directly
    result = await mcp_caller.call_mcp_server_tool(
        server="filesystem",
        tool_name="read_file",
        args={"path": "/tmp/example.txt"}
    )
    print(result)

asyncio.run(main())
```

## Event Handling

Monitor tool execution with event callbacks:

```python
from cliver import AgentCore
from cliver.tool_events import ToolEvent, ToolEventType

def handle_tool_event(event: ToolEvent):
    if event.event_type == ToolEventType.TOOL_START:
        print(f"Starting tool: {event.tool_name}")
    elif event.event_type == ToolEventType.TOOL_END:
        print(f"Tool completed: {event.tool_name} ({event.duration_ms:.0f}ms)")
    elif event.event_type == ToolEventType.TOOL_ERROR:
        print(f"Tool error: {event.tool_name} - {event.error}")

agent = AgentCore(
    llm_models=llm_models,
    mcp_servers={},
    on_tool_event=handle_tool_event
)
```

## Permission Management

Control which tools can execute automatically:

```python
from cliver import AgentCore
from cliver.permissions import PermissionManager, PermissionAction

# Create a permission manager
perm_manager = PermissionManager(config_dir="/path/to/config")

# Allow Read tool globally
perm_manager.grant_global("Read", PermissionAction.ALLOW)

# Deny Write tool globally
perm_manager.grant_global("Write", PermissionAction.DENY)

agent = AgentCore(
    llm_models=llm_models,
    mcp_servers={},
    permission_manager=perm_manager
)
```

## Next Steps

Now that you understand how to extend CLIver, explore:

- [Skills](skills.md) for creating specialized capabilities
- [Memory & Identity](memory-identity.md) for agent personalization
- [Configuration](configuration.md) for model and provider setup
- [Roadmap](roadmap.md) for upcoming features