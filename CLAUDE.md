# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CLIver is an AI agent that enhances your command-line interface with intelligent capabilities. It integrates with MCP (Model Context Protocol) servers and various LLM providers to provide an interactive CLI experience.

## Architecture

The project follows a modular architecture:

1. **Core CLI Module** (`src/cliver/cli.py`): Main entry point that handles user interaction using prompt_toolkit
2. **Configuration Management** (`src/cliver/config.py`): Handles configuration of MCP servers and LLM models using Pydantic models
3. **LLM Integration** (`src/cliver/llm/`): Manages different LLM inference engines and task execution
4. **MCP Server Integration** (`src/cliver/mcp_server_caller.py`): Interfaces with MCP servers using langchain_mcp_adapters
5. **Command System** (`src/cliver/commands/`): Extensible command system using Click framework
6. **Builtin Tools** (`src/cliver/tools/`): Core and contextual tools available to the LLM agent
7. **Permissions** (`src/cliver/permissions.py`): Layered permission system for tool execution
8. **CliverProfile** (`src/cliver/agent_profile.py`): Instance-scoped resource management (memory, identity, sessions)
9. **Skill Manager** (`src/cliver/skill_manager.py`): Discovers and activates SKILL.md files
10. **Utilities** (`src/cliver/util.py`): Helper functions for common operations

## Development Commands

### Setup and Installation
```bash
uv venv
uv sync --all-extras --dev --locked
```

### Running the Application for LLM inference
```bash
uv run cliver chat
```

### Testing
```bash
uv run pytest
```

### Code Quality
```bash
# Format code
uv run ruff format

# Check formatting
uv run ruff format --check

# Linting
uv run ruff check
```

## Code Structure

- **Commands**: New commands can be added by creating Python files in `src/cliver/commands/` with Click group definitions
- **Command Dispatcher** (`src/cliver/command_dispatcher.py`): Unified event loop replacing Click-based TUI routing, supports pending input injection
- **Chat Handlers** (`src/cliver/commands/chat_handler.py`, `src/cliver/commands/handlers.py`): Chat session management and handler dispatch
- **Configuration**: The app uses YAML-based configuration stored in the user's config directory (`~/.config/cliver/config.yaml`)
- **MCP Integration**: Supports stdio, SSE, streamable_http, and websocket transport mechanisms for MCP servers
- **LLM Support**: Currently supports Ollama, OpenAI, and vLLM providers with extensible engine system
- **Builtin Tools**: 17 tools (12 core + 5 contextual) in `src/cliver/tools/`
- **Skills**: SKILL.md files discovered from `.cliver/skills/` (project) and `{config_dir}/skills/` (global)
- **Permissions**: Layered system with persistent rules, session grants, and workflow-scoped overrides
- **Workflows**: LangGraph-powered multi-step workflows with `WorkflowCompiler`, `SubAgentFactory`, and SQLite checkpointing

## Key Components

1. **Cliver Class**: Main application class that manages the CLI session and task execution
2. **ConfigManager**: Handles all configuration operations for MCP servers and LLM models
3. **AgentCore**: Central component for processing user input through LLMs with tool calling (Re-Act pattern)
4. **MCPServersCaller**: Manages interactions with configured MCP servers
5. **PermissionManager**: Layered permission system controlling tool execution
6. **CliverProfile**: Manages instance-scoped resources (memory, identity, sessions, tasks)
7. **SkillManager**: Discovers and loads SKILL.md files for LLM-driven activation
8. **ConversationCompressor**: LLM-based conversation compression with truncation fallback
9. **WorkflowCompiler**: Compiles workflow YAML into LangGraph StateGraph with subagent isolation
10. **CommandDispatcher**: Unified event loop for TUI command routing with pending input injection

## Template Rendering

CLIver now includes Jinja2 template rendering capabilities with the following features:

1. **Environment Variables**: Access any environment variable via `{{ env.VARIABLE_NAME }}`
2. **Keyring Integration**: Access system secrets via `{{ keyring('service', 'key') }}`
3. **Custom Functions/Variables**: Ability to add custom functions and variables to the Jinja2 environment
4. **Performance Optimization**: Templates are only rendered if they contain `{{` and `}}` markers

Example usage in config.yaml:
```yaml
api_key: "{{ env.OPENAI_API_KEY or 'dummy' }}"
api_key: "{{ keyring('cliver', 'deepseek_key') }}"
```

Provider-level pricing configuration:
```yaml
providers:
  deepseek:
    pricing:
      currency: CNY
      input: 1.0
      output: 4.0
      cached_input: 0.25
```

## Extensibility

- Commands can be extended by adding new Python files to the commands directory
- New LLM providers can be added by implementing the LLMInferenceEngine interface
- MCP servers can be configured through the config command or YAML configuration
- Skills can be added as SKILL.md files in `.cliver/skills/` or `{config_dir}/skills/`

## Important notes on `process_user_input()` and `stream_user_input()` in `cliver.llm.AgentCore`

- This is the core of the whole Cliver AI agent.
- Command: `cliver chat "questions"` calls `cliver.llm.AgentCore.process_user_input()` method or `cliver.llm.AgentCore.stream_user_input()` for streaming mode
- The `process_user_input()` and `stream_user_input()` will do loops of inferences with a selected LLM until it gets final answer, before that, it will do tool calling.
- It follows `Re-Act` pattern to do inferences, and it also supports `Thinking` mode if LLM supports.
- The initial `process_user_input()` and `stream_user_input()` will try to organize the knowledge from some known places like: `README.md`, `CLAUDE.md` file, etc. Uses can also provide additional context via option: `-f context.md`
- Then it attaches mcp tools according to the users specification and context definitions in the skill files
- The request and response should be in `multipart` MIME type and the response can be used as request to another `chat`.

## Tool Permissions

CLIver has a layered permission system for tool execution. Each tool has an action kind (`safe`, `read`, `write`, `execute`, `fetch`) that determines its default behavior under the current permission mode.

### Permission Modes

| Mode        | Behavior                                                  |
|-------------|-----------------------------------------------------------|
| `default`   | Safe tools auto-allow; all others ask for confirmation    |
| `auto-edit` | Safe + read + write auto-allow; execute/fetch ask         |
| `yolo`      | All tools auto-allow (no confirmations)                   |

### Settings Files

- **Global**: `~/.config/cliver/cliver-settings.yaml`
- **Local (project)**: `.cliver/cliver-settings.yaml`

Local overrides global. Rules use regex for tool matching and fnmatch globs for resources.

### Tool Identity

- Builtin tools: `read_file`, `run_shell_command`, etc.
- MCP tools: `<server>#<tool>`, e.g. `github#create_issue`

### Key Files

- `src/cliver/permissions.py`: PermissionManager, rules, tool metadata registry
- `src/cliver/commands/permissions.py`: `/permissions` command (persistent rules)
- `src/cliver/commands/session_cmd.py`: `/session permission` (session grants)

### CLI Commands

- `/permissions rules` — show all persistent rules
- `/permissions add` — interactive rule builder
- `/permissions remove <index>` — remove a rule
- `/permissions mode [mode]` — show/set mode (persisted)
- `/session permission` — show session grants
- `/session permission mode <mode>` — session-only mode override
- `/session permission grant <tool>` — allow tool this session
- `/session permission deny <tool>` — deny tool this session
- `/session permission clear` — clear session grants

### Workflow/Task Permissions

Workflows, LLM steps, and tasks can declare `permissions` in their YAML:
```yaml
permissions:
  mode: auto-edit
  rules:
    - tool: "read_file"
      action: allow
    - tool: "run_shell_command"
      resource: "git *"
      action: allow
```

## Future features (TODO list)

- Introduce another command called `cliver deep-search "questions"` which will be a multiple steps workflow each of which is an actually `chat` command.
- In the new `cliver deep-search "questions"`, it accepts a yaml file to define each steps in line, or a markdown file with a mermaid diagram defined for the steps.
- Input watcher: file-system or event-driven triggers to start workflows automatically.

**Completed (no longer future)**:
- LangGraph-powered workflow engine with subagent isolation, SQLite checkpointing, and pause/resume support.

## IMPORTANT NOTES
Please use `uv run xxx` before any testing you are working on

### LLM modules to use
when helping me to implement some features or bug fixing, please make try to use the following LLM for the testing using `uv run chat -m <model> "what time is it in Beijing, Tokyo, London and New York now ?"`

- model: qwen
- model: deepseek-r1


### NOT DO

- Please do not update file of `~/.config/cliver/config.yaml` in your work.
- Please do not create test workflow to `~/.config/cliver/workflows` in your work, create in `.cliver/workflows` in current working directory
