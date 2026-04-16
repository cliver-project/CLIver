---
title: Permissions
description: Control which tools can execute and what resources they can access
---

# Tool Permissions

CLIver provides a permission system that controls tool execution during LLM interactions. You can configure which tools are allowed, denied, or require user confirmation — with rules scoped to specific resources like file paths, URLs, or shell commands.

## Permission Modes

Three modes control the baseline behavior:

| Mode        | Behavior                                                  |
|-------------|-----------------------------------------------------------|
| `default`   | Safe tools auto-allow; all others ask for confirmation    |
| `auto-edit` | Safe + read + write auto-allow; execute/fetch ask         |
| `yolo`      | All tools auto-allow (no confirmations)                   |

**Safe tools** (always auto-allowed): `skill`, `todo_read`, `todo_write`, `memory_read`, `memory_write`, `identity_update`, `ask_user_question`

## Settings File

Permissions are configured in `cliver-settings.yaml` at two levels:

- **Global**: `~/.config/cliver/cliver-settings.yaml` — user-wide defaults
- **Local**: `.cliver/cliver-settings.yaml` — project-specific overrides

Local settings take precedence over global.

### Example

```yaml
permission_mode: auto-edit

permissions:
  # Allow reading files
  - tool: "read_file"
    action: allow

  # Allow specific shell commands
  - tool: "run_shell_command"
    resource: "git *"
    action: allow
  - tool: "run_shell_command"
    resource: "uv *"
    action: allow

  # Allow all tools from a specific MCP server
  - tool: "github#.*"
    action: allow

  # Deny dangerous commands
  - tool: "run_shell_command"
    resource: "rm -rf *"
    action: deny
```

### Rule Format

Each rule has three fields:

| Field      | Format        | Description                                    |
|------------|---------------|------------------------------------------------|
| `tool`     | Regex         | Matched against tool identity via `re.fullmatch` |
| `resource` | fnmatch glob  | Optional — matched against the tool's resource (path, URL, or command) |
| `action`   | `allow`/`deny`| What to do when the rule matches               |

### Tool Identity

- **Builtin tools**: `read_file`, `write_file`, `run_shell_command`, etc.
- **MCP tools**: `<server>#<tool>`, e.g. `github#create_issue`, `ocp#get_pods`

### Common Patterns

```yaml
# All tools from an MCP server
- tool: "github#.*"
  action: allow

# Specific MCP tools
- tool: "ocp#(get_pods|get_nodes|describe_.*)"
  action: allow

# Grant all (project-level yolo)
- tool: ".*"
  action: allow
```

### Constrained Allow

When a tool has resource-specific rules, non-matching resources are implicitly denied. This prevents a broad wildcard from overriding tool-specific constraints:

```yaml
permissions:
  - tool: "read_file"
    resource: "/data/reports/**"
    action: allow
  - tool: ".*"
    action: allow
```

With this config, `read_file("/data/reports/q1.csv")` is allowed, but `read_file("/etc/passwd")` still requires confirmation — even though `".*"` would otherwise allow it.

## Managing Permissions via CLI

### Persistent Rules (`/permissions`)

```bash
# Show all rules
Cliver> /permissions rules

# Interactive rule builder
Cliver> /permissions add

# Remove a rule by index
Cliver> /permissions remove 2

# Show or set permission mode (saved to file)
Cliver> /permissions mode
Cliver> /permissions mode auto-edit
```

### Session Grants (`/session permission`)

Session grants are in-memory and cleared when the session ends:

```bash
# Show session grants
Cliver> /session permission

# Override mode for this session only
Cliver> /session permission mode yolo

# Allow/deny specific tools this session
Cliver> /session permission grant web_fetch
Cliver> /session permission deny docker_run

# Clear all session grants
Cliver> /session permission clear
```

### Interactive Prompt

When a tool requires permission, CLIver prompts:

```
  ⚠ Permission required: run_shell_command
    Resource: docker ps
    [y]es / [n]o / [a]lways allow / [d]eny always >
```

- `y` — allow this one time
- `n` — deny this one time
- `a` — allow this tool for the rest of the session
- `d` — deny this tool for the rest of the session

## Workflow Permissions

Workflows, LLM steps, and tasks can declare permissions in their YAML definition. This is important for automated or scheduled tasks where no user is present to approve tools:

```yaml
name: daily-report
permissions:
  mode: auto-edit
  rules:
    - tool: "read_file"
      resource: "/data/reports/**"
      action: allow
    - tool: "web_fetch"
      resource: "https://api.internal.com/**"
      action: allow
steps:
  - id: analyze
    name: Analyze data
    type: llm
    prompt: "Summarize the latest report"
    model: qwen
```

Workflow permissions are scoped — they are active only during execution and automatically cleaned up when the workflow completes.

## API-Level Usage

The permission system works independently of the CLI. When using CLIver as a Python library, you can configure `PermissionManager` and pass it to `AgentCore`:

```python
from cliver.permissions import PermissionManager, PermissionMode
from cliver.llm import AgentCore

# Create permission manager with custom settings
pm = PermissionManager(
    global_config_dir=Path("~/.config/cliver").expanduser(),
    local_dir=Path(".cliver"),
)
pm.set_mode(PermissionMode.AUTO_EDIT)

# Pass to AgentCore
executor = AgentCore(
    llm_models=models,
    mcp_servers=servers,
    permission_manager=pm,
    on_permission_prompt=my_custom_prompt_handler,
)
```

The `on_permission_prompt` callback lets you implement your own approval UI — a web form, a Slack message, or an auto-approver for testing.
