---
title: Gateway
description: CLIver Gateway - Long-running daemon for messaging platform integrations and background task scheduling
---

# Gateway

CLIver Gateway is a long-running daemon process that extends CLIver beyond one-shot CLI invocations to enable:

- **Messaging platform integrations** — Connect CLIver to Telegram, Discord, Slack, or Feishu (Lark) for bot-based conversations
- **Background task scheduling** — Run cron-scheduled or one-time tasks with result delivery
- **OpenAI-compatible REST API** — Expose CLIver's agent capabilities via HTTP endpoints
- **Admin web portal** — Monitor and manage tasks, sessions, and adapters

The gateway runs as a separate process from the CLI, allowing you to deploy CLIver as a persistent service that responds to external events, executes scheduled work, and serves API requests.

## Architecture

```mermaid
graph TB
    subgraph "Gateway Daemon"
        GW[Gateway]
        API[API Server<br/>Starlette/Uvicorn]
        SCHED[CronScheduler<br/>60s ticks]
        ADAPT[AdapterManager]
        STORE[TaskStore<br/>SQLite]
        SM[SessionManager]
        CORE[AgentCore]
    end
    
    subgraph "Platform Adapters"
        TG[Telegram]
        DC[Discord]
        SL[Slack]
        FS[Feishu]
    end
    
    subgraph "External Clients"
        USER[Users via IM]
        HTTP[HTTP Clients]
        ADMIN[Admin Portal]
    end
    
    USER -->|messages| TG
    USER -->|messages| DC
    USER -->|messages| SL
    USER -->|messages| FS
    
    TG --> ADAPT
    DC --> ADAPT
    SL --> ADAPT
    FS --> ADAPT
    
    ADAPT -->|MessageEvent| GW
    GW --> CORE
    CORE -->|response| GW
    GW -->|reply| ADAPT
    
    HTTP -->|/v1/chat/completions| API
    ADMIN -->|/admin/*| API
    API --> GW
    
    SCHED -->|tick| GW
    GW -->|execute| CORE
    GW <--> STORE
    GW <--> SM
    
    style GW fill:#4a90e2,stroke:#2e5c8a,stroke-width:3px,color:#fff
    style CORE fill:#e27d60,stroke:#a85a47,stroke-width:2px,color:#fff
```

## Quick Start

### 1. Configure platforms

Edit your `~/.cliver/config.yaml`:

```yaml
gateway:
  host: "127.0.0.1"
  port: 8321
  api_key: "your-api-key-here"  # Optional, for OpenAI API auth
  admin_username: "admin"         # Required for admin portal
  admin_password: "secret"        # Required for admin portal
  
  platforms:
    telegram:
      type: telegram
      token: "{{env.TELEGRAM_BOT_TOKEN}}"
      allowed_users: ["123456789"]  # Optional whitelist
    
    discord:
      type: discord
      token: "{{env.DISCORD_BOT_TOKEN}}"
      allowed_users: []  # Empty = open access
    
    slack:
      type: slack
      token: "{{env.SLACK_BOT_TOKEN}}"
      app_token: "{{env.SLACK_APP_TOKEN}}"  # Socket Mode
    
    feishu:
      type: feishu
      token: "{{env.FEISHU_APP_SECRET}}"
      app_id: "{{env.FEISHU_APP_ID}}"
      verification_token: "{{env.FEISHU_VERIFICATION_TOKEN}}"
```

### 2. Launch the gateway

```bash
cliver gateway start
```

The gateway will:

- Acquire a PID-based file lock (`~/.cliver/cliver-gateway.pid`) — only one instance per agent
- Initialize AgentCore with your configured LLM models
- Start the API server on `http://127.0.0.1:8321`
- Connect all configured platform adapters
- Begin the cron scheduler (60-second ticks)

### 3. Verify status

```bash
# Check health endpoint
curl http://127.0.0.1:8321/health

# Or visit the admin portal
open http://127.0.0.1:8321/admin
```

## Platform Adapters

Each adapter connects CLIver to a messaging platform. Messages are bridged bidirectionally: users send text/media to the bot, CLIver processes the input, and the response is delivered back to the same conversation thread.

### Common Features

All adapters support:

- **Text messages** with markdown formatting (converted to platform-specific format)
- **Media attachments** — images, files, voice messages
- **Threading** — replies are kept in-thread where supported
- **Session persistence** — each IM thread gets its own conversation session
- **User allowlists** — restrict access via `allowed_users` (empty list = open access)
- **Typing indicators** — shown while AgentCore is processing

### Telegram

**Dependencies**: `pip install cliver[telegram]` (installs `python-telegram-bot`)

**Configuration**:

```yaml
telegram:
  type: telegram
  token: "{{env.TELEGRAM_BOT_TOKEN}}"
  allowed_users: ["123456789"]  # Telegram user IDs
```

**Bot Setup**:

1. Create a bot via [@BotFather](https://t.me/botfather)
2. Copy the bot token to your environment: `export TELEGRAM_BOT_TOKEN=123:ABC...`
3. Obtain your user ID: send `/start` to [@userinfobot](https://t.me/userinfobot)

**Formatting**: Markdown is converted to Telegram MarkdownV2 (`**bold**` → `*bold*`)

**Message limit**: 4096 characters (auto-chunked if exceeded)

### Discord

**Dependencies**: `pip install cliver[discord]` (installs `discord.py`)

**Configuration**:

```yaml
discord:
  type: discord
  token: "{{env.DISCORD_BOT_TOKEN}}"
  allowed_users: ["123456789012345678"]  # Discord user IDs
```

**Bot Setup**:

1. Create an application at [Discord Developer Portal](https://discord.com/developers/applications)
2. Add a bot user and enable "Message Content Intent"
3. Copy the bot token to your environment: `export DISCORD_BOT_TOKEN=...`
4. Invite the bot to your server with `bot` and `applications.commands` scopes

**Formatting**: Standard markdown (Discord supports it natively)

**Message limit**: 2000 characters (auto-chunked if exceeded)

### Slack

**Dependencies**: `pip install cliver[slack]` (installs `slack-bolt`, `slack-sdk`)

**Configuration**:

```yaml
slack:
  type: slack
  token: "{{env.SLACK_BOT_TOKEN}}"       # Bot User OAuth Token (xoxb-...)
  app_token: "{{env.SLACK_APP_TOKEN}}"   # App-Level Token (xapp-...)
  allowed_users: ["U1234567890"]         # Slack user IDs (optional)
```

**Bot Setup**:

1. Create a Slack app at [api.slack.com/apps](https://api.slack.com/apps)
2. Enable **Socket Mode** and generate an app-level token with `connections:write`
3. Add bot token scopes: `chat:write`, `files:write`, `channels:history`, `groups:history`, `im:history`, `mpim:history`
4. Install the app to your workspace
5. Copy tokens to environment:
   - `SLACK_BOT_TOKEN` — Bot User OAuth Token (Settings → OAuth & Permissions)
   - `SLACK_APP_TOKEN` — App-Level Token (Settings → Basic Information → App-Level Tokens)

**Formatting**: Markdown is converted to Slack mrkdwn (`**bold**` → `*bold*`)

**Message limit**: 4000 characters (auto-chunked if exceeded)

### Feishu (Lark)

**Dependencies**: `pip install cliver[feishu]` (installs `aiohttp`)

**Configuration**:

```yaml
feishu:
  type: feishu
  token: "{{env.FEISHU_APP_SECRET}}"
  app_id: "{{env.FEISHU_APP_ID}}"
  verification_token: "{{env.FEISHU_VERIFICATION_TOKEN}}"
  allowed_users: ["ou_123456789abcdef"]  # Open IDs (optional)
```

**Bot Setup**:

1. Create an app at [Feishu Open Platform](https://open.feishu.cn/)
2. Enable event subscriptions and configure the webhook URL (e.g., `https://yourhost/webhook/feishu`)
3. Subscribe to `im.message.receive_v1` event
4. Copy credentials to environment:
   - `FEISHU_APP_ID` — App ID
   - `FEISHU_APP_SECRET` — App Secret
   - `FEISHU_VERIFICATION_TOKEN` — Verification Token (Events & Callbacks)

**Formatting**: Standard markdown (Feishu supports it natively)

**Message limit**: 4000 characters (auto-chunked if exceeded)

## Conversation Sessions

The gateway maintains persistent conversation history for each IM thread. Sessions are stored in `~/.cliver/<agent>/gateway-sessions/` (separate from CLI sessions).

**Session key format**: `platform:channel_id:thread_id`

- Each thread gets its own session — top-level messages and threaded replies are separate
- History is loaded on every message and appended after the response
- Automatic compression kicks in when history exceeds the model's context window
- Sessions are trimmed to `session.max_turns_per_session` (default: 100) and cleaned up after `session.max_age_days` (default: 30)

**Session linking with tasks**:

When you create a task from an IM conversation, the task is linked to the session. Subsequent messages in that thread continue the same context, and scheduled task results are delivered back to the originating thread.

## Background Tasks

Tasks are YAML definitions stored in `~/.cliver/<agent>/tasks/` and tracked in `gateway.db`. The gateway's cron scheduler evaluates all tasks every 60 seconds and dispatches any that are due.

### Task Definition

Create a task via the CLI:

```bash
cliver task create daily_report \
  --prompt "Summarize today's GitHub activity for cliver-project" \
  --schedule "0 17 * * *"
```

Or manually in `~/.cliver/<agent>/tasks/daily_report.yaml`:

```yaml
name: daily_report
prompt: "Summarize today's GitHub activity for cliver-project"
schedule: "0 17 * * *"      # Cron: daily at 5pm
model: claude-sonnet-4.5     # Optional, uses default if omitted
skills: []                   # Optional: pre-activate skills
origin:                      # Optional: IM origin for result delivery
  platform: telegram
  channel_id: "123456789"
  thread_id: "42"
session_id: "abc-123"        # Optional: link to a session for context
```

### Scheduling Syntax

**Cron expression** (recurring):

```bash
# Minute Hour Day Month Weekday
  0      17   *   *     *        # Daily at 5pm
  */30   *    *   *     *        # Every 30 minutes
  0      9    *   *     1-5      # Weekdays at 9am
```

**One-time execution** (ISO datetime):

```yaml
run_at: "2025-05-10T14:30:00"  # Runs once at this time
```

### Task Lifecycle

1. **Pending** — Task is registered and waiting for next schedule tick
2. **Running** — Task is executing (AgentCore processes the prompt)
3. **Completed** — Task finished successfully, result saved or delivered
4. **Failed** — Task encountered an error (logged in run history)
5. **Suspended** — Task is paused because its platform adapter is disconnected

**Automatic suspension/resumption**:

- Tasks with an IM origin are suspended if the adapter disconnects
- When the adapter reconnects, suspended tasks are automatically resumed

### Result Delivery

**IM-origin tasks**:

- Results are delivered back to the originating IM thread (via `origin.platform`, `channel_id`, `thread_id`)
- Synthetic turns are appended to the linked session (user: `[Task 'name' executed]`, assistant: `<result>`)

**Non-IM tasks**:

- Results are saved as JSON files in `~/.cliver/<agent>/tasks/<task_name>/<task_name>_execution_<id>.json`

### Run History

All executions are recorded in `gateway.db` → `task_runs` table. View via the admin portal (`/admin/tasks/<name>`) or CLI:

```bash
cliver task list
cliver task info daily_report
```

## REST API

The gateway exposes an OpenAI-compatible API for chat completions, plus custom health and status endpoints.

### Authentication

Set an API key in config:

```yaml
gateway:
  api_key: "sk-cliver-1234567890"
```

Clients pass it in the `Authorization` header:

```bash
Authorization: Bearer sk-cliver-1234567890
```

### Endpoints

#### `GET /health`

Returns gateway uptime and adapter statuses.

**Response**:

```json
{
  "status": "ok",
  "uptime": 3600,
  "tasks_run": 42,
  "platforms": ["telegram", "slack"],
  "adapters": [
    {"name": "telegram", "state": "connected", "error": ""},
    {"name": "slack", "state": "connecting", "error": ""}
  ]
}
```

#### `GET /v1/models`

Lists available LLM models.

**Response**:

```json
{
  "object": "list",
  "data": [
    {"id": "claude-sonnet-4.5", "object": "model", "created": 0, "owned_by": "cliver"},
    {"id": "gpt-4o", "object": "model", "created": 0, "owned_by": "cliver"}
  ]
}
```

#### `POST /v1/chat/completions`

OpenAI-compatible chat completion endpoint. Supports streaming and non-streaming modes.

**Request**:

```json
{
  "model": "claude-sonnet-4.5",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"}
  ],
  "stream": false,
  "temperature": 0.7
}
```

**Response (non-streaming)**:

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1704067200,
  "model": "claude-sonnet-4.5",
  "choices": [
    {
      "index": 0,
      "message": {"role": "assistant", "content": "The capital of France is Paris."},
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 20,
    "completion_tokens": 10,
    "total_tokens": 30
  }
}
```

**Streaming mode** (`stream: true`):

Responses are sent as Server-Sent Events (SSE) with `data:` prefixed JSON chunks:

```
data: {"id": "chatcmpl-abc123", "object": "chat.completion.chunk", "created": 1704067200, "model": "claude-sonnet-4.5", "choices": [{"index": 0, "delta": {"content": "The"}, "finish_reason": null}]}

data: {"id": "chatcmpl-abc123", "object": "chat.completion.chunk", "created": 1704067200, "model": "claude-sonnet-4.5", "choices": [{"index": 0, "delta": {"content": " capital"}, "finish_reason": null}]}

data: [DONE]
```

**Server mode**:

The API automatically configures AgentCore for headless operation:

- Permissions set to YOLO mode (all tools auto-allowed)
- Ask tool is disabled (no interactive prompts)
- System message appended: "You are running as a backend API service. Make autonomous decisions."

## Admin Portal

The admin portal is a web-based UI for monitoring and managing the gateway.

**URL**: `http://127.0.0.1:8321/admin`

**Authentication**: Cookie-based session auth (username/password from config)

### Features

**Gateway Dashboard**:

- Live status (uptime, tasks run, connected adapters)
- Adapter connection states and error messages

**Tasks**:

- List all tasks (DB-first: shows registered tasks + YAML load status)
- View task details (definition, schedule, run history, origin, live state)
- Manually trigger a task execution
- Delete tasks (removes YAML file, DB entry, and run history)

**Sessions**:

- Browse gateway sessions (IM conversations) and CLI sessions separately
- View conversation turns for any session
- Delete stale sessions

**Skills**:

- List all discovered skills with full metadata
- View skill source, allowed tools, and body

**Adapters**:

- View configured platform adapters
- See connection status and auth errors
- Secrets are masked (e.g., `1234****5678`)

**Agent**:

- View agent name, identity, and memory
- Show LLM model configuration and MCP servers

**Chat UI**:

- Interactive chat interface for testing prompts
- Model selection, tool filtering, and system message injection
### Admin API

The admin portal is built on REST endpoints under `/admin/api/`:

- `GET /admin/api/status` — Gateway status
- `GET /admin/api/tasks` — List tasks
- `GET /admin/api/tasks/{name}` — Task detail
- `POST /admin/api/tasks/{name}/run` — Trigger task execution
- `DELETE /admin/api/tasks/{name}` — Delete task
- `GET /admin/api/sessions/{source}` — List sessions (`source`: `cli` or `gateway`)
- `GET /admin/api/sessions/{source}/{id}` — Session turns
- `DELETE /admin/api/sessions/{source}/{id}` — Delete session
- `GET /admin/api/skills` — List skills
- `GET /admin/api/adapters` — List adapters
- `GET /admin/api/agent` — Agent info
- `GET /admin/api/config` — Config overview (models, providers, MCP servers)
- `GET /admin/api/models` — Available LLM models
- `POST /admin/api/chat` — Streaming chat endpoint

## Configuration Reference

### Gateway Config

```yaml
gateway:
  host: "127.0.0.1"                      # API server bind address
  port: 8321                             # API server port
  api_key: "sk-..."                      # Optional: API key for /v1/* endpoints
  admin_username: "admin"                # Required for admin portal
  admin_password: "secret"               # Required for admin portal
  platforms: {}                          # Platform adapter configs (see below)
```

### Platform Config

All platforms share these base fields:

```yaml
platforms:
  <name>:
    type: telegram | discord | slack | feishu  # Adapter type
    token: "..."                                # Bot token or app secret
    allowed_users: []                           # User ID whitelist (empty = open)
    home_channel: ""                            # Optional default channel
```

**Telegram-specific**:

```yaml
telegram:
  type: telegram
  token: "123:ABC..."
```

**Discord-specific**:

```yaml
discord:
  type: discord
  token: "..."
```

**Slack-specific**:

```yaml
slack:
  type: slack
  token: "xoxb-..."     # Bot User OAuth Token
  app_token: "xapp-..."  # App-Level Token (Socket Mode)
```

**Feishu-specific**:

```yaml
feishu:
  type: feishu
  token: "..."                  # App Secret
  app_id: "..."                 # App ID
  verification_token: "..."     # Event subscription verification token
```

### Custom Adapters

You can load custom adapter classes by specifying a fully-qualified module path:

```yaml
platforms:
  my_platform:
    type: "mypackage.adapters.MyCustomAdapter"
    token: "..."
```

The class must inherit from `cliver.gateway.platform_adapter.PlatformAdapter` and implement all abstract methods.

## CLI Commands

```bash
# Start the gateway
cliver gateway start

# Stop the gateway
cliver gateway stop

# Restart the gateway
cliver gateway restart

# Check gateway status
cliver gateway status

# Create a task
cliver task create <name> --prompt "..." --schedule "0 9 * * *"

# Link a task to the current IM thread (when invoked from IM)
cliver task create <name> --prompt "..." --schedule "..." --reply-to telegram:123456789:42

# List tasks
cliver task list

# View task details
cliver task info <name>

# Delete a task
cliver task delete <name>
```

## Process Management

The gateway is designed to run as a long-lived daemon. Only one instance can run per agent (enforced via PID file locking).

**Start as a background process**:

```bash
# Foreground (logs to console)
cliver gateway start

# Background (recommended for production)
nohup cliver gateway start > ~/.cliver/gateway.log 2>&1 &

# Or use a process supervisor (systemd, supervisord, etc.)
```

**Systemd unit example**:

```ini
[Unit]
Description=CLIver Gateway
After=network.target

[Service]
Type=simple
User=youruser
WorkingDirectory=/home/youruser
ExecStart=/home/youruser/.local/bin/cliver gateway start
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

**Lifecycle hooks**:

- `_on_startup()` — Acquire flock, initialize AgentCore, start scheduler and adapters
- `_on_cleanup()` — Stop adapters, close DB, release flock

**Graceful shutdown**:

The gateway intercepts `SIGTERM` and `SIGINT` to shut down cleanly. On shutdown:

1. Cron task is cancelled
2. Adapters are stopped (pending messages may be lost)
3. Task store DB is closed
4. PID file is released

## Database Schema

The gateway uses SQLite (`~/.cliver/<agent>/gateway.db`) to persist task registry, run history, and live state.

### `tasks` table

Stores the task registry with IM origin and session linkage.

```sql
CREATE TABLE tasks (
    name TEXT PRIMARY KEY,
    yaml_path TEXT NOT NULL,
    session_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    origin_source TEXT,
    origin_platform TEXT,
    origin_channel_id TEXT,
    origin_thread_id TEXT,
    origin_user_id TEXT,
    state_status TEXT,
    state_suspend_reason TEXT
);
```

### `task_runs` table

Stores execution history for all tasks.

```sql
CREATE TABLE task_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_name TEXT NOT NULL,
    execution_id TEXT NOT NULL,
    status TEXT NOT NULL,           -- running, completed, failed
    started_at TEXT NOT NULL,
    finished_at TEXT,
    error TEXT,
    result TEXT
);

CREATE INDEX idx_task_runs_task_name ON task_runs(task_name);
CREATE INDEX idx_task_runs_started_at ON task_runs(task_name, started_at DESC);
```

## Deployment Tips

### Production Checklist

- [ ] Set strong `admin_password` and `api_key`
- [ ] Use `allowed_users` to restrict IM access
- [ ] Run behind a reverse proxy (nginx, Caddy) with HTTPS for public endpoints
- [ ] Configure firewall rules (only allow trusted IPs to reach port 8321)
- [ ] Set up log rotation (`gateway.log`, `tasks/*.log`)
- [ ] Monitor disk usage (`gateway.db`, `gateway-sessions/`, `tasks/`)
- [ ] Use a process supervisor (systemd, supervisord) for auto-restart
- [ ] Configure session cleanup (`session.max_age_days`, `session.max_sessions`)
- [ ] Test adapter reconnection (disconnect network, verify suspension/resumption)

### Scaling

The gateway is designed for single-instance deployment (one AgentCore, one set of adapters, one cron scheduler). For multi-instance deployments:

- **Horizontal scaling**: Not supported — multiple gateways would compete for the same PID lock and task store
- **Vertical scaling**: Increase `session.max_turns_per_session` and model context window to handle longer conversations
- **Task isolation**: Run multiple agents (different `--agent` names) with separate gateways for workload isolation

### Logging

The gateway logs to `stdout` (uvicorn access logs) and `stderr` (application logs). Tool events are logged to `cliver.gateway.tools`.

**Log levels**:

- `INFO` — Normal operation (adapter connection, task execution, message handling)
- `WARNING` — Recoverable errors (invalid cron, adapter reconnect, auth failure)
- `ERROR` — Unrecoverable errors (AgentCore crash, adapter init failure)

**Configure logging**:

```yaml
gateway:
  log_level: INFO  # DEBUG, INFO, WARNING, ERROR
```

## Troubleshooting

### Gateway won't start

**Symptom**: `RuntimeError: Another gateway is already running`

**Cause**: PID file lock is held by another process

**Fix**:

```bash
# Check if another gateway is running
ps aux | grep "cliver gateway"

# If no process found, remove stale lock file
rm ~/.cliver/cliver-gateway.pid
```

### Adapter fails to connect

**Symptom**: Adapter state shows "error" in `/health` or admin portal

**Common causes**:

- **Invalid token**: Check that `token`, `app_token`, `app_id` match your bot credentials
- **Missing scopes**: Verify bot permissions (Slack: `chat:write`, `files:write`, Discord: Message Content Intent)
- **Network issue**: Test API connectivity (`curl https://api.telegram.org/bot<token>/getMe`)
- **Missing dependency**: Install adapter extras (`pip install cliver[telegram]`)

**Debug**:

- Check gateway logs for detailed error messages
- Run with `--log-level DEBUG` for verbose output

### Task not executing

**Symptom**: Task appears in `cliver task list` but never runs

**Common causes**:

- **Invalid cron**: Syntax error in `schedule` field (check logs for warnings)
- **Suspended**: Adapter disconnected (check `state_status` in admin portal)
- **Already ran**: One-shot `run_at` tasks clear after execution (check run history)

**Debug**:

```bash
# View task state
cliver task info <name>

# Check last run time
sqlite3 ~/.cliver/<agent>/gateway.db "SELECT * FROM task_runs WHERE task_name='<name>' ORDER BY id DESC LIMIT 1;"
```

### IM messages not reaching the bot

**Symptom**: User sends a message, bot doesn't respond

**Common causes**:

- **User not in allowlist**: Check `allowed_users` in platform config
- **Adapter disconnected**: Check `/health` endpoint or admin portal
- **Bot not invited**: Verify bot is a member of the channel/group (Discord, Slack)
- **Message content intent disabled**: Enable in Discord Developer Portal (Bot → Privileged Gateway Intents)

**Debug**:

- Check gateway logs for `Slack message ignored: user <id> not in allowed list`
- Test with a DM (bypasses group membership issues)

## Advanced Topics

### Voice Message Transcription

Voice messages are automatically transcribed to text via OpenAI Whisper (if an audio-capable provider is configured). The transcript is prepended to the user's message text.

**Disable transcription**:

Modify `_handle_message_inner()` in `gateway.py` to skip the `transcribe_voice_message` call.

### Session Compression

When conversation history exceeds the model's context window, the gateway automatically compresses it using `ConversationCompressor`. The compression is logged (`INFO` level).

**Tune compression**:

```python
# In gateway.py, _compress_history()
compressor = ConversationCompressor(context_window=32768)  # Adjust window
```

### Tool Filtering in IM

The `Ask` tool is automatically disabled in IM conversations (no interactive UI to respond to prompts). Other tools are available unless explicitly filtered via `filter_tools`.

**Add custom filters**:

```python
# In gateway.py, _im_filter_tools()
async def _im_filter_tools(user_input, tools):
    # Block specific tools by name
    return [t for t in tools if t.name not in ("Ask", "MyDangerousTool")]
```

### Custom System Messages

IM conversations receive an auto-injected system message with task creation rules and IM context. Modify `_im_system_appender()` in `gateway.py` to customize.

## See Also

- [Tasks](tasks.md) — Task management CLI reference
- [Session Management](session-management.md) — Conversation history and session options
- [Permissions](permissions.md) — Tool permission modes
