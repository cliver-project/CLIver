# Contributing to CLIver

Thank you for your interest in contributing to CLIver! This guide will help you get started with development and understand how to contribute effectively.

## Development Setup

### Prerequisites
- Python 3.10 or higher
- `uv` (recommended for Python package management)
- Node.js 22 or higher + npm (for the admin portal)
- Git

### Initial Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/cliver-project/CLIver.git
   cd CLIver
   ```

2. **Initialize development environment**
   ```bash
   make init
   ```
   This command sets up a Python virtual environment and installs all dependencies including development tools.

3. **Install admin portal dependencies**
   ```bash
   make admin-install
   ```

4. **Run tests**
   ```bash
   make test
   ```
   Verify that the setup is correct by running the test suite.

## Development Workflow

### Before You Start

1. **Create a fork** of the repository on GitHub
2. **Clone your fork** and add an upstream remote:
   ```bash
   git clone https://github.com/YOUR_USERNAME/CLIver.git
   cd CLIver
   git remote add upstream https://github.com/cliver-project/CLIver.git
   ```

### Making Changes

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or for bug fixes:
   git checkout -b fix/bug-description
   ```

2. **Make your changes** following the code style guidelines (see below)

3. **Test your changes**
   ```bash
   make test
   ```

4. **Format and lint your code**
   ```bash
   make format
   make lint
   ```

5. **Commit your changes**
   - Use clear, descriptive commit messages
   - Reference issues when relevant (e.g., "Fixes #123")
   - Keep commits focused and atomic

6. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Create a Pull Request**
   - Provide a clear title and description
   - Link to related issues
   - Ensure all tests pass in CI
   - Be responsive to code review feedback

## Admin Portal Development

The admin portal is a React SPA in the `admin/` directory. It communicates with the Python gateway API.

### Development Setup (Two Terminals)

**Terminal 1 — Python gateway (API backend):**
```bash
uv run cliver gateway start
# Runs on http://localhost:8321
```

**Terminal 2 — Vite dev server (frontend with hot-reload):**
```bash
make admin-dev
# Runs on http://localhost:5173
```

Open `http://localhost:5173/admin/` in your browser. The Vite dev server proxies all `/admin/api/*` requests to the Python gateway at `:8321`, so authentication, workflow execution, and all API calls work seamlessly. Edits to React components hot-reload instantly.

### Admin Portal Build

To build the production SPA:
```bash
make admin-build
```

This compiles the React app and copies the output to `src/cliver/gateway/admin_dist/`, which is bundled into the Python package. When users install CLIver and run `cliver gateway start`, the admin portal is served from this bundled directory — no Node.js required.

### Admin Portal Lint

```bash
make admin-lint
```

### Key Technologies

- **React 19** + **TypeScript** — UI framework
- **Vite 6** — Build tool and dev server
- **Tailwind CSS 4** + **Shadcn/ui** — Styling and components
- **ReactFlow** (`@xyflow/react`) — Workflow DAG editor with drag-drop
- **Tanstack Query** — API data fetching and caching
- **React Router 7** — Client-side routing

### Admin Portal Structure

```
admin/
├── src/
│   ├── main.tsx              # React root + router + QueryClient
│   ├── App.tsx               # Layout shell (icon sidebar + Outlet)
│   ├── lib/
│   │   ├── api.ts            # Fetch wrapper for /admin/api/*
│   │   └── utils.ts          # cn() helper
│   ├── hooks/
│   │   └── use-api.ts        # Tanstack Query hooks for all endpoints
│   ├── components/
│   │   ├── ui/               # Shadcn/ui components
│   │   ├── sidebar.tsx       # Icon sidebar navigation
│   │   └── theme-toggle.tsx  # Dark/light mode toggle
│   └── pages/
│       ├── dashboard.tsx     # Gateway status
│       ├── login.tsx         # Login form
│       ├── workflows/
│       │   ├── list.tsx      # Workflow grid + executions
│       │   ├── detail.tsx    # ReactFlow editor
│       │   └── components/   # Canvas, nodes, toolbar, panel
│       ├── tasks/
│       │   ├── list.tsx      # Task table
│       │   └── detail.tsx    # Task detail
│       └── ...               # sessions, skills, agent, config
├── package.json
├── vite.config.ts
└── tailwind.config.ts
```

## Code Style

CLIver uses **Ruff** for code formatting and linting. All code must conform to these standards.

### Format Your Code

Before committing, format your code:
```bash
make format
```

This runs:
- `ruff check --fix` — Auto-fix linting issues
- `ruff format` — Format code according to project standards

### Lint Check

Verify your code passes linting without auto-fixes:
```bash
make lint
```

This runs:
- `ruff check` — Check for linting issues
- `ruff format --check` — Check formatting compliance

### Code Style Guidelines

- Follow PEP 8 conventions
- Use meaningful variable and function names
- Write docstrings for classes and public functions
- Keep functions focused and modular
- Add type hints where appropriate
- No trailing whitespace

## Project Structure

Understanding the codebase structure will help you locate relevant files and understand the architecture.

```
admin/                       # React SPA (admin portal)
├── src/                     # TypeScript source
├── package.json             # Node.js dependencies
└── vite.config.ts           # Build configuration

src/cliver/
├── cli.py                   # Main CLI entry point
├── config.py                # Configuration management
├── permissions.py           # Permission system for tool execution
├── skill_manager.py         # Skill loading and activation
├── agent_profile.py         # Agent profiles and session history
│
├── llm/                     # LLM provider integrations
│   └── llm.py               # AgentCore (Re-Act loop)
│
├── commands/                # CLI subcommand implementations
│
├── gateway/                 # Gateway daemon and admin portal
│   ├── gateway.py           # Gateway server (Starlette + uvicorn)
│   ├── admin.py             # Admin API routes + SPA serving
│   └── admin_dist/          # Built SPA assets (generated by make admin-build)
│
├── workflow/                # Workflow engine (LangGraph)
│   ├── workflow_models.py   # Pydantic models (LLMStep, PythonStep, Workflow)
│   ├── compiler.py          # YAML → LangGraph StateGraph
│   ├── executor.py          # Thin wrapper around graph.ainvoke()
│   ├── node_runners.py      # Pure node functions (llm, python)
│   ├── ref_resolver.py      # ${ref} substitution + auto-inject
│   ├── condition_eval.py    # Safe dot-path expression evaluator
│   └── persistence.py       # YAML CRUD + SQLite execution tracking
│
├── tools/                   # Built-in tool implementations
│
└── skills/                  # Built-in SKILL.md files

tests/                       # Test suite
docs/                        # Documentation (mkdocs-material)
Makefile                     # Build and development targets
pyproject.toml               # Project metadata and dependencies
```

### Key Modules

- **agent_profile.py**: Stores user agent profiles, preferences, and session history
- **builtin_tools.py**: Defines built-in tools available to the agent
- **cli.py**: Main entry point with Click command decorators
- **config.py**: Configuration loading, validation, and provider setup
- **permissions.py**: Permission checking before tool execution
- **skill_manager.py**: Dynamic skill loading and registration
- **session_manager.py**: Manages conversation sessions and context
- **gateway/**: Platform adapters for Telegram, Discord, Slack, etc.
- **llm/**: Provider-specific LLM implementations

## Testing

### Run All Tests
```bash
make test
```

### Run Tests with Coverage
```bash
uv run pytest --cov
```

### Run a Specific Test File
```bash
uv run pytest tests/test_cli.py
```

### Run Tests Matching a Pattern
```bash
uv run pytest tests/ -k "test_permission"
```

### Writing Tests

- Place tests in the `tests/` directory
- Use descriptive test names starting with `test_`
- Use pytest fixtures for setup/teardown
- Mock external dependencies (LLM calls, filesystem operations)
- Aim for good coverage of critical paths
- Test both happy paths and error cases

### Test Coverage

Generate a coverage report:
```bash
uv run pytest --cov=src/cliver --cov-report=html
```

This creates an HTML report in `htmlcov/index.html`.

## Documentation

CLIver uses **mkdocs-material** for documentation. Contributions to documentation are welcome!

### Serve Documentation Locally
```bash
make docs-serve
```

This builds and serves the documentation at `http://localhost:8000`.

### Build Documentation
```bash
make docs-build
```

This generates a static site in the `site/` directory.

### Documentation Files

- Documentation files are in the `docs/` directory
- Use Markdown format (.md)
- Follow the existing structure and navigation
- Include examples where appropriate

## Reporting Issues

Found a bug or have a feature request? Please report it on GitHub.

### Bug Reports
- Use the [Bug Report template](.github/ISSUE_TEMPLATE/bug_report.md)
- Include steps to reproduce
- Provide error messages and logs
- Include your environment (OS, Python version, CLIver version)

### Feature Requests
- Use the [Feature Request template](.github/ISSUE_TEMPLATE/feature_request.md)
- Describe the problem you're solving
- Explain the expected behavior
- Provide use case examples

## License

All contributions to CLIver are licensed under the **Apache License 2.0**. By submitting a pull request, you agree that your code will be licensed under this license.

See the [LICENSE](LICENSE) file for the full license text.

## Code of Conduct

We're committed to providing a welcoming and inclusive environment. Be respectful, professional, and constructive in all interactions.

## Getting Help

- **Documentation**: https://cliver-project.github.io/CLIver/
- **GitHub Issues**: https://github.com/cliver-project/CLIver/issues
- **Discussions**: https://github.com/cliver-project/CLIver/discussions

---

Thank you for contributing to CLIver! Your effort helps make CLIver better for everyone.
