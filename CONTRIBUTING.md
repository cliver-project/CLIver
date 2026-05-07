# Contributing to CLIver

Thank you for your interest in contributing to CLIver! This guide will help you get started with development and understand how to contribute effectively.

## Development Setup

### Prerequisites
- Python 3.10 or higher
- `uv` (recommended for Python package management)
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

3. **Run tests**
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
src/cliver/
├── agent_profile.py         # User agent profiles and configurations
├── builtin_tools.py         # Built-in tool definitions
├── cli.py                   # Main CLI entry point (Click commands)
├── config.py                # Configuration management and defaults
├── permissions.py           # Permission system for tool execution
├── skill_manager.py         # Skill loading and activation
├── session_manager.py       # Conversation session management
├── tool_registry.py         # Tool registry and discovery
├── constants.py             # Project-wide constants
├── cost_tracker.py          # LLM cost tracking utilities
├── db.py                    # Database management
│
├── commands/                # CLI subcommand implementations
│
├── gateway/                 # Gateway daemon and platform adapters
│   ├── telegram_adapter.py
│   ├── discord_adapter.py
│   └── ...
│
├── llm/                     # LLM provider integrations
│   ├── openai_provider.py
│   ├── ollama_provider.py
│   ├── deepseek_provider.py
│   └── ...
│
├── tools/                   # Individual tool implementations
│   ├── bash_tool.py
│   ├── python_tool.py
│   └── ...
│
└── cli_*.py                 # CLI support modules for specific features
    ├── cli_llm_call.py
    ├── cli_tool_progress.py
    └── ...

tests/                       # Test suite
├── test_*.py                # Unit tests
└── ...

docs/                        # Documentation (mkdocs-material)
├── index.md
├── permissions.md
└── ...

mkdocs.yml                   # Documentation configuration
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
