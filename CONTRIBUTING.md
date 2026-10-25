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
packages/cliver-core/        # Embeddable agent library (published to PyPI)
└── src/cliver_core/
    ├── agent.py             # AgentCore: chat(), stream(), run_skill()
    ├── bindings.py          # ModelBindings / ModelBinding
    ├── presets.py           # Live-verified provider bindings
    ├── provider/            # OpenAI + Anthropic protocol engines, media pipeline
    ├── tool.py              # CLIverTool, ToolRegistry, @tool
    ├── tools/               # Nine builtin tools (discovered by scanning)
    ├── skill.py             # SKILL.md parsing + SkillManager
    ├── skill_tools.py       # FindSkills / ActivateSkill / ReadSkillFile
    ├── mcp/                 # MCP client, discovery, adapters
    ├── policy.py            # Decision, ExecutionPolicy, scopes
    ├── capabilities/        # image / video / tts / asr / embed / rerank APIs
    ├── local/               # Optional in-process embed/rerank backends
    └── tests/               # Core test suite

packages/cliver/             # CLI application built on cliver-core
├── src/cliver/
│   ├── paths.py             # Config-directory resolution
│   └── skills/              # Eight-source discovery + four builtin skills
└── tests/                   # App test suite

website/                     # Documentation site (Astro 5 + Starlight)
Makefile                     # Build and development targets
pyproject.toml               # uv workspace + shared tooling config
```

### Key Modules

- **agent.py**: The Re-Act loop with tool calling, skill dispatch, MCP, and events
- **bindings.py**: Per-purpose model bindings (chat, image, video, TTS, ASR, embed, rerank)
- **presets.py**: Live-verified provider presets (DeepSeek, GLM, MiniMax, Seedance, …)
- **provider/**: OpenAI and Anthropic protocol engines plus the media pipeline
- **tool.py / tools/**: Tool abstraction, registry, and the nine builtin tools
- **skill.py / skill_tools.py**: Skill parsing, model-driven discovery, and activation
- **policy.py**: Opt-in permission gates and execution scopes
- **packages/cliver/src/cliver/skills/**: The CLI's eight-source skill discovery

## Testing

### Run All Tests
```bash
make test
```

### Run a Specific Test File
```bash
uv run pytest packages/cliver-core/src/cliver_core/tests/test_agent.py
```

### Run Tests Matching a Pattern
```bash
uv run pytest -k "test_permission"
```

### Writing Tests

- Core tests live in `packages/cliver-core/src/cliver_core/tests/`; app tests in
  `packages/cliver/tests/`
- Use descriptive test names starting with `test_`
- Use pytest fixtures for setup/teardown
- Mock external dependencies (LLM calls, filesystem operations)
- Aim for good coverage of critical paths
- Test both happy paths and error cases

## Documentation

CLIver uses **Astro 5 + Starlight** for the documentation website. Contributions to
documentation are welcome!

### Serve Documentation Locally
```bash
cd website
npm install
npm run dev
```

This serves the documentation at `http://localhost:4321`.

### Build Documentation
```bash
cd website
npm run build
```

This generates a static site in `website/dist/`.

### Documentation Files

- Documentation lives in `website/src/content/docs/`
- Use MDX format
- Follow the existing structure and navigation defined in `website/astro.config.mjs`
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
