---
name: ci
description: Guides the agent through CI/CD workflows — read task, implement, test, commit, and optionally create a PR. Use when running in CI pipelines or automated batch jobs.
keywords: ci, cd, pipeline, automated, batch, github actions, jenkins, prow
allowed-tools: Read LS Grep Bash Write TodoWrite TodoRead
---

# CI/CD Workflow

You are running in a CI/CD pipeline. Follow this structured workflow to complete the task reliably and autonomously.

## Workflow Steps

### 1. Understand the Task
- Read the user's prompt carefully to understand the objective
- If a PR number or issue is mentioned, use available tools to fetch its description
- Identify the specific files, features, or bugs involved
- Summarize your understanding in 2-3 sentences before proceeding

### 2. Explore Relevant Code
- Read the project's README.md, CLAUDE.md, or similar documentation files
- Identify the relevant source files and their structure
- Understand existing patterns, conventions, and test structure
- Note the build system and test commands used by the project

### 3. Implement Changes
- Make focused, minimal changes that address the task
- Follow existing code patterns and conventions
- Do not refactor unrelated code
- Do not add features beyond what was asked

### 4. Run Tests and Iterate
- Run the project's test suite to verify your changes
- If tests fail, read the error output carefully
- Fix failures and re-run until all tests pass
- Run linting/formatting if the project has it configured

### 5. Commit
- Stage only the files you changed
- Write a conventional commit message:
  - `feat:` for new features
  - `fix:` for bug fixes
  - `refactor:` for code restructuring
  - `test:` for test-only changes
  - `docs:` for documentation changes
- Keep the commit message concise (under 72 characters for the subject line)

### 6. Create PR (Optional)
- Only if the user's prompt asks for a PR
- Requires a GitHub MCP server to be configured
- Use a descriptive PR title (under 70 characters)
- Include a summary of changes in the PR body
- Reference any related issues

## Guidelines
- Be autonomous — do not ask questions. Make reasonable decisions and proceed.
- If a step fails after 3 attempts, skip it and note the failure in the commit message or output.
- Prefer running the full test suite over individual test files to catch regressions.
- If no test command is documented, try common patterns: `make test`, `pytest`, `npm test`, `go test ./...`
