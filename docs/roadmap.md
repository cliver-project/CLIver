---
title: Roadmap
description: Future development plans
---

# Roadmap

This document outlines the development status and future plans for CLIver.

## Completed Features

- **Builtin Tools** (17 total): File I/O, shell, web search/fetch, planning (todo), memory, identity, skill activation, workflow creation, Docker, ask user
- **Skills System**: LLM-driven skill activation from SKILL.md files (project and global scopes)
- **Planner**: Complexity-aware planning — Simple (direct), Medium (todo_write), Complex (create_workflow)
- **Agent Profile**: Instance-scoped resources with multi-agent isolation
- **Memory & Identity**: Persistent memory (append/rewrite) and identity profiles
- **Workflow Simplification**: Streamlined workflow engine with LLM-driven workflow generation
- **Skill-Workflow Integration**: Skills can be activated within workflow LLM steps
- **Task Scheduling**: CRUD task management with cron scheduling support
- **Conversation History**: Session persistence with LLM-based compression
- **Token Usage Statistics**: Per-model and per-session token tracking
- **Permissions System**: Layered permission system (persistent rules, session grants, workflow-scoped)
- **Keyring Security**: System keyring integration for API key management
- **Multi-provider Support**: Ollama, OpenAI-compatible, vLLM providers

## Planned Features

- **Deep Search**: `cliver deep-search "questions"` — multi-step research workflow with caching and resume
- **Multi-Model Orchestration**: Different models assigned to different capabilities via `model_routing` config
- **Remote Workflow Definitions**: Support for fetching workflow definitions from remote sources

Thank you for your interest in contributing to CLIver!
