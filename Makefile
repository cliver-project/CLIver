.PHONY: help init init-local test lint format dist clean

help: ## Show this help.
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

init: ## Initialize workspace dev dependencies
	@test -d .venv || uv venv
	uv sync --all-packages

init-local: ## Add the optional in-process local model backends
	uv sync --all-packages --extra local

test: init ## Run the workspace test suite
	uv run pytest

dist: ## Build the cliver-core sdist + wheel into dist/
	uv build --package cliver-core --out-dir dist

lint: ## Lint + format check
	uv run ruff check
	uv run ruff format --check

format: ## Auto-fix lint + format
	uv run ruff check --fix
	uv run ruff format

clean: ## Clean build artifacts and caches
	rm -rf dist/ build/ *.egg-info/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete
	find . -type d -name .pytest_cache -delete
	find . -type d -name .ruff_cache -delete
