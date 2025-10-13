.PHONY: help
help: ## Display this help.
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

.PHONY: init
init: ## Init CLIver development dependencies
	uv venv
	uv sync --all-extras --dev --locked

##@ Development

.PHONY: build
build: ## Build CLIver distribution packages
	uv build

.PHONY: test
test: init ## Run tests
	uv run pytest

.PHONY: lint
lint: ## Run linter
	uv run ruff check

.PHONY: format
format: ## Format code
	uv run ruff format

.PHONY: clean
clean: ## Clean build artifacts
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

##@ Documentention

.PHONY: docs-build
docs-build: ## Build documentation site
	uv sync --group docs
	uv run mkdocs build

.PHONY: docs-serve
docs-serve: docs-build ## Serve documentation site locally
	uv run mkdocs serve

.PHONY: docs-deploy
docs-deploy: docs-build ## Deploy documentation to GitHub Pages
	uv run mkdocs gh-deploy

##@ Release

.PHONY: release
release: build ## Build and release to PyPI
	uv publish

