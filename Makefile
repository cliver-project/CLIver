.PHONY: help
help: ## Display this help.
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

.PHONY: init
init: ## Init CLIver development dependencies
	@test -d .venv || uv venv
	uv sync --all-extras --dev --locked

##@ Development

.PHONY: build
build: admin-build ## Build CLIver distribution packages
	uv build

.PHONY: test
test: init ## Run tests
	uv run pytest

.PHONY: lint
lint: ## Run linter and check formatting
	uv run ruff check
	uv run ruff format --check

.PHONY: format
format: ## Format code
	uv run ruff check --fix
	uv run ruff format

.PHONY: clean
clean: ## Clean build artifacts
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info/
	rm -rf admin/dist src/cliver/gateway/admin_dist
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

##@ Admin Portal

.PHONY: admin-install
admin-install: ## Install admin portal dependencies
	cd admin && npm install

.PHONY: admin-dev
admin-dev: ## Start admin portal dev server (Vite)
	cd admin && npm run dev

.PHONY: admin-build
admin-build: ## Build admin portal for production
	cd admin && npm run build
	rm -rf src/cliver/gateway/admin_dist
	cp -r admin/dist src/cliver/gateway/admin_dist

.PHONY: admin-lint
admin-lint: ## Lint admin portal TypeScript
	cd admin && npm run lint

.PHONY: admin-clean
admin-clean: ## Clean admin portal build artifacts
	rm -rf admin/dist admin/node_modules

##@ Documentation

.PHONY: docs-build
docs-build: ## Build documentation site
	uv sync --group docs
	uv run mkdocs build

.PHONY: docs-serve
docs-serve: docs-build ## Serve documentation site locally
	uv run mkdocs serve

##@ Release

.PHONY: release
release: build ## Build and release to PyPI
	uv publish

