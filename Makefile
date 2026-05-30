.PHONY: help
help: ## Display this help.
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

.PHONY: init
init: ## Init CLIver development dependencies
	@test -d .venv || uv venv
	uv sync --all-extras --dev --locked

##@ Development

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
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

##@ Admin Portal

ADMIN_DIST_DIR := src/cliver/gateway/admin_dist

.PHONY: admin-install
admin-install: ## Install admin portal dependencies
	cd admin && npm ci

.PHONY: admin-build
admin-build: admin-install ## Build admin portal for production
	cd admin && npm run build

.PHONY: admin-clean
admin-clean: ## Remove built admin portal from package
	rm -rf $(ADMIN_DIST_DIR)

.PHONY: admin-package
admin-package: admin-clean admin-build ## Build admin portal and copy into Python package
	mkdir -p $(ADMIN_DIST_DIR)
	cp -r admin/dist/* $(ADMIN_DIST_DIR)/

.PHONY: admin-dev
admin-dev: ## Start admin portal dev server (hot reload)
	cd admin && npm run dev

.PHONY: gateway
gateway: admin-build ## Build admin portal and start gateway
	uv run cliver gateway restart

.PHONY: gateway-dev
gateway-dev: ## Start gateway with admin portal dev proxy (hot reload)
	@echo "Start admin dev server: make admin-dev (in another terminal)"
	@echo "Then start gateway: uv run cliver gateway start"
	@echo "Access admin at http://localhost:5173/admin/ (Vite dev server proxies API to gateway)"

##@ Build & Release

.PHONY: build
build: admin-package ## Build CLIver distribution packages (wheel + sdist, includes admin portal)
	uv build

.PHONY: release
release: build ## Build and publish to PyPI (includes admin portal)
	uv publish

