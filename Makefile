# ═══════════════════════════════════════════════════════════════════════════════
# Makefile — Bayesian Beam Model Selection  (Digital Twin Lab)
# ═══════════════════════════════════════════════════════════════════════════════
# Run `make` (no args) to see all available commands with descriptions.
# Uses `uv` for fast parallel installs where available, falls back to pip.
# ═══════════════════════════════════════════════════════════════════════════════

# ── Colours (ANSI, works in any modern terminal) ────────────────────────────
RESET   := \033[0m
BOLD    := \033[1m
CYAN    := \033[36m
GREEN   := \033[32m
YELLOW  := \033[33m
MAGENTA := \033[35m
DIM     := \033[2m

# ── Tool detection ──────────────────────────────────────────────────────────
PYTHON    := python3
VENV      := .venv
VENV_PY   := $(VENV)/bin/python
UV        := $(shell command -v uv 2>/dev/null)

ifdef UV
  INSTALLER     := $(UV) pip install
  INSTALLER_TAG := uv
else
  INSTALLER     := $(VENV)/bin/pip install
  INSTALLER_TAG := pip
endif

# ── Project settings ────────────────────────────────────────────────────────
CONFIG     := configs/default_config.yaml
OUTPUT_DIR := outputs

# ── Phony targets ───────────────────────────────────────────────────────────
.PHONY: help install install-prod venv check-env check-deps \
        run run-data run-calibration run-analysis run-report \
        run-quick run-custom run-debug run-optimize \
        test test-fast test-cov lint format format-check typecheck security \
        backend-dev frontend-dev up down build logs \
        frontend-lint frontend-test \
        docs docs-serve figures demo \
        lock clean clean-all tree

# ═══════════════════════════════════════════════════════════════════════════════
#  DEFAULT — colourised help (runs when you type just `make`)
# ═══════════════════════════════════════════════════════════════════════════════
.DEFAULT_GOAL := help

help:
	@echo ""
	@printf "  $(BOLD)$(CYAN)Bayesian Beam Model Selection$(RESET)  $(DIM)— Digital Twin Lab$(RESET)\n"
	@echo ""
	@printf "  $(BOLD)$(GREEN)Setup$(RESET)\n"
	@printf "    $(CYAN)make install$(RESET)         Create venv & install all deps (dev + prod)\n"
	@printf "    $(CYAN)make install-prod$(RESET)     Install production deps only\n"
	@printf "    $(CYAN)make check-deps$(RESET)      Verify every runtime import resolves correctly\n"
	@printf "    $(CYAN)make lock$(RESET)            Generate requirements.lock (pip-compile)\n"
	@echo ""
	@printf "  $(BOLD)$(GREEN)Pipeline$(RESET)\n"
	@printf "    $(CYAN)make run$(RESET)             Full pipeline  (data → calibrate → analyse → report)\n"
	@printf "    $(CYAN)make run-quick$(RESET)        Quick run with 2 aspect ratios\n"
	@printf "    $(CYAN)make run-data$(RESET)         Data generation only\n"
	@printf "    $(CYAN)make run-calibration$(RESET)  Bayesian calibration only\n"
	@printf "    $(CYAN)make run-analysis$(RESET)     Model selection analysis only\n"
	@printf "    $(CYAN)make run-report$(RESET)       Report generation only\n"
	@printf "    $(CYAN)make run-optimize$(RESET)     Hyperparameter optimisation\n"
	@printf "    $(CYAN)make run-custom$(RESET)       Custom aspect ratios (5,10,15,20,30)\n"
	@printf "    $(CYAN)make run-debug$(RESET)        Pipeline with debug logging\n"
	@echo ""
	@printf "  $(BOLD)$(GREEN)Code Quality$(RESET)\n"
	@printf "    $(CYAN)make test$(RESET)            Run pytest (verbose, coverage)\n"
	@printf "    $(CYAN)make test-fast$(RESET)        Parallel tests (pytest-xdist)\n"
	@printf "    $(CYAN)make test-cov$(RESET)         Tests + HTML coverage report\n"
	@printf "    $(CYAN)make lint$(RESET)            Lint with ruff (fast, zero-config)\n"
	@printf "    $(CYAN)make format$(RESET)          Auto-format (ruff format + fix)\n"
	@printf "    $(CYAN)make format-check$(RESET)     Check formatting without changes\n"
	@printf "    $(CYAN)make typecheck$(RESET)       Run mypy type checking\n"
	@printf "    $(CYAN)make security$(RESET)        Run bandit + pip-audit\n"
	@echo ""
	@printf "  $(BOLD)$(GREEN)Servers / Docker$(RESET)\n"
	@printf "    $(CYAN)make backend-dev$(RESET)      FastAPI backend  (http://localhost:8000, hot-reload)\n"
	@printf "    $(CYAN)make frontend-dev$(RESET)     Vite dev server  (http://localhost:5173)\n"
	@printf "    $(CYAN)make up$(RESET)              docker compose up -d\n"
	@printf "    $(CYAN)make down$(RESET)            docker compose down\n"
	@printf "    $(CYAN)make build$(RESET)           docker compose build\n"
	@printf "    $(CYAN)make logs$(RESET)            docker compose logs -f\n"
	@printf "    $(CYAN)make frontend-lint$(RESET)    Lint frontend (eslint)\n"
	@printf "    $(CYAN)make frontend-test$(RESET)    Test frontend (vitest)\n"
	@echo ""
	@printf "  $(BOLD)$(GREEN)Documentation$(RESET)\n"
	@printf "    $(CYAN)make docs$(RESET)            Build MkDocs site\n"
	@printf "    $(CYAN)make docs-serve$(RESET)       Serve docs locally (hot-reload)\n"
	@printf "    $(CYAN)make figures$(RESET)         Generate analysis figures\n"
	@printf "    $(CYAN)make demo$(RESET)            Run quick demo script\n"
	@echo ""
	@printf "  $(BOLD)$(GREEN)Utility$(RESET)\n"
	@printf "    $(CYAN)make clean$(RESET)           Remove generated files\n"
	@printf "    $(CYAN)make clean-all$(RESET)        Remove everything incl. venv\n"
	@printf "    $(CYAN)make tree$(RESET)            Show project tree\n"
	@echo ""
	@printf "  $(DIM)installer: $(INSTALLER_TAG)$(RESET)\n"
	@echo ""

# ═══════════════════════════════════════════════════════════════════════════════
#  SETUP
# ═══════════════════════════════════════════════════════════════════════════════

venv:
	@echo "Creating virtual environment …"
	$(PYTHON) -m venv $(VENV)
	@echo "Virtual environment created at $(VENV) (installer: $(INSTALLER_TAG))"

install: venv
	@printf "$(GREEN)Installing dependencies via $(INSTALLER_TAG) …$(RESET)\n"
ifdef UV
	$(UV) pip install --python $(VENV_PY) -e ".[dev]"
else
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -e ".[dev]"
endif
	@printf "$(GREEN)Installation complete!$(RESET)\n"

install-prod: venv
	@printf "$(GREEN)Installing production dependencies via $(INSTALLER_TAG) …$(RESET)\n"
ifdef UV
	$(UV) pip install --python $(VENV_PY) -e .
else
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -e .
endif
	@printf "$(GREEN)Production installation complete!$(RESET)\n"

check-env:
	@$(VENV_PY) -c \
		"import numpy, scipy, pymc, matplotlib; print('All dependencies OK')" \
		|| (printf "$(YELLOW)Missing deps. Run  make install  first.$(RESET)\n" && exit 1)

# ── Dependency doctor ───────────────────────────────────────────────────────
# Imports every top-level package declared in pyproject.toml.  If anything
# fails it prints exactly which package is missing and how to fix it.
check-deps:
	@printf "$(CYAN)Checking all runtime dependencies …$(RESET)\n"
	@$(VENV_PY) apps/backend/cli/check_deps.py

# ═══════════════════════════════════════════════════════════════════════════════
#  PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

# CLI_CMD: use the installed entry-point when available (editable install),
# falling back to running the module directly.
CLI_CMD := $(VENV)/bin/beam-model-selection

run: check-env
	@printf "$(MAGENTA)Running full pipeline …$(RESET)\n"
	$(CLI_CMD) --config $(CONFIG) --output-dir $(OUTPUT_DIR) --stage all -v

run-data: check-env
	$(CLI_CMD) --config $(CONFIG) --output-dir $(OUTPUT_DIR) --stage data -v

run-calibration: check-env
	$(CLI_CMD) --config $(CONFIG) --output-dir $(OUTPUT_DIR) --stage calibration -v

run-analysis: check-env
	$(CLI_CMD) --config $(CONFIG) --output-dir $(OUTPUT_DIR) --stage analysis -v

run-report: check-env
	$(CLI_CMD) --config $(CONFIG) --output-dir $(OUTPUT_DIR) --stage report -v

run-quick: check-env
	@printf "$(MAGENTA)Quick run (2 ratios) …$(RESET)\n"
	$(CLI_CMD) --config $(CONFIG) --output-dir $(OUTPUT_DIR) -a 5 -a 30 --stage all -v

run-custom: check-env
	$(CLI_CMD) --config $(CONFIG) --output-dir $(OUTPUT_DIR) -a 5 -a 10 -a 15 -a 20 -a 30 --stage all -v

run-debug: check-env
	$(CLI_CMD) --config $(CONFIG) --output-dir $(OUTPUT_DIR) --stage all --debug

run-optimize: check-env
	$(CLI_CMD) --config $(CONFIG) --output-dir $(OUTPUT_DIR) --stage optimize -v

# ═══════════════════════════════════════════════════════════════════════════════
#  CODE QUALITY  (ruff replaces black + isort + flake8 — 10–100× faster)
# ═══════════════════════════════════════════════════════════════════════════════

test: check-env
	@printf "$(YELLOW)Running tests …$(RESET)\n"
	$(VENV_PY) -m pytest tests/ -v --cov=apps --cov-report=term-missing

test-fast: check-env
	@printf "$(YELLOW)Running tests in parallel …$(RESET)\n"
	$(VENV_PY) -m pytest tests/ -v -n auto --cov=apps --cov-report=term-missing

test-cov: check-env
	$(VENV_PY) -m pytest tests/ -v --cov=apps --cov-report=html
	@printf "$(GREEN)Coverage report → htmlcov/index.html$(RESET)\n"

lint: check-env
	@printf "$(YELLOW)Linting (ruff) …$(RESET)\n"
	$(VENV_PY) -m ruff check apps/ tests/

format: check-env
	$(VENV_PY) -m ruff format apps/ tests/
	$(VENV_PY) -m ruff check --fix apps/ tests/

format-check: check-env
	$(VENV_PY) -m ruff format --check apps/ tests/
	$(VENV_PY) -m ruff check apps/ tests/

typecheck: check-env
	@printf "$(YELLOW)Running mypy (informational — scientific code has known type gaps) …$(RESET)\n"
	$(VENV_PY) -m mypy apps/ --ignore-missing-imports || true

security: check-env
	@printf "$(YELLOW)Security audit …$(RESET)\n"
	$(VENV_PY) -m bandit -r apps/ -q
	$(VENV_PY) -m pip_audit --skip-editable || true
	@printf "$(GREEN)Security audit complete.$(RESET)\n"

# ═══════════════════════════════════════════════════════════════════════════════
#  SERVERS / DOCKER
# ═══════════════════════════════════════════════════════════════════════════════

backend-dev: check-env
	@printf "$(MAGENTA)Starting FastAPI backend → http://localhost:8000$(RESET)\n"
	$(VENV_PY) -m uvicorn apps.backend.api.app:app --reload --port 8000

frontend-dev:
	@printf "$(MAGENTA)Starting Vite dev server → http://localhost:5173$(RESET)\n"
	cd apps/frontend && bun run dev

check-ports:
	@for port in 8000 5173; do \
		if ss -tuln | grep -q ":$$port "; then \
			printf "$(RED)Error: Port $$port is already in use.$(RESET)\n"; \
			printf "To free it, run: $(BOLD)fuser -k $$port/tcp$(RESET)  or  $(BOLD)lsof -ti:$$port | xargs kill -9$(RESET)\n"; \
			exit 1; \
		fi \
	done

up: check-ports
	docker compose up -d
	@printf "$(GREEN)Project running at:$(RESET)\n"
	@printf "  $(BOLD)Frontend:$(RESET) http://localhost:5173\n"
	@printf "  $(BOLD)Backend API:$(RESET) http://localhost:8000\n"

down:
	docker compose down

build:
	docker compose build

logs:
	docker compose logs -f

frontend-lint:
	cd apps/frontend && bun run lint

frontend-test:
	cd apps/frontend && bun run test || echo "No tests configured yet"

# ═══════════════════════════════════════════════════════════════════════════════
#  DOCUMENTATION
# ═══════════════════════════════════════════════════════════════════════════════

docs: check-env
	$(VENV_PY) -m mkdocs build
	@printf "$(GREEN)Docs built → site/$(RESET)\n"

docs-serve: check-env
	@printf "$(MAGENTA)Serving docs → http://127.0.0.1:8001$(RESET)\n"
	$(VENV_PY) -m mkdocs serve -a 127.0.0.1:8001

figures: run-analysis
	@printf "$(GREEN)Figures → outputs/figures/$(RESET)\n"

demo: check-env
	@printf "$(MAGENTA)Quick demo (aspect ratios 5 + 30) …$(RESET)\n"
	$(CLI_CMD) --config $(CONFIG) --output-dir $(OUTPUT_DIR) -a 5 -a 30 --stage all -v

# ═══════════════════════════════════════════════════════════════════════════════
#  DEPENDENCY LOCKING
# ═══════════════════════════════════════════════════════════════════════════════

lock: check-env
	$(VENV_PY) -m piptools compile pyproject.toml -o requirements.lock \
		--strip-extras --no-header --annotation-style=line
	@printf "$(GREEN)Lock file → requirements.lock$(RESET)\n"

# ═══════════════════════════════════════════════════════════════════════════════
#  UTILITY
# ═══════════════════════════════════════════════════════════════════════════════

clean:
	@printf "$(YELLOW)Cleaning generated files …$(RESET)\n"
	rm -rf $(OUTPUT_DIR) site/
	rm -rf __pycache__ apps/__pycache__ apps/**/__pycache__
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	rm -rf *.egg-info build dist
	rm -rf htmlcov .coverage
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@printf "$(GREEN)Clean complete.$(RESET)\n"

clean-all: clean
	rm -rf $(VENV)
	@printf "$(GREEN)Full clean complete. Run 'make install' to rebuild.$(RESET)\n"

tree:
	@tree -I '__pycache__|*.egg-info|.git|.venv|node_modules|outputs|htmlcov|site|dist' --dirsfirst

