# Makefile for Bayesian Beam Model Selection Project
# =================================================

.PHONY: all install run test lint format clean help docs check-env

# Python and environment settings
PYTHON := python3
PIP := pip
VENV := .venv
VENV_PYTHON := $(VENV)/bin/python
VENV_PIP := $(VENV)/bin/pip

# Project settings
CONFIG := configs/default_config.yaml
OUTPUT_DIR := outputs

# Default target
all: install

# ==============================================================================
# ENVIRONMENT SETUP
# ==============================================================================

# Create virtual environment
venv:
	@echo "Creating virtual environment..."
	$(PYTHON) -m venv $(VENV)
	@echo "Virtual environment created at $(VENV)"
	@echo "Activate with: source $(VENV)/bin/activate"

# Install dependencies
install: venv
	@echo "Installing dependencies..."
	$(VENV_PIP) install --upgrade pip
	$(VENV_PIP) install -e ".[dev]"
	@echo "Installation complete!"

# Install without dev dependencies
install-prod: venv
	@echo "Installing production dependencies..."
	$(VENV_PIP) install --upgrade pip
	$(VENV_PIP) install -e .
	@echo "Production installation complete!"

# Check if environment is ready
check-env:
	@echo "Checking environment..."
	@$(VENV_PYTHON) -c "import numpy; import scipy; import pymc; import matplotlib; print('All dependencies installed!')" || \
		(echo "Missing dependencies. Run 'make install' first." && exit 1)

# ==============================================================================
# RUNNING THE PIPELINE
# ==============================================================================

# Run the full pipeline
run: check-env
	@echo "Running full Bayesian model selection pipeline..."
	$(VENV_PYTHON) main.py --config $(CONFIG) --output-dir $(OUTPUT_DIR) --stage all -v

# Run with default settings (quick alias)
run-default: run

# Run only data generation
run-data: check-env
	@echo "Running data generation stage..."
	$(VENV_PYTHON) main.py --config $(CONFIG) --output-dir $(OUTPUT_DIR) --stage data -v

# Run only calibration
run-calibration: check-env
	@echo "Running Bayesian calibration stage..."
	$(VENV_PYTHON) main.py --config $(CONFIG) --output-dir $(OUTPUT_DIR) --stage calibration -v

# Run only analysis
run-analysis: check-env
	@echo "Running model selection analysis..."
	$(VENV_PYTHON) main.py --config $(CONFIG) --output-dir $(OUTPUT_DIR) --stage analysis -v

# Run only report generation
run-report: check-env
	@echo "Generating reports..."
	$(VENV_PYTHON) main.py --config $(CONFIG) --output-dir $(OUTPUT_DIR) --stage report -v

# Run with custom aspect ratios
run-custom: check-env
	@echo "Running with custom aspect ratios..."
	$(VENV_PYTHON) main.py --config $(CONFIG) --output-dir $(OUTPUT_DIR) \
		-a 5 -a 10 -a 15 -a 20 -a 30 --stage all -v

# Run in debug mode
run-debug: check-env
	@echo "Running in debug mode..."
	$(VENV_PYTHON) main.py --config $(CONFIG) --output-dir $(OUTPUT_DIR) --stage all --debug

# ==============================================================================
# DEVELOPMENT
# ==============================================================================

# Run tests
test: check-env
	@echo "Running tests..."
	$(VENV_PYTHON) -m pytest tests/ -v --cov=apps --cov-report=term-missing

# Run tests with coverage report
test-cov: check-env
	@echo "Running tests with coverage..."
	$(VENV_PYTHON) -m pytest tests/ -v --cov=apps --cov-report=html
	@echo "Coverage report generated at htmlcov/index.html"

# Lint code
lint: check-env
	@echo "Linting code..."
	$(VENV_PYTHON) -m ruff check apps/ tests/ main.py
	$(VENV_PYTHON) -m mypy apps/ --ignore-missing-imports

# Format code
format: check-env
	@echo "Formatting code..."
	$(VENV_PYTHON) -m black apps/ tests/ main.py
	$(VENV_PYTHON) -m isort apps/ tests/ main.py

# Check formatting without making changes
format-check: check-env
	@echo "Checking code formatting..."
	$(VENV_PYTHON) -m black --check apps/ tests/ main.py
	$(VENV_PYTHON) -m isort --check-only apps/ tests/ main.py

# ==============================================================================
# DETAILED ANALYSIS (for presentations/meetings)
# ==============================================================================

# Generate detailed analysis figures
figures: check-env
	@echo "Generating detailed analysis figures..."
	$(VENV_PYTHON) examples/detailed_analysis.py
	$(VENV_PYTHON) examples/parameter_analysis.py
	@echo "Figures saved to outputs/figures/detailed/"

# Quick demo with visualizations
demo: check-env
	@echo "Running quick demo..."
	$(VENV_PYTHON) examples/quick_demo.py

# ==============================================================================
# DOCUMENTATION
# ==============================================================================

# Generate documentation
docs: check-env
	@echo "Generating documentation..."
	cd docs && make html
	@echo "Documentation generated at docs/_build/html/index.html"

# ==============================================================================
# UTILITIES
# ==============================================================================

# Clean generated files
clean:
	@echo "Cleaning generated files..."
	rm -rf $(OUTPUT_DIR)
	rm -rf __pycache__ apps/__pycache__ apps/**/__pycache__
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	rm -rf *.egg-info build dist
	rm -rf htmlcov .coverage
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	@echo "Clean complete!"

# Clean all including virtual environment
clean-all: clean
	@echo "Removing virtual environment..."
	rm -rf $(VENV)
	@echo "Full clean complete!"

# Show project structure
tree:
	@echo "Project structure:"
	@tree -I '__pycache__|*.egg-info|.git|.venv|outputs|htmlcov' --dirsfirst

# ==============================================================================
# HELP
# ==============================================================================

help:
	@echo ""
	@echo "Bayesian Beam Model Selection - Makefile Commands"
	@echo "================================================="
	@echo ""
	@echo "Setup Commands:"
	@echo "  make install      - Create venv and install all dependencies"
	@echo "  make install-prod - Install production dependencies only"
	@echo "  make venv         - Create virtual environment"
	@echo ""
	@echo "Run Commands:"
	@echo "  make run          - Run full pipeline (data → calibration → analysis)"
	@echo "  make run-data     - Generate synthetic measurement data only"
	@echo "  make run-calibration - Run Bayesian calibration only"
	@echo "  make run-analysis - Run model selection analysis only"
	@echo "  make run-report   - Generate reports only"
	@echo "  make run-custom   - Run with custom aspect ratios"
	@echo "  make run-debug    - Run in debug mode with verbose logging"
	@echo ""
	@echo "Development Commands:"
	@echo "  make test         - Run unit tests"
	@echo "  make test-cov     - Run tests with coverage report"
	@echo "  make lint         - Check code style and types"
	@echo "  make format       - Format code with black and isort"
	@echo ""
	@echo "Utility Commands:"
	@echo "  make clean        - Remove generated files"
	@echo "  make clean-all    - Remove all generated files including venv"
	@echo "  make help         - Show this help message"
	@echo ""
