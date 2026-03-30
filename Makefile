# =============================================================================
# Makefile — A/B Testing & Statistical Inference Dashboard
# =============================================================================
#
# Usage:
#   make install       — create venv and install all dependencies
#   make run           — start the Streamlit dashboard
#   make test          — run full test suite with coverage
#   make lint          — run ruff linter
#   make format        — auto-format with ruff
#   make clean         — remove build artefacts and caches
#   make help          — show this message
#
# Requires: Python 3.10+, pip
# =============================================================================

# ── Configuration ─────────────────────────────────────────────────────────────
PYTHON      := python3
VENV        := .venv
PIP         := $(VENV)/bin/pip
STREAMLIT   := $(VENV)/bin/streamlit
PYTEST      := $(VENV)/bin/pytest
RUFF        := $(VENV)/bin/ruff
APP         := app.py
PORT        := 8501

# Detect OS for activation command
ifeq ($(OS),Windows_NT)
    ACTIVATE := $(VENV)/Scripts/activate
    SEP      := ;
else
    ACTIVATE := $(VENV)/bin/activate
    SEP      := :
endif

.DEFAULT_GOAL := help

# ── Phony targets (not real files) ────────────────────────────────────────────
.PHONY: help install install-dev run test test-fast test-cov lint format \
        format-check clean clean-all setup check

# ─────────────────────────────────────────────────────────────────────────────
# HELP
# ─────────────────────────────────────────────────────────────────────────────

help:  ## Show available make targets
	@echo ""
	@echo "  A/B Testing Dashboard — Developer Commands"
	@echo "  ─────────────────────────────────────────────────────"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
	@echo ""


# ─────────────────────────────────────────────────────────────────────────────
# SETUP & INSTALL
# ─────────────────────────────────────────────────────────────────────────────

$(VENV)/bin/activate:  ## Create virtual environment
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip setuptools wheel

install: $(VENV)/bin/activate  ## Install production dependencies
	$(PIP) install -r requirements.txt
	@echo ""
	@echo "  ✓ Dependencies installed. Run: make run"

install-dev: $(VENV)/bin/activate  ## Install all dependencies including dev tools
	$(PIP) install -r requirements.txt
	$(PIP) install -e ".[dev]"
	@cp -n .env.template .env 2>/dev/null && echo "  ✓ .env created from template" || echo "  · .env already exists"
	@mkdir -p logs data/samples data/exports
	@echo ""
	@echo "  ✓ Dev environment ready. Run: make run"

setup: install-dev  ## Alias for install-dev (full first-time setup)


# ─────────────────────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────────────────────

run:  ## Start the Streamlit dashboard (default port 8501)
	$(STREAMLIT) run $(APP) \
		--server.port $(PORT) \
		--server.headless false \
		--browser.gatherUsageStats false

run-prod:  ## Start dashboard in production mode (headless)
	$(STREAMLIT) run $(APP) \
		--server.port $(PORT) \
		--server.headless true \
		--browser.gatherUsageStats false


# ─────────────────────────────────────────────────────────────────────────────
# TESTING
# ─────────────────────────────────────────────────────────────────────────────

test:  ## Run full test suite with coverage report
	$(PYTEST) tests/ \
		--cov=src \
		--cov-report=term-missing \
		--cov-report=html \
		--cov-fail-under=80 \
		-v

test-fast:  ## Run tests excluding slow property-based tests
	$(PYTEST) tests/ -m "not slow" \
		--cov=src \
		--cov-report=term-missing \
		-v --tb=short

test-bayesian:  ## Run only Bayesian inference tests
	$(PYTEST) tests/ -m bayesian -v

test-frequentist:  ## Run only frequentist statistical tests
	$(PYTEST) tests/ -m frequentist -v

test-cov:  ## Open HTML coverage report in browser
	$(PYTEST) tests/ --cov=src --cov-report=html -q
	@open htmlcov/index.html 2>/dev/null || xdg-open htmlcov/index.html 2>/dev/null || \
		echo "  Coverage report written to htmlcov/index.html"


# ─────────────────────────────────────────────────────────────────────────────
# LINTING & FORMATTING
# ─────────────────────────────────────────────────────────────────────────────

lint:  ## Run ruff linter on all source files
	$(RUFF) check src/ tests/ app.py config/

format:  ## Auto-format all source files with ruff
	$(RUFF) format src/ tests/ app.py config/
	$(RUFF) check --fix src/ tests/ app.py config/
	@echo "  ✓ Formatting complete"

format-check:  ## Check formatting without modifying files (CI use)
	$(RUFF) format --check src/ tests/ app.py config/
	$(RUFF) check src/ tests/ app.py config/

check: lint format-check test-fast  ## Run lint + format check + fast tests (pre-commit)


# ─────────────────────────────────────────────────────────────────────────────
# CLEAN
# ─────────────────────────────────────────────────────────────────────────────

clean:  ## Remove Python caches and test artefacts
	find . -type d -name "__pycache__"  -not -path "./.venv/*" | xargs rm -rf
	find . -type d -name ".pytest_cache" -not -path "./.venv/*" | xargs rm -rf
	find . -type d -name ".ruff_cache"   -not -path "./.venv/*" | xargs rm -rf
	find . -type d -name "htmlcov"       -not -path "./.venv/*" | xargs rm -rf
	find . -type f -name "*.pyc"         -not -path "./.venv/*" | xargs rm -f
	find . -type f -name ".coverage"     -not -path "./.venv/*" | xargs rm -f
	@echo "  ✓ Cache files removed"

clean-all: clean  ## Remove caches AND virtual environment (full reset)
	rm -rf $(VENV)
	@echo "  ✓ Virtual environment removed. Run: make install-dev"
