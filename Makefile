# README-Mentor Makefile
# Follows clean code principles (DRY/KISS) and project standards

.PHONY: help install test lint format docs clean

# Default target
help:
	@echo "Available targets:"
	@echo "  install  - Install dependencies with Poetry"
	@echo "  test     - Run tests with pytest"
	@echo "  lint     - Run linting with Ruff"
	@echo "  format   - Format code with Ruff"
	@echo "  docs     - Build Sphinx documentation"
	@echo "  clean    - Clean build artifacts"

# Install dependencies
install:
	poetry install

# Run tests
test:
	poetry run pytest

# Run linting
lint:
	poetry run ruff check .

# Format code
format:
	poetry run ruff format .

# Build Sphinx documentation
docs:
	poetry run sphinx-build -b html docs/source docs/build/html

# Clean build artifacts
clean:
	rm -rf cache/
	rm -rf docs/build/
	rm -rf .ruff_cache/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
