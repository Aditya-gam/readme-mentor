"""Error handling tests for the ingestion pipeline."""

import logging
import subprocess
import sys
from pathlib import Path

import pytest

from app.embeddings.ingest import ingest_repository

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_ingest_repository_error_handling():
    """Test error handling for invalid repository URLs."""
    logger.info("Starting error handling test")

    # Test with invalid repository URL
    invalid_repo_url = "https://github.com/nonexistent/repo"

    with pytest.raises((Exception, ValueError)):
        ingest_repository(invalid_repo_url)

    logger.info("Error handling test completed")


def test_cli_help():
    """Test CLI help functionality."""
    logger.info("Starting CLI help test")

    # Test help command
    result = subprocess.run(
        [sys.executable, "scripts/ingest_cli.py", "--help"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent.parent,
    )

    assert result.returncode == 0, f"CLI help failed: {result.stderr}"
    assert "GitHub repository URL to ingest" in result.stdout
    assert "--chunk-size" in result.stdout
    assert "--file-glob" in result.stdout

    logger.info("CLI help test completed successfully")


def test_cli_basic_ingestion():
    """Test basic CLI ingestion functionality."""
    logger.info("Starting CLI basic ingestion test")

    # Test basic ingestion command
    result = subprocess.run(
        [
            sys.executable,
            "scripts/ingest_cli.py",
            "https://github.com/octocat/Hello-World",
            "--chunk-size",
            "512",
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent.parent,
    )

    # CLI should complete successfully
    assert result.returncode == 0, f"CLI ingestion failed: {result.stderr}"
    assert "Ingestion completed successfully" in result.stdout
    assert "Collection name:" in result.stdout

    logger.info("CLI basic ingestion test completed successfully")


def test_cli_invalid_repo():
    """Test CLI error handling for invalid repository."""
    logger.info("Starting CLI invalid repo test")

    # Test with invalid repository URL
    result = subprocess.run(
        [
            sys.executable,
            "scripts/ingest_cli.py",
            "https://github.com/nonexistent/repo",
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent.parent,
    )

    # CLI should fail gracefully
    assert result.returncode == 1, "CLI should fail for invalid repo"
    assert "Ingestion failed" in result.stdout

    logger.info("CLI invalid repo test completed successfully")
