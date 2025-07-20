"""Shared fixtures for readme-mentor tests."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture(scope="function")
def clean_data_dir():
    """Create a clean data directory for tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = os.getcwd()
        os.chdir(temp_dir)

        # Create data directory
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)

        yield data_dir

        os.chdir(original_cwd)


@pytest.fixture(scope="function")
def unique_collection_name():
    """Generate a unique collection name for tests."""
    import uuid

    return f"test_collection_{uuid.uuid4().hex[:8]}"


@pytest.fixture(scope="function")
def mock_github_token():
    """Mock GitHub token for tests."""
    with patch.dict(os.environ, {"GITHUB_TOKEN": "test_token"}):
        yield "test_token"


@pytest.fixture(scope="function")
def mock_openai_key():
    """Mock OpenAI API key for tests."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test_openai_key"}):
        yield "test_openai_key"


@pytest.fixture(scope="function")
def mock_secret_key():
    """Mock secret key for tests."""
    with patch.dict(os.environ, {"SECRET_KEY": "test_secret_key_123"}):
        yield "test_secret_key_123"
