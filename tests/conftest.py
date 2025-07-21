"""Test configuration and fixtures."""

import importlib
import os
import shutil
import sys
import tempfile
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(scope="function")
def clean_data_dir():
    """Create a clean data directory for each test."""
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    data_dir = Path(temp_dir) / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    yield data_dir

    # Clean up
    shutil.rmtree(temp_dir)


@pytest.fixture(scope="function")
def unique_collection_name():
    """Generate a unique collection name for each test."""
    return f"test_collection_{uuid.uuid4().hex[:8]}"


@pytest.fixture(scope="function")
def mock_github_token():
    """Mock GitHub token for tests."""
    with patch.dict(os.environ, {"GITHUB_TOKEN": "test_token"}):
        yield "test_token"


@pytest.fixture(scope="function")
def mock_openai_key():
    """Mock OpenAI key for tests."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test_openai_key"}):
        yield "test_openai_key"


@pytest.fixture(scope="function")
def mock_secret_key():
    """Mock secret key for tests."""
    with patch.dict(os.environ, {"SECRET_KEY": "test_secret_key"}):
        yield "test_secret_key"


@pytest.fixture(autouse=True)
def reset_mocks():
    """Reset all mocks between tests to ensure isolation."""
    # This fixture runs automatically for every test
    # It ensures that mocks from previous tests don't interfere

    # Store original modules that might be affected
    _ = {}

    yield

    # Clean up mocks and reload modules that might have been affected
    modules_to_reload = [
        "app.rag.chain",
        "app.prompts",
        "app.config",
        "app.github.loader",
        "app.embeddings.ingest",
        "app.backend",
    ]

    for module_name in modules_to_reload:
        if module_name in sys.modules:
            try:
                importlib.reload(sys.modules[module_name])
            except (ImportError, AttributeError):
                # Module might not exist or might not be reloadable
                pass


@pytest.fixture(scope="function")
def mock_uuid():
    """Mock UUID generation for deterministic test results."""
    mock_uuid_obj = MagicMock()
    mock_uuid_obj.hex = "12345678-1234-1234-1234-123456789abc"
    mock_uuid_obj.__str__ = MagicMock(
        return_value="12345678-1234-1234-1234-123456789abc"
    )

    with patch("uuid.uuid4", return_value=mock_uuid_obj):
        yield mock_uuid_obj


@pytest.fixture(scope="function")
def mock_logger():
    """Mock logger to prevent log output during tests."""
    with patch("logging.getLogger") as mock_get_logger:
        mock_logger_instance = MagicMock()
        mock_get_logger.return_value = mock_logger_instance
        yield mock_logger_instance
