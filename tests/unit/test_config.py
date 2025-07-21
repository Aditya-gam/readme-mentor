"""Tests for configuration management and environment variable handling."""

import importlib
import os
import sys
import tempfile
from pathlib import Path

import pytest
from dotenv import load_dotenv

from app.config import get_settings

# Ensure the app directory is in the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


@pytest.fixture(autouse=True)
def cleanup_env_vars():
    """Fixture to clean up environment variables after each test."""
    original_environ = os.environ.copy()
    yield
    os.environ.clear()
    os.environ.update(original_environ)


@pytest.fixture(autouse=True)
def reload_app_modules():
    """Fixture to reload app modules to ensure fresh state for each test."""
    yield
    for module_name in list(sys.modules.keys()):
        if module_name.startswith("app"):
            del sys.modules[module_name]


def test_load_dotenv_with_env_file(cleanup_env_vars, reload_app_modules):
    """Test that Settings loads values from a .env file."""
    with tempfile.TemporaryDirectory() as temp_dir:
        env_file = Path(temp_dir) / ".env"
        env_content = """APP_NAME=test-app
DEBUG=true
LOG_LEVEL=DEBUG
SECRET_KEY=test_secret_from_env
"""
        env_file.write_text(env_content)

        # Clear any existing environment variables that might interfere
        for key in ["APP_NAME", "DEBUG", "LOG_LEVEL", "SECRET_KEY"]:
            os.environ.pop(key, None)

        # Load the .env file into the environment
        load_dotenv(dotenv_path=env_file, override=True)

        # Get settings after loading the .env file
        settings = get_settings()

        assert settings.APP_NAME == "test-app"
        assert settings.DEBUG is True
        assert settings.LOG_LEVEL == "DEBUG"
        assert settings.SECRET_KEY == "test_secret_from_env"


def test_settings_default_values(cleanup_env_vars, reload_app_modules):
    """Test that Settings class provides correct default values when no .env is present or env vars are cleared."""
    # Ensure no relevant environment variables are set for this test
    settings = get_settings()
    assert settings.APP_NAME == "readme-mentor"
    assert settings.DEBUG is False
    assert settings.LOG_LEVEL == "INFO"
    assert settings.SECRET_KEY == "your_secret_key_here"


def test_settings_validation(cleanup_env_vars, reload_app_modules):
    """Test that Settings validation works correctly."""
    # Test with default secret key (should raise ValueError)
    settings = get_settings()
    # Ensure it's the default for this test
    settings.SECRET_KEY = "your_secret_key_here"
    with pytest.raises(ValueError, match="SECRET_KEY must be set"):
        settings.validate()

    # Test with a proper secret key
    os.environ["SECRET_KEY"] = "a-real-secret-key"
    settings = get_settings()
    settings.validate()  # Should not raise


def test_required_api_keys_handling(cleanup_env_vars, reload_app_modules):
    """Test that required API keys are properly handled."""
    # Clear any existing API key environment variables
    for key in ["GITHUB_TOKEN", "OPENAI_API_KEY"]:
        os.environ.pop(key, None)

    settings = get_settings()
    assert settings.GITHUB_TOKEN is None
    assert settings.OPENAI_API_KEY is None

    os.environ["GITHUB_TOKEN"] = "test_github_token"
    os.environ["OPENAI_API_KEY"] = "test_openai_key"
    settings = get_settings()
    assert settings.GITHUB_TOKEN == "test_github_token"
    assert settings.OPENAI_API_KEY == "test_openai_key"


def test_environment_functions(cleanup_env_vars, reload_app_modules):
    """Test the environment utility functions in app/__init__.py."""
    import app

    assert app.get_environment() == "development"

    os.environ["ENVIRONMENT"] = "production"
    importlib.reload(app)
    assert app.get_environment() == "production"
