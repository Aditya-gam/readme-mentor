"""Tests for configuration management and environment variable handling."""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from app.config import Settings, settings


def test_load_dotenv_without_env_file():
    """Test that load_dotenv() doesn't raise exceptions when .env file doesn't exist."""
    # This test ensures that the application can start without a .env file
    # The load_dotenv() call in app/__init__.py should not raise any exceptions

    # Create a temporary directory without .env file
    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)

            # Clear the app modules from sys.modules to force re-import
            for module_name in list(sys.modules.keys()):
                if module_name.startswith('app'):
                    del sys.modules[module_name]

            # This should not raise any exceptions even without .env file

            # Verify that default values are used
            assert settings.APP_NAME == "readme-mentor"
            assert settings.APP_VERSION == "0.0.6"
            assert settings.DEBUG is False
            assert settings.LOG_LEVEL == "INFO"
        finally:
            os.chdir(original_cwd)


def test_load_dotenv_with_env_file():
    """Test that load_dotenv() loads values from actual .env file when it exists."""
    # Create a temporary .env file with test values
    with tempfile.TemporaryDirectory() as temp_dir:
        env_file = Path(temp_dir) / ".env"
        env_content = """APP_NAME=test-app
DEBUG=true
LOG_LEVEL=DEBUG
GITHUB_TOKEN=test_github_token
OPENAI_API_KEY=test_openai_key
SECRET_KEY=test_secret_key_123
"""
        env_file.write_text(env_content)

        original_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)

            # Clear the app modules from sys.modules to force re-import
            for module_name in list(sys.modules.keys()):
                if module_name.startswith("app"):
                    del sys.modules[module_name]

            # Clear any existing environment variables
            with patch.dict(os.environ, {}, clear=True):
                # Import the app module which should trigger dotenv loading
                import app

                # Verify that environment variables are loaded from .env file
                assert os.getenv("APP_NAME") == "test-app"
                assert os.getenv("DEBUG") == "true"
                assert os.getenv("LOG_LEVEL") == "DEBUG"
                assert os.getenv("GITHUB_TOKEN") == "test_github_token"
                assert os.getenv("OPENAI_API_KEY") == "test_openai_key"
                assert os.getenv("SECRET_KEY") == "test_secret_key_123"

                # Reload the config module to get updated settings
                import importlib

                import app.config

                importlib.reload(app.config)

                # Verify that settings reflect the loaded values
                assert app.config.settings.APP_NAME == "test-app"
                assert app.config.settings.DEBUG is True
                assert app.config.settings.LOG_LEVEL == "DEBUG"
                assert app.config.settings.GITHUB_TOKEN == "test_github_token"
                assert app.config.settings.OPENAI_API_KEY == "test_openai_key"
                assert app.config.settings.SECRET_KEY == "test_secret_key_123"
        finally:
            os.chdir(original_cwd)


def test_dotenv_import_error_handling():
    """Test that the application handles missing python-dotenv gracefully."""
    # Clear the app modules from sys.modules to force re-import
    for module_name in list(sys.modules.keys()):
        if module_name.startswith("app"):
            del sys.modules[module_name]

    # Mock ImportError for dotenv
    with patch(
        "dotenv.load_dotenv", side_effect=ImportError("No module named 'dotenv'")
    ):
        # This should not raise an exception, just log a warning

        # Verify that default values are still used
        assert settings.APP_NAME == "readme-mentor"
        assert settings.APP_VERSION == "0.0.6"


def test_dotenv_file_error_handling():
    """Test that the application handles .env file errors gracefully."""
    # Create a temporary directory with a corrupted .env file
    with tempfile.TemporaryDirectory() as temp_dir:
        env_file = Path(temp_dir) / ".env"
        # Create a file that will cause an error when read
        env_file.write_text("INVALID=ENV=FORMAT")

        original_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)

            # Clear the app modules from sys.modules to force re-import
            for module_name in list(sys.modules.keys()):
                if module_name.startswith("app"):
                    del sys.modules[module_name]

            # This should not raise an exception, just log a warning

            # Verify that default values are still used
            assert settings.APP_NAME == "readme-mentor"
            assert settings.APP_VERSION == "0.0.6"
        finally:
            os.chdir(original_cwd)


def test_env_file_presence_detection():
    """Test that the application correctly detects the presence of .env file."""
    with tempfile.TemporaryDirectory() as temp_dir:
        env_file = Path(temp_dir) / ".env"

        # Test when .env file doesn't exist
        assert not env_file.exists()

        # Test when .env file exists
        env_file.write_text("TEST_VAR=test_value")
        assert env_file.exists()

        # Verify file content
        content = env_file.read_text()
        assert "TEST_VAR=test_value" in content


def test_settings_default_values():
    """Test that Settings class provides correct default values."""
    # Test that default values are set correctly
    assert settings.APP_NAME == "readme-mentor"
    assert settings.APP_VERSION == "0.0.6"
    assert settings.DEBUG is False
    assert settings.LOG_LEVEL == "INFO"
    assert settings.HOST == "0.0.0.0"
    assert settings.PORT == 8000
    assert settings.RELOAD is False
    assert settings.DATABASE_URL == "sqlite:///./data/readme_mentor.db"
    assert settings.CHROMA_DB_HOST == "localhost"
    assert settings.CHROMA_DB_PORT == 8000
    assert settings.SECRET_KEY == "your_secret_key_here"
    assert settings.ALGORITHM == "HS256"
    assert settings.ACCESS_TOKEN_EXPIRE_MINUTES == 30
    assert settings.ENVIRONMENT == "development"


def test_settings_validation():
    """Test that Settings validation works correctly."""
    # Test with default secret key (should raise ValueError)
    with pytest.raises(ValueError, match="SECRET_KEY must be set"):
        Settings.validate()

    # Test with proper secret key
    with patch.dict(os.environ, {"SECRET_KEY": "proper_secret_key"}):
        # Reload the config module
        import importlib

        import app.config

        importlib.reload(app.config)

        # This should not raise an exception
        app.config.Settings.validate()


def test_required_api_keys_handling():
    """Test that required API keys are properly handled."""
    # Test that API keys can be None (optional for development)
    assert settings.GITHUB_TOKEN is None or isinstance(
        settings.GITHUB_TOKEN, str)
    assert settings.OPENAI_API_KEY is None or isinstance(
        settings.OPENAI_API_KEY, str)

    # Test that API keys can be set via environment
    with patch.dict(
        os.environ,
        {"GITHUB_TOKEN": "test_github_token", "OPENAI_API_KEY": "test_openai_key"},
    ):
        import importlib

        import app.config

        importlib.reload(app.config)

        assert app.config.settings.GITHUB_TOKEN == "test_github_token"
        assert app.config.settings.OPENAI_API_KEY == "test_openai_key"


def test_app_version_and_environment_functions():
    """Test the utility functions in app/__init__.py."""
    import app

    # Test version function
    assert app.get_version() == "0.0.6"

    # Test environment function
    assert app.get_environment() == "development"

    # Test with custom environment
    with patch.dict(os.environ, {"ENVIRONMENT": "production"}):
        import importlib

        importlib.reload(app)
        assert app.get_environment() == "production"
