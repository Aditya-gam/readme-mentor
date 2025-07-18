"""Tests for configuration management."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from app.config import Settings, settings


def test_load_dotenv_without_env_file():
    """Test that load_dotenv() doesn't raise exceptions when .env file doesn't exist."""
    # This test ensures that the application can start without a .env file
    # The load_dotenv() call in config.py should not raise any exceptions

    # Create a temporary directory without .env file
    with tempfile.TemporaryDirectory() as temp_dir:
        with patch("app.config.env_path", Path(temp_dir) / ".env"):
            # This should not raise any exceptions
            from app.config import settings  # noqa: F401

            # Verify that default values are used
            assert settings.APP_NAME == "readme-mentor"
            assert settings.APP_VERSION == "0.0.5"
            assert settings.DEBUG is False
            assert settings.LOG_LEVEL == "INFO"


def test_load_dotenv_with_env_file():
    """Test that load_dotenv() loads values from .env file when it exists."""
    # Create a temporary .env file
    with tempfile.TemporaryDirectory() as temp_dir:
        env_file = Path(temp_dir) / ".env"
        env_file.write_text("APP_NAME=test-app\nDEBUG=true\nLOG_LEVEL=DEBUG")

        # Test that load_dotenv can be called with the env file
        from dotenv import load_dotenv
        load_dotenv(env_file)

        # Verify that environment variables are loaded
        assert os.getenv("APP_NAME") == "test-app"
        assert os.getenv("DEBUG") == "true"
        assert os.getenv("LOG_LEVEL") == "DEBUG"


def test_settings_default_values():
    """Test that Settings class provides correct default values."""
    # Test that default values are set correctly
    assert settings.APP_NAME == "readme-mentor"
    assert settings.APP_VERSION == "0.0.5"
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
