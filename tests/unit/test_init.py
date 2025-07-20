"""Unit tests for app initialization module."""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


class TestAppInit:
    """Test app initialization functionality."""

    def test_get_environment_default(self):
        """Test get_environment returns default value when ENVIRONMENT not set."""
        with patch.dict(os.environ, {}, clear=True):
            from app import get_environment

            result = get_environment()
            assert result == "development"

    def test_get_environment_custom(self):
        """Test get_environment returns custom environment value."""
        with patch.dict(os.environ, {"ENVIRONMENT": "production"}):
            from app import get_environment

            result = get_environment()
            assert result == "production"

    def test_version_attributes(self):
        """Test that version attributes are available."""
        import app

        assert hasattr(app, "__version__")
        assert hasattr(app, "get_version")
        assert isinstance(app.__version__, str)
        assert len(app.__version__) > 0

    def test_author_attributes(self):
        """Test that author attributes are available."""
        import app

        assert hasattr(app, "__author__")
        assert hasattr(app, "__email__")
        assert app.__author__ == "Aditya Gambhir"
        assert app.__email__ == "67105262+Aditya-gam@users.noreply.github.com"

    def test_logger_configured(self):
        """Test that logger is properly configured."""
        import app

        assert hasattr(app, "logger")
        assert app.logger.name == "app"


class TestDotenvLoading:
    """Test dotenv loading functionality."""

    def test_dotenv_load_success(self):
        """Test successful dotenv loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            env_file = Path(temp_dir) / ".env"
            env_content = "TEST_VAR=test_value"
            env_file.write_text(env_content)

            original_cwd = os.getcwd()
            os.chdir(temp_dir)

            try:
                # Clear app modules to force re-import
                for module_name in list(sys.modules.keys()):
                    if module_name.startswith("app"):
                        del sys.modules[module_name]

                # Import app which should trigger dotenv loading

                # Verify environment variable was loaded
                assert os.getenv("TEST_VAR") == "test_value"

            finally:
                os.chdir(original_cwd)

    def test_dotenv_load_missing_file(self):
        """Test dotenv loading when .env file doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            original_cwd = os.getcwd()
            os.chdir(temp_dir)

            try:
                # Clear app modules to force re-import
                for module_name in list(sys.modules.keys()):
                    if module_name.startswith("app"):
                        del sys.modules[module_name]

                # Import app which should handle missing .env gracefully
                import app

                # Should not raise any exceptions
                assert app is not None

            finally:
                os.chdir(original_cwd)

    def test_dotenv_import_error(self):
        """Test app initialization when dotenv is not available."""
        # Clear app modules to force re-import
        for module_name in list(sys.modules.keys()):
            if module_name.startswith("app"):
                del sys.modules[module_name]

        # Mock ImportError for dotenv
        with patch.dict(sys.modules, {"dotenv": None}):
            # This should not raise an exception
            import app

            assert app is not None

    def test_dotenv_load_error(self):
        """Test app initialization when dotenv load fails."""
        # Clear app modules to force re-import
        for module_name in list(sys.modules.keys()):
            if module_name.startswith("app"):
                del sys.modules[module_name]

        with patch("app.load_dotenv") as mock_load:
            mock_load.side_effect = Exception("Load failed")

            # This should not raise an exception
            import app

            assert app is not None


class TestLoggingConfiguration:
    """Test logging configuration."""

    def test_logging_basic_config(self):
        """Test that basic logging is configured."""
        import logging

        # Check that basic config has been called
        # This is a bit tricky to test directly, but we can verify the logger works
        logger = logging.getLogger("app")
        assert logger.level <= logging.INFO

    def test_app_logger_available(self):
        """Test that app logger is available and functional."""
        import app

        # Test that logger can be used
        app.logger.info("Test message")
        # Should not raise any exceptions


if __name__ == "__main__":
    pytest.main([__file__])
