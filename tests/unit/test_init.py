"""Unit tests for app initialization module."""

import importlib
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from dotenv import load_dotenv

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


class TestAppInit:
    """Test app initialization functionality."""

    def test_get_environment_default(self, cleanup_env_vars, reload_app_modules):
        """Test get_environment returns default value when ENVIRONMENT not set."""
        import app

        result = app.get_environment()
        assert result == "development"

    def test_get_environment_custom(self, cleanup_env_vars, reload_app_modules):
        """Test get_environment returns custom environment value."""
        os.environ["ENVIRONMENT"] = "production"
        import app

        result = app.get_environment()
        assert result == "production"

    def test_version_attributes(self, cleanup_env_vars, reload_app_modules):
        """Test that version attributes are available."""
        import app

        assert hasattr(app, "__version__")
        assert isinstance(app.__version__, str)
        assert len(app.__version__) > 0


class TestDotenvLoading:
    """Test dotenv loading functionality."""

    def test_dotenv_load_success(self, cleanup_env_vars, reload_app_modules):
        """Test successful dotenv loading from a temporary .env file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            env_file = Path(temp_dir) / ".env"
            env_content = "TEST_VAR=test_value"
            env_file.write_text(env_content)

            # Clear any existing TEST_VAR environment variable
            os.environ.pop("TEST_VAR", None)

            # Reload app modules to ensure fresh state
            for module_name in list(sys.modules.keys()):
                if module_name.startswith("app"):
                    del sys.modules[module_name]

            # Patch load_dotenv to use our test file
            with patch("dotenv.load_dotenv") as mock_load_dotenv:

                def side_effect(dotenv_path=None, **kwargs):
                    if dotenv_path is None:
                        load_dotenv(dotenv_path=env_file, override=True)
                    else:
                        load_dotenv(dotenv_path=dotenv_path, **kwargs)

                mock_load_dotenv.side_effect = side_effect

                # Import app which should load the .env file
                assert os.getenv("TEST_VAR") == "test_value"

    def test_dotenv_load_missing_file(self, cleanup_env_vars, reload_app_modules):
        """Test that no error occurs if the .env file is missing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            non_existent_env_path = Path(temp_dir) / ".env"
            with patch("app.config.env_path", non_existent_env_path):
                importlib.reload(sys.modules["app.config"])
                import app

                assert app is not None

    def test_dotenv_import_error(self, cleanup_env_vars, reload_app_modules):
        """Test app initialization when dotenv is not available."""
        with patch.dict(sys.modules, {"dotenv": None}):
            import app

            assert app is not None

    def test_dotenv_load_error(self, cleanup_env_vars, reload_app_modules):
        """Test app initialization when dotenv load fails."""
        with patch("dotenv.load_dotenv") as mock_load:
            mock_load.side_effect = Exception("Load failed")
            import app

            assert app is not None


class TestLoggingConfiguration:
    """Test logging configuration."""

    def test_logger_configured(self, cleanup_env_vars, reload_app_modules):
        """Test that the app logger is properly configured."""
        import app

        assert hasattr(app, "logger")
        assert app.logger.name == "app"
