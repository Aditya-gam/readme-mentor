"""Edge case tests for app initialization."""

import os
import sys
from unittest.mock import patch


class TestAppInitEdgeCases:
    """Test edge cases for app initialization."""

    def test_get_environment_with_environment_variable(self):
        """Test get_environment with ENVIRONMENT variable set."""
        with patch.dict(os.environ, {"ENVIRONMENT": "production"}):
            from app import get_environment

            result = get_environment()
            assert result == "production"

    def test_get_environment_without_environment_variable(self):
        """Test get_environment without ENVIRONMENT variable."""
        with patch.dict(os.environ, {}, clear=True):
            from app import get_environment

            result = get_environment()
            assert result == "development"

    def test_get_environment_with_empty_environment_variable(self):
        """Test get_environment with empty ENVIRONMENT variable."""
        with patch.dict(os.environ, {"ENVIRONMENT": ""}):
            from app import get_environment

            result = get_environment()
            assert result == ""

    def test_dotenv_import_error_handling(self):
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

    def test_dotenv_load_error_handling(self):
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

    def test_app_attributes_availability(self):
        """Test that app attributes are available after initialization."""
        # Clear app modules to force re-import
        for module_name in list(sys.modules.keys()):
            if module_name.startswith("app"):
                del sys.modules[module_name]

        import app

        # Test that required attributes are available
        assert hasattr(app, "__version__")
        assert hasattr(app, "__author__")
        assert hasattr(app, "__email__")
        assert hasattr(app, "get_version")
        assert hasattr(app, "get_environment")
        assert hasattr(app, "logger")

        # Test that attributes have expected values
        assert isinstance(app.__version__, str)
        assert app.__author__ == "Aditya Gambhir"
        assert app.__email__ == "67105262+Aditya-gam@users.noreply.github.com"

    def test_logging_configuration(self):
        """Test that logging is properly configured."""
        # Clear app modules to force re-import
        for module_name in list(sys.modules.keys()):
            if module_name.startswith("app"):
                del sys.modules[module_name]

        import app

        # Test that logger is configured
        assert hasattr(app, "logger")
        assert app.logger.name == "app"
        assert app.logger.level <= 20  # INFO or lower
