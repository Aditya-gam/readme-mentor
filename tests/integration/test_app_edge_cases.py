"""Edge case tests for app initialization and version functionality."""

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


class TestVersionEdgeCases:
    """Test edge cases for version management."""

    def test_get_version_file_not_found(self):
        """Test get_version when pyproject.toml is not found."""
        with patch("builtins.open", side_effect=FileNotFoundError):
            with patch("app.version.Path") as mock_path:
                mock_path.return_value.parent.parent = Path("/nonexistent")

                from app.version import get_version

                result = get_version()
                assert result == "0.0.0"

    def test_get_version_invalid_toml(self):
        """Test get_version with invalid TOML file."""
        with patch("builtins.open") as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = (
                b"invalid toml content"
            )

            with patch("app.version.tomllib.load") as mock_load:
                mock_load.side_effect = Exception("Invalid TOML")

                from app.version import get_version

                result = get_version()
                assert result == "0.0.0"

    def test_get_version_missing_version_key(self):
        """Test get_version when version key is missing from TOML."""
        with patch("builtins.open") as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = (
                b"[project]\nname = 'test'"
            )

            with patch("app.version.tomllib.load") as mock_load:
                mock_load.side_effect = KeyError("version")

                from app.version import get_version

                result = get_version()
                assert result == "0.0.0"


class TestAppInitEdgeCases:
    """Test edge cases for app initialization."""

    def test_get_environment(self):
        """Test get_environment function."""
        with patch.dict(os.environ, {}, clear=True):
            from app import get_environment

            result = get_environment()
            assert result == "development"

        with patch.dict(os.environ, {"ENVIRONMENT": "production"}):
            from app import get_environment

            result = get_environment()
            assert result == "production"

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

    def test_extract_repo_info_no_match(self):
        """Test _extract_repo_info with URL that doesn't match pattern."""
        from app.github.loader import _extract_repo_info

        with pytest.raises(ValueError):
            _extract_repo_info("https://gitlab.com/user/repo")

    def test_validate_repo_url_normalize_error(self):
        """Test validate_repo_url when normalization fails."""
        with patch("app.utils.validators._normalize_github_url") as mock_normalize:
            mock_normalize.side_effect = ValueError("Invalid URL")

            with pytest.raises(ValueError):
                from app.utils.validators import validate_repo_url

                validate_repo_url("invalid-url")

    def test_validate_repo_url_invalid_pattern(self):
        """Test validate_repo_url with invalid URL pattern."""
        with pytest.raises(Exception):
            from app.utils.validators import validate_repo_url

            validate_repo_url("not-a-url")
