"""Unit tests for version management."""

import tomllib
from pathlib import Path
from unittest.mock import patch

from app.version import get_version


class TestGetVersion:
    """Test version retrieval functionality."""

    def test_get_version_success(self):
        """Test successful version retrieval from pyproject.toml."""
        # Mock the pyproject.toml content
        mock_toml_content = b"""
[project]
name = "readme-mentor"
version = "1.2.3"
description = "A tool for analyzing README files"
"""

        with patch("builtins.open") as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = (
                mock_toml_content
            )

            with patch("app.version.Path") as mock_path:
                mock_path.return_value.parent.parent = Path("/mock/project/root")

                result = get_version()
                assert result == "1.2.3"

    def test_get_version_file_not_found(self):
        """Test version retrieval when pyproject.toml is not found."""
        with patch("builtins.open", side_effect=FileNotFoundError):
            with patch("app.version.Path") as mock_path:
                mock_path.return_value.parent.parent = Path("/nonexistent")

                result = get_version()
                assert result == "0.0.0"

    def test_get_version_invalid_toml(self):
        """Test version retrieval with invalid TOML file."""
        with patch("builtins.open") as mock_open:
            mock_open.side_effect = tomllib.TOMLDecodeError("Invalid TOML", "", 0)

            result = get_version()
            assert result == "0.0.0"

    def test_get_version_missing_version_key(self):
        """Test version retrieval when version key is missing from TOML."""
        with patch("builtins.open") as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = (
                b"[project]\nname = 'test'"
            )

            with patch("app.version.tomllib.load") as mock_load:
                mock_load.side_effect = KeyError("version")

                result = get_version()
                assert result == "0.0.0"

    def test_get_version_missing_project_section(self):
        """Test version retrieval when project section is missing."""
        with patch("builtins.open") as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = (
                b"[tool.poetry]\nname = 'test'"
            )

            with patch("app.version.tomllib.load") as mock_load:
                mock_load.side_effect = KeyError("project")

                result = get_version()
                assert result == "0.0.0"


class TestVersionModule:
    """Test version module attributes."""

    def test_version_attribute(self):
        """Test that __version__ attribute is available."""
        from app import version

        assert hasattr(version, "__version__")
        assert isinstance(version.__version__, str)
        assert len(version.__version__) > 0
