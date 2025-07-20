"""Smoke tests for readme-mentor project."""

from pathlib import Path

import pytest


def test_project_structure() -> None:
    """Test that the core directory structure exists."""
    project_root = Path(__file__).parent.parent.parent

    # Check core directories exist
    assert (project_root / "app").exists(), "app directory should exist"
    assert (project_root / "scripts").exists(), "scripts directory should exist"
    assert (project_root / "tests").exists(), "tests directory should exist"
    # assert (project_root / "data").exists(), "data directory should exist"
    assert (project_root / "docs").exists(), "docs directory should exist"

    # Check app package is properly initialized
    assert (project_root / "app" / "__init__.py").exists(), (
        "app/__init__.py should exist"
    )


def test_pyproject_toml_exists() -> None:
    """Test that pyproject.toml exists and is valid."""
    project_root = Path(__file__).parent.parent.parent
    pyproject_path = project_root / "pyproject.toml"

    assert pyproject_path.exists(), "pyproject.toml should exist"

    # Basic validation that it's a TOML file
    content = pyproject_path.read_text()
    assert "[project]" in content, "pyproject.toml should contain [project] section"
    assert "readme-mentor" in content, "pyproject.toml should contain project name"


def test_app_package_importable() -> None:
    """Test that the app package can be imported."""
    try:
        import app

        assert app is not None, "app package should be importable"
    except ImportError as e:
        pytest.fail(f"app package should be importable: {e}")


if __name__ == "__main__":
    pytest.main([__file__])
