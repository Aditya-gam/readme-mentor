"""Version management for readme-mentor application."""

import tomllib
from pathlib import Path


def get_version() -> str:
    """Get the current version from pyproject.toml."""
    try:
        # Get the project root directory (two levels up from this file)
        project_root = Path(__file__).parent.parent
        pyproject_path = project_root / "pyproject.toml"

        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)
            return data["project"]["version"]
    except (FileNotFoundError, KeyError, tomllib.TOMLDecodeError):
        # Fallback version if pyproject.toml cannot be read
        return "0.0.0"


# Export the version as a module-level variable
__version__ = get_version()
