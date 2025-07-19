#!/usr/bin/env python3
"""Manual test script for GitHub repository content loader."""

import logging
import sys
from pathlib import Path

from app.github.loader import (
    _create_repo_slug,
    _extract_repo_info,
    _is_markdown_file,
    _matches_readme_pattern,
    fetch_repository_files,
)

# Add the app directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def test_utility_functions():
    """Test utility functions."""
    print("Testing utility functions...")

    # Test URL parsing
    test_urls = [
        "https://github.com/octocat/Hello-World",
        "https://github.com/octocat/Hello-World.git",
        "https://github.com/octocat/Hello-World/",
    ]

    for url in test_urls:
        owner, repo = _extract_repo_info(url)
        slug = _create_repo_slug(owner, repo)
        print(f"  {url} -> {owner}/{repo} -> {slug}")

    # Test markdown detection
    test_files = ["README.md", "documentation.markdown", "script.py"]
    for file in test_files:
        is_md = _is_markdown_file(file)
        is_readme = _matches_readme_pattern(file)
        print(f"  {file}: markdown={is_md}, readme={is_readme}")

    print()


def test_repository_fetch():
    """Test repository file fetching."""
    print("Testing repository file fetching...")

    # Test with a small public repository
    repo_url = "https://github.com/octocat/Hello-World"

    try:
        print(f"Fetching files from {repo_url}...")
        saved_files = fetch_repository_files(repo_url)

        if saved_files:
            print(f"Successfully saved {len(saved_files)} files:")
            for file_path in saved_files:
                print(f"  - {file_path}")

                # Show file size
                file_size = Path(file_path).stat().st_size
                print(f"    Size: {file_size} bytes")
        else:
            print("No files were saved (possibly due to rate limits or missing files)")

    except Exception as e:
        print(f"Error: {e}")
        print("Note: This might be due to GitHub API rate limits for anonymous access.")
        print("Set GITHUB_TOKEN environment variable for better access.")


def main():
    """Main test function."""
    print("GitHub Repository Content Loader - Manual Test")
    print("=" * 50)

    test_utility_functions()
    test_repository_fetch()

    print("\nTest completed!")


if __name__ == "__main__":
    main()
