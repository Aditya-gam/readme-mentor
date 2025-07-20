#!/usr/bin/env python3
"""Simple test script to check coverage for ingest.py without import issues."""

import sys
from pathlib import Path

import coverage

# Add the app directory to the path
sys.path.insert(0, str(Path(__file__).parent / "app"))


def run_coverage_test():
    """Run coverage analysis on ingest.py."""
    # Start coverage measurement
    cov = coverage.Coverage()
    cov.start()

    try:
        # Import and test the functions we can test without langchain
        from embeddings.ingest import (
            DEFAULT_BATCH_SIZE,
            DEFAULT_CHUNK_OVERLAP,
            DEFAULT_CHUNK_SIZE,
            DEFAULT_EMBEDDING_MODEL,
            _create_persist_directory,
            _extract_repo_slug,
            _generate_collection_name,
        )

        # Test _extract_repo_slug
        assert (
            _extract_repo_slug("https://github.com/octocat/Hello-World")
            == "octocat_Hello-World"
        )
        assert (
            _extract_repo_slug("https://github.com/octocat/Hello-World.git")
            == "octocat_Hello-World"
        )
        assert (
            _extract_repo_slug("https://github.com/octocat/Hello-World/")
            == "octocat_Hello-World"
        )
        assert (
            _extract_repo_slug("https://gitlab.com/user/repo") == "gitlab.com_user_repo"
        )
        assert (
            _extract_repo_slug("http://github.com/octocat/Hello-World")
            == "github.com_octocat_Hello-World"
        )
        assert _extract_repo_slug("https://github.com/octocat") == "github.com_octocat"

        # Test _create_persist_directory
        persist_dir = _create_persist_directory("test_repo")
        assert "test_repo" in str(persist_dir)
        assert "chroma" in str(persist_dir)

        # Test _generate_collection_name
        collection_name = _generate_collection_name("test_repo")
        assert collection_name.startswith("test_repo_")
        assert len(collection_name) == len("test_repo_") + 8  # 8 chars from uuid

        # Test constants
        assert DEFAULT_CHUNK_SIZE == 1024
        assert DEFAULT_CHUNK_OVERLAP == 128
        assert DEFAULT_BATCH_SIZE == 64
        assert DEFAULT_EMBEDDING_MODEL == "sentence-transformers/all-MiniLM-L6-v2"

        print("✅ Basic function tests passed")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

    finally:
        # Stop coverage measurement
        cov.stop()
        cov.save()

        # Generate coverage report
        cov.report()

        # Save HTML report
        cov.html_report(directory="htmlcov")

    return True


if __name__ == "__main__":
    success = run_coverage_test()
    sys.exit(0 if success else 1)
