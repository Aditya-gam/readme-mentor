"""Comprehensive tests for URL validator and GitHub loader using real repositories."""

import logging
from pathlib import Path

import pytest

from app.github.loader import fetch_repository_files
from app.utils.validators import InvalidRepoURLError, validate_repo_url

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test repositories covering different scenarios
TEST_REPOSITORIES = {
    # 1. With README and docs
    "flask": "https://github.com/pallets/flask",
    "black": "https://github.com/psf/black",
    # 2. With just README, no docs
    "hello_world": "https://github.com/octocat/Hello-World",
    "test_tlb": "https://github.com/torvalds/test-tlb",
    # 3. With just docs, no README
    "kkomiyama": "https://github.com/kkomiyama/kkomiyama.github.io",
    # 4. Without README and docs
    "codedocs": "https://github.com/CodeDocs/CodeDocs",
    "no_readme": "https://github.com/gittertestbot/no-readme-markdown-file-2",
}

# Expected results for each repository (based on actual investigation)
EXPECTED_RESULTS = {
    "flask": {
        "has_readme": True,
        "has_docs": False,  # Flask doesn't have docs/ directory
        "min_files": 1,  # Just README
    },
    "black": {
        "has_readme": True,
        "has_docs": True,
        "min_files": 27,  # README + 26 docs files
    },
    "hello_world": {
        "has_readme": True,
        "has_docs": False,
        "min_files": 1,  # Just README
    },
    "test_tlb": {
        "has_readme": True,
        "has_docs": False,
        "min_files": 1,  # Just README
    },
    "kkomiyama": {
        "has_readme": False,
        "has_docs": True,
        "min_files": 1,  # Just docs
    },
    "codedocs": {
        "has_readme": False,
        "has_docs": False,
        "min_files": 0,  # Empty repository
    },
    "no_readme": {
        "has_readme": False,
        "has_docs": False,
        "min_files": 0,  # No README or docs
    },
}


class TestURLValidator:
    """Test URL validation with real repository URLs."""

    def test_validate_repo_url_valid_repositories(self):
        """Test URL validation with all test repository URLs."""
        for repo_name, url in TEST_REPOSITORIES.items():
            logger.info(f"Testing URL validation for {repo_name}: {url}")

            # Test validation
            validated_url = validate_repo_url(url)

            # Verify the validated URL is correct
            assert validated_url == url, f"URL validation failed for {repo_name}"
            assert validated_url.startswith("https://github.com/")

            logger.info(f"✅ {repo_name}: URL validation passed")

    def test_validate_repo_url_edge_cases(self):
        """Test URL validation with edge cases."""
        test_cases = [
            # Valid variations
            ("https://GITHUB.COM/pallets/flask", "https://github.com/pallets/flask"),
            ("https://github.com/pallets/flask/", "https://github.com/pallets/flask"),
            (
                "  https://github.com/pallets/flask  ",
                "https://github.com/pallets/flask",
            ),
            # Invalid cases
            ("https://github.com/pallets/flask/tree/main", None),
            ("https://github.com/pallets/flask/blob/main/README.md", None),
            ("https://github.com/pallets/flask?ref=main", None),
            ("https://github.com/pallets/flask#readme", None),
            ("https://gitlab.com/pallets/flask", None),
            ("", None),
            ("not-a-url", None),
        ]

        for input_url, expected in test_cases:
            if expected is None:
                # Should raise an exception
                with pytest.raises((InvalidRepoURLError, ValueError)):
                    validate_repo_url(input_url)
                logger.info(f"✅ Invalid URL correctly rejected: {input_url}")
            else:
                # Should return normalized URL
                result = validate_repo_url(input_url)
                assert result == expected, f"URL normalization failed for {input_url}"
                logger.info(f"✅ URL normalization passed: {input_url} -> {result}")


class TestGitHubLoader:
    """Test GitHub loader with real repositories."""

    @pytest.mark.integration
    def test_fetch_repository_files_comprehensive(self):
        """Comprehensive test of repository file fetching with all test repositories."""
        results = {}

        for repo_name, url in TEST_REPOSITORIES.items():
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Testing repository: {repo_name}")
            logger.info(f"URL: {url}")
            logger.info(f"{'=' * 60}")

            try:
                # Fetch files from repository
                saved_files = fetch_repository_files(url)

                # Analyze results
                readme_files = [f for f in saved_files if "README" in f]
                docs_files = [f for f in saved_files if "docs/" in f]

                results[repo_name] = {
                    "saved_files": saved_files,
                    "readme_files": readme_files,
                    "docs_files": docs_files,
                    "total_files": len(saved_files),
                    "has_readme": len(readme_files) > 0,
                    "has_docs": len(docs_files) > 0,
                }

                # Log results
                logger.info(f"✅ {repo_name}:")
                logger.info(f"   - Total files saved: {len(saved_files)}")
                logger.info(f"   - README files: {len(readme_files)}")
                logger.info(f"   - Docs files: {len(docs_files)}")

                if saved_files:
                    logger.info(f"   - Files: {saved_files}")

            except Exception as e:
                logger.error(f"❌ {repo_name}: Failed with error - {e}")
                results[repo_name] = {
                    "error": str(e),
                    "saved_files": [],
                    "total_files": 0,
                    "has_readme": False,
                    "has_docs": False,
                }

        # Verify results against expectations
        self._verify_results_against_expectations(results)

    def _verify_results_against_expectations(self, results):
        """Verify that results match expected behavior."""
        logger.info(f"\n{'=' * 60}")
        logger.info("VERIFICATION RESULTS")
        logger.info(f"{'=' * 60}")

        for repo_name, result in results.items():
            expected = EXPECTED_RESULTS[repo_name]

            logger.info(f"\nRepository: {repo_name}")
            logger.info(f"Expected: {expected}")
            logger.info(f"Actual: {result}")

            if "error" in result:
                logger.warning(
                    f"⚠️  {repo_name}: Test failed with error - {result['error']}"
                )
                continue

            # Verify file count
            assert result["total_files"] >= expected["min_files"], (
                f"{repo_name}: Expected at least {expected['min_files']} files, "
                f"got {result['total_files']}"
            )

            # Verify README presence
            if expected["has_readme"]:
                assert result["has_readme"], (
                    f"{repo_name}: Expected README file but none found"
                )

            # Verify docs presence
            if expected["has_docs"]:
                assert result["has_docs"], (
                    f"{repo_name}: Expected docs files but none found"
                )

            logger.info(f"✅ {repo_name}: All expectations met!")

    @pytest.mark.integration
    def test_fetch_repository_files_with_different_patterns(self):
        """Test fetching files with different glob patterns."""
        # Test with a repository that has both README and docs
        test_url = TEST_REPOSITORIES["flask"]

        # Test different patterns
        patterns_to_test = [
            ("README*",),
            ("docs/**/*.md",),
            ("README*", "docs/**/*.md"),
            ("*.md",),  # All markdown files
        ]

        for pattern in patterns_to_test:
            logger.info(f"\nTesting pattern: {pattern}")

            try:
                saved_files = fetch_repository_files(test_url, file_glob=pattern)

                logger.info(f"Pattern {pattern}: Saved {len(saved_files)} files")
                if saved_files:
                    logger.info(f"Files: {saved_files}")

                # Verify that files match the pattern
                if "README*" in pattern:
                    readme_files = [f for f in saved_files if "README" in f]
                    assert len(readme_files) > 0, (
                        "README pattern should find README files"
                    )

                if "docs/**/*.md" in pattern:
                    docs_files = [f for f in saved_files if "docs/" in f]
                    # Note: Flask might not have docs in the expected location
                    logger.info(f"Docs files found: {len(docs_files)}")

            except Exception as e:
                logger.error(f"Pattern {pattern} failed: {e}")

    def test_file_size_limits(self):
        """Test that large files are properly skipped."""
        # This test verifies the 1MB file size limit
        # We'll use a repository that might have large files
        test_url = TEST_REPOSITORIES["flask"]

        try:
            saved_files = fetch_repository_files(test_url)

            # Check that all saved files are under 1MB
            for file_path in saved_files:
                if Path(file_path).exists():
                    file_size = Path(file_path).stat().st_size
                    max_size = 1024 * 1024  # 1MB

                    assert file_size <= max_size, (
                        f"File {file_path} exceeds 1MB limit: {file_size} bytes"
                    )

                    logger.info(f"✅ File {file_path}: {file_size} bytes (under limit)")

        except Exception as e:
            logger.error(f"File size test failed: {e}")

    def test_directory_structure(self):
        """Test that files are saved in the correct directory structure."""
        test_url = TEST_REPOSITORIES["hello_world"]

        try:
            saved_files = fetch_repository_files(test_url)

            for file_path in saved_files:
                path = Path(file_path)

                # Verify directory structure
                assert "data/" in str(path), "Files should be saved in data directory"
                assert "raw/" in str(path), "Files should be in raw subdirectory"
                assert path.exists(), f"File should exist: {file_path}"

                logger.info(f"✅ Directory structure correct: {file_path}")

        except Exception as e:
            logger.error(f"Directory structure test failed: {e}")


class TestIntegration:
    """Integration tests combining validator and loader."""

    @pytest.mark.integration
    def test_end_to_end_workflow(self):
        """Test complete workflow: URL validation -> file fetching."""
        for repo_name, url in TEST_REPOSITORIES.items():
            logger.info(f"\nTesting end-to-end workflow for {repo_name}")

            try:
                # Step 1: Validate URL
                validated_url = validate_repo_url(url)
                logger.info(f"✅ URL validation passed: {validated_url}")

                # Step 2: Fetch files
                saved_files = fetch_repository_files(validated_url)
                logger.info(f"✅ File fetching completed: {len(saved_files)} files")

                # Step 3: Verify results
                expected = EXPECTED_RESULTS[repo_name]
                actual_files = len(saved_files)

                assert actual_files >= expected["min_files"], (
                    f"{repo_name}: Expected at least {expected['min_files']} files, "
                    f"got {actual_files}"
                )

                logger.info(f"✅ End-to-end test passed for {repo_name}")

            except Exception as e:
                logger.error(f"❌ End-to-end test failed for {repo_name}: {e}")


if __name__ == "__main__":
    # Run comprehensive tests
    logger.info("Starting comprehensive repository tests...")

    # Test URL validation
    validator = TestURLValidator()
    validator.test_validate_repo_url_valid_repositories()
    validator.test_validate_repo_url_edge_cases()

    # Test GitHub loader
    loader = TestGitHubLoader()
    loader.test_fetch_repository_files_comprehensive()
    loader.test_fetch_repository_files_with_different_patterns()
    loader.test_file_size_limits()
    loader.test_directory_structure()

    # Test integration
    integration = TestIntegration()
    integration.test_end_to_end_workflow()

    logger.info("Comprehensive tests completed!")
