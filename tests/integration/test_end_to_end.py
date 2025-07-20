"""Tests for edge cases and error conditions to improve coverage.

This module imports tests from modular test files to maintain backward compatibility
while providing a cleaner, more organized test structure.
"""

# Import all test classes from modular files to maintain backward compatibility
from .test_app_edge_cases import TestAppInitEdgeCases, TestVersionEdgeCases
from .test_github_loader_edge_cases import TestGitHubLoaderEdgeCases
from .test_validators_edge_cases import TestValidatorsEdgeCases

# Re-export all test classes for backward compatibility
__all__ = [
    "TestGitHubLoaderEdgeCases",
    "TestValidatorsEdgeCases",
    "TestVersionEdgeCases",
    "TestAppInitEdgeCases",
]
