"""Smoke tests for GitHub repository content loader.

This module imports tests from modular test files to maintain backward compatibility
while providing a cleaner, more organized test structure.
"""

# Import all test classes from modular files to maintain backward compatibility
from .test_edge_cases import TestEdgeCases
from .test_file_fetching import TestFileFetching
from .test_url_parsing import TestURLParsing

# Re-export all test classes for backward compatibility
__all__ = [
    "TestURLParsing",
    "TestFileFetching",
    "TestEdgeCases",
]
