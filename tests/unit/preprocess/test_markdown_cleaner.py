"""Unit tests for markdown content cleaning and normalization.

This module imports tests from modular test files to maintain backward compatibility
while providing a cleaner, more organized test structure.
"""

# Import all test classes from modular files to maintain backward compatibility
from .test_cleaned_document import TestCleanedDocument
from .test_content_removal import (
    TestRemoveFencedCodeBlocks,
    TestRemoveHtmlComments,
    TestRemoveIndentedCodeBlocks,
    TestRemoveYamlFrontMatter,
)
from .test_encoding import TestDetectEncoding
from .test_integration import (
    TestCleanMarkdownContent,
    TestConvenienceFunctions,
    TestEdgeCases,
    TestLineNumberPreservation,
    TestLineOffsetMappingIntegration,
)
from .test_line_offset_mapper import TestLineOffsetMapper
from .test_markdown_conversion import TestMarkdownToPlainText

# Re-export all test classes for backward compatibility
__all__ = [
    "TestDetectEncoding",
    "TestRemoveHtmlComments",
    "TestRemoveYamlFrontMatter",
    "TestRemoveFencedCodeBlocks",
    "TestRemoveIndentedCodeBlocks",
    "TestMarkdownToPlainText",
    "TestLineOffsetMapper",
    "TestCleanedDocument",
    "TestLineOffsetMappingIntegration",
    "TestEdgeCases",
    "TestConvenienceFunctions",
    "TestCleanMarkdownContent",
    "TestLineNumberPreservation",
]
