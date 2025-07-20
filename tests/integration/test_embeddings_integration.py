"""Integration tests for the ingestion pipeline.

This module imports tests from modular test files to maintain backward compatibility
while providing a cleaner, more organized test structure.
"""

# Import all test functions from modular files to maintain backward compatibility
from .test_error_handling import (
    test_cli_basic_ingestion,
    test_cli_help,
    test_cli_invalid_repo,
    test_ingest_repository_error_handling,
)
from .test_ingest_basic import (
    test_ingest_repository_basic,
    test_ingest_repository_chunking_parameters,
    test_ingest_repository_custom_file_patterns,
    test_ingest_repository_full_pipeline,
    test_ingest_repository_no_docs_folder,
    test_ingest_repository_persistent_storage,
)
from .test_metadata_validation import (
    test_ingest_repository_line_mapping_accuracy,
    test_ingest_repository_metadata_consistency,
)
from .test_vector_search import (
    test_ingest_repository_content_accuracy,
    test_ingest_repository_vector_search,
)

# Re-export all test functions for backward compatibility
__all__ = [
    "test_ingest_repository_basic",
    "test_ingest_repository_full_pipeline",
    "test_ingest_repository_no_docs_folder",
    "test_ingest_repository_custom_file_patterns",
    "test_ingest_repository_persistent_storage",
    "test_ingest_repository_chunking_parameters",
    "test_ingest_repository_vector_search",
    "test_ingest_repository_content_accuracy",
    "test_ingest_repository_metadata_consistency",
    "test_ingest_repository_line_mapping_accuracy",
    "test_ingest_repository_error_handling",
    "test_cli_help",
    "test_cli_basic_ingestion",
    "test_cli_invalid_repo",
]
