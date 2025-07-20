"""Unit tests for the ingest module.

This module imports tests from modular test files to maintain backward compatibility
while providing a cleaner, more organized test structure.
"""

# Import all test classes from modular files to maintain backward compatibility
from .test_chunking import TestChunkTextWithLineMapping, TestProcessFileForChunking
from .test_embeddings import TestGenerateEmbeddingsBatch
from .test_ingest_repository import TestIngestRepository
from .test_ingest_utils import (
    TestCreatePersistDirectory,
    TestExtractRepoSlug,
    TestGenerateCollectionName,
)

# Re-export all test classes for backward compatibility
__all__ = [
    "TestExtractRepoSlug",
    "TestCreatePersistDirectory",
    "TestGenerateCollectionName",
    "TestChunkTextWithLineMapping",
    "TestProcessFileForChunking",
    "TestGenerateEmbeddingsBatch",
    "TestIngestRepository",
]
