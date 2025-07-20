"""Metadata validation tests for the ingestion pipeline."""

import logging
import shutil
from pathlib import Path

import pytest

from app.embeddings.ingest import ingest_repository

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test repository (octocat/Hello-World)
TEST_REPO_URL = "https://github.com/octocat/Hello-World"


@pytest.fixture(scope="function")
def clean_data_dir():
    """Clean up data directory before and after each test."""
    data_dir = Path("data")

    # Clean up before test
    if data_dir.exists():
        shutil.rmtree(data_dir)

    yield data_dir

    # Clean up after test
    if data_dir.exists():
        shutil.rmtree(data_dir)


@pytest.fixture(scope="function")
def unique_collection_name():
    """Generate a unique collection name for each test."""
    import uuid

    return f"test_collection_{uuid.uuid4().hex[:8]}"


def test_ingest_repository_metadata_consistency(clean_data_dir, unique_collection_name):
    """Test that metadata is consistent across all stored documents."""
    logger.info("Starting metadata consistency test")

    # Run ingestion with in-memory storage for testing
    vectorstore = ingest_repository(
        TEST_REPO_URL,
        collection_name=unique_collection_name,
        persist_directory=None,  # Use in-memory storage
    )

    # Get all documents from the vector store
    # Note: This is a workaround since Chroma doesn't have a direct method to get all documents
    # We'll use a broad search to get most documents
    all_results = vectorstore.similarity_search("", k=100)  # Get up to 100 documents

    assert len(all_results) > 0, "No documents found in vector store"

    # Verify metadata consistency for all documents
    for i, doc in enumerate(all_results):
        metadata = doc.metadata

        # Check required fields
        required_fields = ["file", "start_line", "end_line"]
        for field in required_fields:
            assert field in metadata, (
                f"Missing required field '{field}' in document {i}"
            )
            assert metadata[field] is not None, (
                f"Field '{field}' is None in document {i}"
            )

        # Check data types
        assert isinstance(metadata["file"], str), (
            f"File field is not string in document {i}"
        )
        assert isinstance(metadata["start_line"], int), (
            f"start_line is not int in document {i}"
        )
        assert isinstance(metadata["end_line"], int), (
            f"end_line is not int in document {i}"
        )

        # Check logical constraints
        assert metadata["start_line"] >= 0, f"Invalid start_line in document {i}"
        assert metadata["end_line"] >= metadata["start_line"], (
            f"Invalid line range in document {i}"
        )

        # Check file exists
        file_path = Path(metadata["file"])
        assert file_path.exists(), f"Referenced file does not exist: {metadata['file']}"

        logger.info(
            f"Document {i}: {metadata['file']} (lines {metadata['start_line']}-{metadata['end_line']})"
        )


def test_ingest_repository_line_mapping_accuracy(
    clean_data_dir, unique_collection_name
):
    """Test that line mapping information is accurate."""
    logger.info("Starting line mapping accuracy test")

    # Run ingestion with in-memory storage for testing
    vectorstore = ingest_repository(
        TEST_REPO_URL,
        collection_name=unique_collection_name,
        persist_directory=None,  # Use in-memory storage
    )

    # Search for a distinctive phrase to get specific results
    search_phrase = "Hello World"
    results = vectorstore.similarity_search(search_phrase, k=3)

    assert len(results) > 0, "No results found for line mapping test"

    # Verify line mapping for each result
    for result in results:
        metadata = result.metadata

        # Check that line numbers are reasonable
        assert metadata["start_line"] >= 0, "Negative start line"
        assert metadata["end_line"] >= metadata["start_line"], "Invalid line range"

        # Check that the referenced file exists and has enough lines
        file_path = Path(metadata["file"])
        assert file_path.exists(), f"Referenced file does not exist: {metadata['file']}"

        # Read the actual file to verify line numbers
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Verify end_line doesn't exceed file length
        assert metadata["end_line"] <= len(lines), (
            f"End line {metadata['end_line']} exceeds file length {len(lines)}"
        )

        # Verify start_line is within file bounds
        assert metadata["start_line"] < len(lines), (
            f"Start line {metadata['start_line']} exceeds file length {len(lines)}"
        )

        logger.info(
            f"Line mapping verified: {metadata['file']} (lines {metadata['start_line']}-{metadata['end_line']})"
        )
