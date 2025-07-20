"""Integration tests for the ingestion pipeline.

This module tests the complete ingestion pipeline including text chunking,
embedding generation, and vector store storage.
"""

import logging
import shutil
from pathlib import Path

import pytest
from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.embeddings.ingest import ingest_repository

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test repository (octocat/Hello-World)
TEST_REPO_URL = "https://github.com/octocat/Hello-World"
TEST_REPO_SLUG = "octocat_Hello-World"


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


def test_ingest_repository_basic(clean_data_dir, unique_collection_name):
    """Test basic repository ingestion functionality."""
    logger.info("Starting basic ingestion test")

    # Run ingestion with in-memory storage for testing
    vectorstore = ingest_repository(
        TEST_REPO_URL,
        collection_name=unique_collection_name,
        persist_directory=None,  # Use in-memory storage
    )

    # Verify vectorstore is created
    assert vectorstore is not None
    assert isinstance(vectorstore, Chroma)

    # Verify data directory structure
    raw_dir = clean_data_dir / TEST_REPO_SLUG / "raw"

    assert raw_dir.exists(), f"Raw directory not created: {raw_dir}"
    # Note: chroma_dir is not created when using in-memory storage

    # Verify at least one file was downloaded
    raw_files = list(raw_dir.rglob("*"))
    assert len(raw_files) > 0, "No files downloaded to raw directory"

    logger.info(f"Found {len(raw_files)} files in raw directory")


def test_ingest_repository_vector_search(clean_data_dir, unique_collection_name):
    """Test vector search functionality after ingestion."""
    logger.info("Starting vector search test")

    # Run ingestion with in-memory storage for testing
    vectorstore = ingest_repository(
        TEST_REPO_URL,
        collection_name=unique_collection_name,
        persist_directory=None,  # Use in-memory storage
    )

    # Test search functionality
    # Search for distinctive phrases from the Hello-World README
    test_queries = ["Hello World", "GitHub", "repository", "octocat"]

    for query in test_queries:
        logger.info(f"Testing search query: '{query}'")

        # Perform similarity search
        results = vectorstore.similarity_search(query, k=3)

        # Verify results
        assert len(results) > 0, f"No results found for query: {query}"

        # Verify result structure
        for result in results:
            assert isinstance(result, Document)
            assert hasattr(result, "page_content")
            assert hasattr(result, "metadata")

            # Verify metadata fields
            metadata = result.metadata
            assert "file" in metadata, "Missing 'file' in metadata"
            assert "start_line" in metadata, "Missing 'start_line' in metadata"
            assert "end_line" in metadata, "Missing 'end_line' in metadata"

            # Verify metadata values are not empty
            assert metadata["file"], "Empty file path in metadata"
            assert metadata["start_line"] is not None, "None start_line in metadata"
            assert metadata["end_line"] is not None, "None end_line in metadata"

            # Verify line numbers are valid
            assert metadata["start_line"] >= 0, "Invalid start_line (negative)"
            assert metadata["end_line"] >= metadata["start_line"], "Invalid line range"

            logger.info(
                f"Found result: {metadata['file']} (lines {metadata['start_line']}-{metadata['end_line']})"
            )


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


def test_ingest_repository_content_accuracy(clean_data_dir, unique_collection_name):
    """Test that search results contain the expected content."""
    logger.info("Starting content accuracy test")

    # Run ingestion with in-memory storage for testing
    vectorstore = ingest_repository(
        TEST_REPO_URL,
        collection_name=unique_collection_name,
        persist_directory=None,  # Use in-memory storage
    )

    # Search for a specific phrase that should be in the README
    search_phrase = "Hello World"
    results = vectorstore.similarity_search(search_phrase, k=5)

    assert len(results) > 0, f"No results found for phrase: {search_phrase}"

    # Check that at least one result contains the search phrase
    found_phrase = False
    for result in results:
        if search_phrase.lower() in result.page_content.lower():
            found_phrase = True
            logger.info(f"Found phrase in result: {result.metadata['file']}")
            break

    assert found_phrase, f"Search phrase '{search_phrase}' not found in any results"


def test_ingest_repository_line_mapping_accuracy(
    clean_data_dir, unique_collection_name
):
    """Test that line mapping in metadata is accurate."""
    logger.info("Starting line mapping accuracy test")

    # Run ingestion with in-memory storage for testing
    vectorstore = ingest_repository(
        TEST_REPO_URL,
        collection_name=unique_collection_name,
        persist_directory=None,  # Use in-memory storage
    )

    # Search for a distinctive phrase
    search_phrase = "Hello World"
    results = vectorstore.similarity_search(search_phrase, k=3)

    assert len(results) > 0, f"No results found for phrase: {search_phrase}"

    # Check line mapping accuracy for the first result
    result = results[0]
    metadata = result.metadata
    file_path = Path(metadata["file"])

    # Read the original file
    assert file_path.exists(), f"File not found: {file_path}"

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Check that line numbers are within valid range
    start_line = metadata["start_line"]
    end_line = metadata["end_line"]

    assert start_line < len(lines), (
        f"start_line {start_line} exceeds file length {len(lines)}"
    )
    assert end_line < len(lines), (
        f"end_line {end_line} exceeds file length {len(lines)}"
    )

    # Extract the lines from the original file
    file_lines = lines[start_line : end_line + 1]
    file_content = "".join(file_lines)

    # Check that the chunk content is contained in the file lines
    chunk_content = result.page_content
    assert chunk_content in file_content, "Chunk content not found in mapped file lines"

    logger.info(
        f"Line mapping verified: {metadata['file']} (lines {start_line}-{end_line})"
    )


def test_ingest_repository_error_handling():
    """Test error handling for invalid repository URLs."""
    logger.info("Starting error handling test")

    # Test with invalid repository URL
    invalid_url = "https://github.com/nonexistent/repository"

    # Import the exception that will be raised
    from github.GithubException import UnknownObjectException

    with pytest.raises(UnknownObjectException):
        ingest_repository(invalid_url)

    logger.info("Error handling test completed")


@pytest.mark.integration
def test_ingest_repository_full_pipeline(clean_data_dir, unique_collection_name):
    """Full integration test of the ingestion pipeline."""
    logger.info("Starting full pipeline integration test")

    # Test the complete pipeline
    vectorstore = ingest_repository(
        repo_url=TEST_REPO_URL,
        chunk_size=1024,
        chunk_overlap=128,
        batch_size=64,
        collection_name=unique_collection_name,
        persist_directory=None,  # Use in-memory storage
    )

    # Verify vector store properties
    assert vectorstore is not None
    assert isinstance(vectorstore, Chroma)

    # Test search functionality
    results = vectorstore.similarity_search("Hello World", k=3)
    assert len(results) > 0

    # Verify result quality
    for result in results:
        assert len(result.page_content) > 0
        assert result.metadata["file"]
        assert result.metadata["start_line"] >= 0
        assert result.metadata["end_line"] >= result.metadata["start_line"]

    logger.info("Full pipeline integration test completed successfully")


if __name__ == "__main__":
    # Run tests manually if needed
    pytest.main([__file__, "-v"])
