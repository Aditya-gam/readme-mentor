"""Vector search tests for the ingestion pipeline."""

import logging
import shutil
from pathlib import Path

import pytest
from langchain_core.documents import Document

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

    # Verify at least one result contains the search phrase
    found_phrase = False
    for result in results:
        if search_phrase.lower() in result.page_content.lower():
            found_phrase = True
            break

    assert found_phrase, f"Search phrase '{search_phrase}' not found in any results"

    # Verify result content is not empty
    for result in results:
        assert result.page_content.strip(), "Empty page content in result"
        assert len(result.page_content) > 10, "Result content too short"

    logger.info(f"Found {len(results)} results for '{search_phrase}'")
