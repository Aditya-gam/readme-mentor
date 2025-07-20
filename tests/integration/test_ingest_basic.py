"""Basic ingestion tests for the ingestion pipeline."""

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


@pytest.mark.integration
def test_ingest_repository_full_pipeline(clean_data_dir, unique_collection_name):
    """Test the complete ingestion pipeline end-to-end."""
    logger.info("Starting full pipeline integration test")

    # Run ingestion with in-memory storage for testing
    vectorstore = ingest_repository(
        TEST_REPO_URL,
        collection_name=unique_collection_name,
        persist_directory=None,  # Use in-memory storage
    )

    # Verify vectorstore is created and functional
    assert vectorstore is not None
    assert isinstance(vectorstore, Chroma)

    # Test search functionality
    results = vectorstore.similarity_search("Hello World", k=3)
    assert len(results) > 0, "No search results returned"

    # Verify result structure and content
    for result in results:
        assert isinstance(result, Document)
        assert result.page_content.strip(), "Empty page content"
        assert "file" in result.metadata, "Missing file metadata"
        assert "start_line" in result.metadata, "Missing start_line metadata"
        assert "end_line" in result.metadata, "Missing end_line metadata"

    logger.info("Full pipeline integration test completed successfully")


def test_ingest_repository_no_docs_folder(clean_data_dir, unique_collection_name):
    """Test ingestion of repository with no docs folder (edge case)."""
    logger.info("Starting no-docs-folder edge case test")

    # Test with custom file glob that only looks for README files
    custom_file_glob = ("README*",)

    try:
        vectorstore = ingest_repository(
            TEST_REPO_URL,
            file_glob=custom_file_glob,
            collection_name=unique_collection_name,
            persist_directory=None,  # Use in-memory storage
        )

        # Verify vectorstore is created
        assert vectorstore is not None
        assert isinstance(vectorstore, Chroma)

        # Test search functionality
        results = vectorstore.similarity_search("Hello", k=3)
        assert len(results) > 0, "No results found with custom file glob"

        logger.info("No-docs-folder edge case test completed successfully")

    except Exception as e:
        # If the test repo doesn't have README files, that's also a valid test case
        logger.info(f"Expected behavior: {e}")
        assert "No files found" in str(e) or "No documents created" in str(e)


def test_ingest_repository_custom_file_patterns(clean_data_dir, unique_collection_name):
    """Test ingestion with custom file patterns."""
    logger.info("Starting custom file patterns test")

    # Test with custom file patterns
    custom_patterns = ("*.md", "*.txt")

    try:
        vectorstore = ingest_repository(
            TEST_REPO_URL,
            file_glob=custom_patterns,
            collection_name=unique_collection_name,
            persist_directory=None,  # Use in-memory storage
        )

        # Verify vectorstore is created
        assert vectorstore is not None
        assert isinstance(vectorstore, Chroma)

        # Test search functionality
        results = vectorstore.similarity_search("test", k=3)
        # Note: May or may not have results depending on repo content
        logger.info(f"Found {len(results)} results with custom patterns")

        logger.info("Custom file patterns test completed successfully")

    except Exception as e:
        # Handle case where no matching files found
        logger.info(f"Expected behavior with custom patterns: {e}")
        assert "No files found" in str(e) or "No documents created" in str(e)


def test_ingest_repository_persistent_storage(clean_data_dir, unique_collection_name):
    """Test ingestion with persistent storage."""
    logger.info("Starting persistent storage test")

    # Create persistent directory
    persist_dir = clean_data_dir / "chroma_test"
    persist_dir.mkdir(parents=True, exist_ok=True)

    # Run ingestion with persistent storage
    vectorstore = ingest_repository(
        TEST_REPO_URL,
        collection_name=unique_collection_name,
        persist_directory=str(persist_dir),
    )

    # Verify vectorstore is created
    assert vectorstore is not None
    assert isinstance(vectorstore, Chroma)

    # Verify persistent directory was created
    assert persist_dir.exists(), "Persistent directory not created"

    # Test search functionality
    results = vectorstore.similarity_search("Hello World", k=3)
    assert len(results) > 0, "No search results returned"

    logger.info("Persistent storage test completed successfully")


def test_ingest_repository_chunking_parameters(clean_data_dir, unique_collection_name):
    """Test ingestion with custom chunking parameters."""
    logger.info("Starting chunking parameters test")

    # Test with smaller chunk size and overlap
    vectorstore = ingest_repository(
        TEST_REPO_URL,
        chunk_size=512,  # Smaller chunks
        chunk_overlap=64,  # Smaller overlap
        collection_name=unique_collection_name,
        persist_directory=None,  # Use in-memory storage
    )

    # Verify vectorstore is created
    assert vectorstore is not None
    assert isinstance(vectorstore, Chroma)

    # Test search functionality
    results = vectorstore.similarity_search("Hello World", k=5)
    assert len(results) > 0, "No search results returned"

    # Verify chunk sizes are reasonable (should be smaller due to smaller chunk_size)
    for result in results:
        content_length = len(result.page_content)
        assert content_length <= 512, f"Chunk too large: {content_length} characters"

    logger.info("Chunking parameters test completed successfully")
