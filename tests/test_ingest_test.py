"""Integration tests for the ingestion pipeline.

This module tests the complete ingestion pipeline including text chunking,
embedding generation, and vector store storage.
"""

import logging
import shutil
import subprocess
import sys
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

# Test repository with minimal structure (no docs folder)
MINIMAL_REPO_URL = "https://github.com/octocat/test-repo-1"


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
    all_results = vectorstore.similarity_search(
        "", k=100)  # Get up to 100 documents

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
        assert file_path.exists(
        ), f"Referenced file does not exist: {metadata['file']}"

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
        assert file_path.exists(
        ), f"Referenced file does not exist: {metadata['file']}"

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


def test_ingest_repository_error_handling():
    """Test error handling for invalid repository URLs."""
    logger.info("Starting error handling test")

    # Test with invalid repository URL
    invalid_repo_url = "https://github.com/nonexistent/repo"

    with pytest.raises((Exception, ValueError)):
        ingest_repository(invalid_repo_url)

    logger.info("Error handling test completed")


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


def test_cli_help():
    """Test CLI help functionality."""
    logger.info("Starting CLI help test")

    # Test help command
    result = subprocess.run(
        [sys.executable, "scripts/ingest_cli.py", "--help"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
    )

    assert result.returncode == 0, f"CLI help failed: {result.stderr}"
    assert "GitHub repository URL to ingest" in result.stdout
    assert "--chunk-size" in result.stdout
    assert "--file-glob" in result.stdout

    logger.info("CLI help test completed successfully")


def test_cli_basic_ingestion(clean_data_dir):
    """Test basic CLI ingestion functionality."""
    logger.info("Starting CLI basic ingestion test")

    # Test basic ingestion command
    result = subprocess.run(
        [
            sys.executable,
            "scripts/ingest_cli.py",
            TEST_REPO_URL,
            "--chunk-size",
            "512",
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
    )

    # CLI should complete successfully
    assert result.returncode == 0, f"CLI ingestion failed: {result.stderr}"
    assert "Ingestion completed successfully" in result.stdout
    assert "Collection name:" in result.stdout

    logger.info("CLI basic ingestion test completed successfully")


def test_cli_invalid_repo():
    """Test CLI error handling for invalid repository."""
    logger.info("Starting CLI invalid repo test")

    # Test with invalid repository URL
    result = subprocess.run(
        [
            sys.executable,
            "scripts/ingest_cli.py",
            "https://github.com/nonexistent/repo",
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
    )

    # CLI should fail gracefully
    assert result.returncode == 1, "CLI should fail for invalid repo"
    assert "Ingestion failed" in result.stdout

    logger.info("CLI invalid repo test completed successfully")
