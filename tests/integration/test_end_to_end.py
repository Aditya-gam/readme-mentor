import logging
import re
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List

import pytest

from app.backend import generate_answer
from app.embeddings.ingest import ingest_repository
from tests.integration.test_performance_config import (
    PerformanceConfig,
    log_performance_metrics,
)

# Configure logging for the test
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def setup_pytest_repo_ingestion():
    """
    Fixture to set up pytest repository ingestion for end-to-end testing.

    This fixture:
    1. Cleans up any existing data for the pytest repository
    2. Ingests the pytest-dev/pytest repository (README and docs)
    3. Provides the repo_id to test functions
    4. Cleans up after tests are complete

    Returns:
        str: Repository ID in format 'owner_repo'
    """
    repo_url = "https://github.com/pytest-dev/pytest"
    repo_owner = "pytest-dev"
    repo_name = "pytest"
    repo_id = f"{repo_owner}_{repo_name}"
    persist_directory = Path("data") / repo_id / "chroma"
    raw_data_directory = Path("data") / repo_id / "raw"

    logger.info(f"Setting up end-to-end test for repository: {repo_url}")
    logger.info(f"Repository ID: {repo_id}")
    logger.info(f"Persist directory: {persist_directory}")
    logger.info(f"Raw data directory: {raw_data_directory}")

    # Clean up any existing data before ingestion
    if persist_directory.exists():
        logger.info(f"Cleaning up existing persist directory: {persist_directory}")
        shutil.rmtree(persist_directory)
    if raw_data_directory.exists():
        logger.info(f"Cleaning up existing raw data directory: {raw_data_directory}")
        shutil.rmtree(raw_data_directory)

    # Ingest the repository
    ingestion_start_time = time.perf_counter()
    logger.info(f"Starting ingestion of {repo_url}...")

    try:
        ingest_repository(repo_url=repo_url, persist_directory=str(persist_directory))
        ingestion_time = time.perf_counter() - ingestion_start_time
        logger.info(f"Ingestion completed successfully in {ingestion_time:.2f} seconds")

        # Verify ingestion created the expected directories
        assert persist_directory.exists(), (
            f"Persist directory not created: {persist_directory}"
        )
        assert raw_data_directory.exists(), (
            f"Raw data directory not created: {raw_data_directory}"
        )

        # Log some basic stats about the ingested data
        if raw_data_directory.exists():
            raw_files = list(raw_data_directory.rglob("*"))
            raw_files = [f for f in raw_files if f.is_file()]
            logger.info(f"Ingested {len(raw_files)} files to raw data directory")

    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise

    yield repo_id  # Provide the repo_id to the test function

    # Teardown: Clean up after tests are done
    logger.info(f"Cleaning up test data for {repo_id}...")
    if persist_directory.exists():
        shutil.rmtree(persist_directory)
        logger.info(f"Removed persist directory: {persist_directory}")
    if raw_data_directory.exists():
        shutil.rmtree(raw_data_directory)
        logger.info(f"Removed raw data directory: {raw_data_directory}")
    logger.info(f"Cleanup complete for {repo_id}.")


@pytest.mark.integration
@pytest.mark.performance
def test_e2e_pytest_qa(setup_pytest_repo_ingestion):
    """
    End-to-end test that validates the complete QA pipeline.

    This test:
    1. Uses the ingested pytest repository from the fixture
    2. Queries the system with a specific question about running test files
    3. Validates the response structure and content
    4. Verifies citation correctness by cross-checking source text
    5. Ensures the answer is grounded in the source documentation
    6. **NEW**: Validates performance - ensures response completes within 3 seconds
    7. **NEW**: Provides detailed performance breakdown for CI monitoring

    The test requires either:
    - A running Ollama instance with the 'llama3:8b' model, or
    - A valid OPENAI_API_KEY set in the environment

    Args:
        setup_pytest_repo_ingestion: Fixture that provides the repo_id
    """
    repo_id = setup_pytest_repo_ingestion
    query = "How do I run a specific test file?"
    history = []

    logger.info(f"Starting end-to-end QA test for repository: {repo_id}")
    logger.info(f"Query: {query}")
    logger.info(f"History length: {len(history)}")

    # Get performance configuration before test
    config = PerformanceConfig.get_config_summary()
    logger.info(f"Performance environment: {config['environment']}")
    logger.info(
        f"E2E QA threshold: {PerformanceConfig.get_threshold('e2e_qa_response')}ms"
    )
    logger.info(f"Strict enforcement: {config['strict_enforcement']}")

    # Execute the QA query with performance measurement
    query_start_time = time.perf_counter()
    try:
        result = generate_answer(query=query, repo_id=repo_id, history=history)
        query_time = time.perf_counter() - query_start_time
        logger.info(f"Query completed in {query_time:.2f} seconds")
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise

    # Validate response structure
    logger.info("Validating response structure...")
    _validate_response_structure(result)

    # Log response details
    logger.info(f"Answer length: {len(result['answer'])} characters")
    logger.info(f"Number of citations: {len(result['citations'])}")
    logger.info(f"Reported latency: {result['latency_ms']:.2f} ms")
    logger.info(f"Actual query time: {query_time * 1000:.2f} ms")

    # **PERFORMANCE VALIDATION**: Use environment-specific thresholds
    actual_latency_ms = result["latency_ms"]

    # Log performance metrics using the new configuration system
    log_performance_metrics("e2e_qa_response", actual_latency_ms)

    # **ENHANCED PERFORMANCE MONITORING**: Detailed breakdown for CI
    threshold = PerformanceConfig.get_threshold("e2e_qa_response")
    performance_status = "PASS" if actual_latency_ms < threshold else "FAIL"

    # CI-friendly performance output
    print("PERFORMANCE_BREAKDOWN:")
    print("  Operation: E2E QA Response")
    print(f"  Environment: {config['environment']}")
    print(f"  Actual Latency: {actual_latency_ms:.2f}ms")
    print(f"  Threshold: {threshold}ms")
    print(f"  Status: {performance_status}")
    print(f"  Strict Enforcement: {config['strict_enforcement']}")
    print(f"  Answer Length: {len(result['answer'])} chars")
    print(f"  Citations Count: {len(result['citations'])}")

    # Performance assertion based on configuration
    if actual_latency_ms >= threshold:
        if PerformanceConfig.should_strict_enforce():
            pytest.fail(
                f"Performance threshold exceeded: {actual_latency_ms:.2f}ms >= "
                f"{threshold}ms (Environment: {config['environment']})"
            )
        else:
            logger.warning(
                f"Performance threshold exceeded but not strictly enforced: "
                f"{actual_latency_ms:.2f}ms >= {threshold}ms (Environment: {config['environment']})"
            )

    # Validate answer content
    logger.info("Validating answer content...")
    _validate_answer_content(result["answer"])

    # Validate citations
    logger.info("Validating citations...")
    _validate_citations(result["citations"], repo_id)

    logger.info("End-to-end test completed successfully!")
    print(f"PERFORMANCE_TEST_COMPLETED: {performance_status}")


@pytest.mark.integration
def test_vector_store_retrieval(setup_pytest_repo_ingestion):
    """
    Test that the vector store can retrieve relevant documents for the test query.

    This test verifies that:
    1. The vector store was created correctly
    2. It can retrieve documents for the test query
    3. The retrieved documents contain relevant content

    Args:
        setup_pytest_repo_ingestion: Fixture that provides the repo_id
    """
    repo_id = setup_pytest_repo_ingestion
    query = "How do I run a specific test file?"

    logger.info(f"Testing vector store retrieval for repository: {repo_id}")
    logger.info(f"Query: {query}")

    # Load the vector store
    from app.embeddings.ingest import get_embedding_model, get_vector_store

    embedding_model = get_embedding_model()
    vector_store = get_vector_store(repo_id, embedding_model)

    # Test basic similarity search with performance measurement
    logger.info("Testing basic similarity search...")
    search_start_time = time.perf_counter()
    results = vector_store.similarity_search(query, k=4)
    search_time = time.perf_counter() - search_start_time
    logger.info(
        f"Basic similarity search returned {len(results)} documents in {search_time * 1000:.2f}ms"
    )

    for i, doc in enumerate(results):
        logger.info(f"Document {i}: {doc.metadata}")
        logger.info(f"Content preview: {doc.page_content[:200]}...")

    # Test MMR search (same as used in the chain) with performance measurement
    logger.info("Testing MMR search...")
    mmr_start_time = time.perf_counter()
    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 4})
    mmr_results = retriever.invoke(query)
    mmr_time = time.perf_counter() - mmr_start_time
    logger.info(
        f"MMR search returned {len(mmr_results)} documents in {mmr_time * 1000:.2f}ms"
    )

    # Log performance metrics for vector search
    log_performance_metrics("vector_search", mmr_time * 1000)

    for i, doc in enumerate(mmr_results):
        logger.info(f"MMR Document {i}: {doc.metadata}")
        logger.info(f"Content preview: {doc.page_content[:200]}...")

    # Verify we got some results
    assert len(results) > 0, "Basic similarity search should return documents"
    assert len(mmr_results) > 0, "MMR search should return documents"

    # Check that at least one document contains relevant content
    relevant_content_found = False
    for doc in mmr_results:
        content_lower = doc.page_content.lower()
        if any(
            keyword in content_lower for keyword in ["pytest", "test", "run", "command"]
        ):
            relevant_content_found = True
            logger.info(f"Found relevant content in document: {doc.metadata}")
            break

    assert relevant_content_found, "Should find documents with relevant content"

    logger.info("Vector store retrieval test completed successfully!")


def _validate_response_structure(result: Dict[str, Any]) -> None:
    """
    Validate the structure of the response from generate_answer.

    Args:
        result: The response dictionary from generate_answer

    Raises:
        AssertionError: If the response structure is invalid
    """
    # Check required keys exist
    required_keys = ["answer", "citations", "latency_ms"]
    for key in required_keys:
        assert key in result, f"Missing required key in response: {key}"

    # Validate answer
    assert isinstance(result["answer"], str), "Answer must be a string"
    assert len(result["answer"]) > 0, "Answer cannot be empty"

    # Validate citations
    assert isinstance(result["citations"], list), "Citations must be a list"
    assert len(result["citations"]) > 0, "Must have at least one citation"

    # Validate latency
    assert isinstance(result["latency_ms"], (int, float)), "Latency must be numeric"
    assert result["latency_ms"] > 0, "Latency must be positive"


def _validate_answer_content(answer: str) -> None:
    """
    Validate that the answer contains relevant content about running tests.

    Args:
        answer: The answer string from the response

    Raises:
        AssertionError: If the answer content is invalid
    """
    answer_lower = answer.lower()

    # Check for relevant keywords
    relevant_keywords = ["pytest", "test", "run", "file", "command"]
    found_keywords = [kw for kw in relevant_keywords if kw in answer_lower]

    logger.info(f"Found relevant keywords in answer: {found_keywords}")
    assert len(found_keywords) >= 2, (
        f"Answer should contain at least 2 relevant keywords. Found: {found_keywords}"
    )

    # Check for command-like patterns
    command_patterns = [
        r"pytest\s+.*?\.py",  # pytest path/to/file.py
        r"pytest\s+.*?::.*?",  # pytest path/to/file.py::test_function
        r"python\s+-m\s+pytest",  # python -m pytest
    ]

    found_patterns = []
    for pattern in command_patterns:
        if re.search(pattern, answer_lower):
            found_patterns.append(pattern)

    logger.info(f"Found command patterns in answer: {found_patterns}")
    assert len(found_patterns) > 0, "Answer should contain at least one command pattern"


def _validate_citations(citations: List[Dict[str, Any]], repo_id: str) -> None:
    """
    Validate the correctness of citations by cross-checking source text.

    Args:
        citations: List of citation dictionaries
        repo_id: Repository identifier

    Raises:
        AssertionError: If citations are invalid or don't match source content
    """
    logger.info(f"Validating {len(citations)} citations...")

    for i, citation in enumerate(citations):
        logger.info(f"Validating citation {i + 1}/{len(citations)}: {citation}")

        # Validate citation structure
        required_citation_keys = ["file", "start_line", "end_line"]
        for key in required_citation_keys:
            assert key in citation, f"Citation missing required key: {key}"

        file_name = citation["file"]
        start_line = citation["start_line"]
        end_line = citation["end_line"]

        # Validate line numbers
        assert isinstance(start_line, int) and start_line > 0, (
            f"Invalid start_line: {start_line}"
        )
        assert isinstance(end_line, int) and end_line > 0, (
            f"Invalid end_line: {end_line}"
        )
        assert start_line <= end_line, (
            f"start_line ({start_line}) must be <= end_line ({end_line})"
        )

        # Construct the path to the raw cited file
        # Handle both relative and absolute paths in the file_name
        if file_name.startswith(f"data/{repo_id}/raw/"):
            # File name already contains the full path
            raw_file_path = Path(file_name)
        else:
            # File name is relative, construct the full path
            raw_file_path = Path("data") / repo_id / "raw" / file_name
        assert raw_file_path.exists(), f"Cited file not found: {raw_file_path}"

        # Read and validate the cited content
        try:
            with open(raw_file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            # Validate line range
            assert start_line <= len(lines), (
                f"start_line ({start_line}) exceeds file length ({len(lines)})"
            )
            assert end_line <= len(lines), (
                f"end_line ({end_line}) exceeds file length ({len(lines)})"
            )

            # Extract the cited content (adjusting for 0-based indexing)
            cited_content_lines = lines[start_line - 1 : end_line]
            cited_content = "".join(cited_content_lines)

            logger.info(f"Citation {i + 1} from {file_name} L{start_line}-{end_line}:")
            logger.info(f"Cited content preview: {cited_content[:200]}...")

            # Validate that cited content contains relevant information
            _validate_cited_content(cited_content, file_name, i + 1)

        except UnicodeDecodeError as e:
            logger.warning(f"Could not decode file {raw_file_path}: {e}")
            # For binary files, we'll skip content validation but still validate structure
            continue
        except Exception as e:
            logger.error(f"Error reading cited file {raw_file_path}: {e}")
            raise


def _validate_cited_content(
    cited_content: str, file_name: str, citation_num: int
) -> None:
    """
    Validate that the cited content contains relevant information about pytest.

    Args:
        cited_content: The content from the cited lines
        file_name: Name of the file being cited
        citation_num: Citation number for logging

    Raises:
        AssertionError: If the cited content is not relevant
    """
    content_lower = cited_content.lower()

    # Check for pytest-related content or test-related content
    pytest_related = "pytest" in content_lower
    test_related = any(kw in content_lower for kw in ["test", "assert", "sample"])

    # At least one citation should contain pytest or be test-related
    if citation_num == 1:  # First citation should be most relevant
        assert pytest_related or test_related, (
            "Primary citation should contain 'pytest' or test-related content"
        )

    # Check for test-related keywords
    test_keywords = ["run", "test", "file", "command", "execute", "assert", "sample"]
    found_keywords = [kw for kw in test_keywords if kw in content_lower]

    logger.info(f"Citation {citation_num} contains keywords: {found_keywords}")

    # At least one citation should contain test-related keywords
    if citation_num == 1:  # First citation should be most relevant
        assert len(found_keywords) >= 1, (
            "Primary citation should contain at least one test-related keyword"
        )

    # Check for command-like patterns in the cited content
    command_patterns = [
        r"pytest\s+.*?\.py",  # pytest path/to/file.py
        r"pytest\s+.*?::.*?",  # pytest path/to/file.py::test_function
        r"python\s+-m\s+pytest",  # python -m pytest
    ]

    found_patterns = []
    for pattern in command_patterns:
        if re.search(pattern, content_lower):
            found_patterns.append(pattern)

    logger.info(f"Citation {citation_num} contains command patterns: {found_patterns}")

    # At least one citation should contain a command pattern
    if citation_num == 1:  # First citation should be most relevant
        assert len(found_patterns) > 0, (
            "Primary citation should contain a command pattern"
        )
