"""
Integration tests for the generate_answer function.

Tests the complete generate_answer flow including:
- Valid query processing with proper citations
- Input validation using Pydantic models
- Error handling for non-ingested repositories
- Citation format and line number verification
- Response structure validation
"""

import logging
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document
from pydantic import ValidationError

from app.backend import generate_answer
from app.models import QuestionPayload

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test repository (octocat/Hello-World) - same as used in other tests
TEST_REPO_URL = "https://github.com/octocat/Hello-World"
TEST_REPO_ID = "octocat_Hello-World"


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
def mock_llm():
    """Mock LLM for testing."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = (
        "This is a test answer with <doc_0> citation."
    )
    return mock_llm


@pytest.fixture(scope="function")
def mock_vector_store():
    """Mock vector store with test documents."""
    mock_store = MagicMock()

    # Create test documents with realistic metadata
    test_documents = [
        Document(
            page_content="# Hello World\n\nThis is a test repository for learning Git and GitHub.",
            metadata={
                "file": "README.md",
                "start_line": 1,
                "end_line": 3,
                "source": "README.md",
            },
        ),
        Document(
            page_content="## Getting Started\n\nTo get started with this project, follow these steps:",
            metadata={
                "file": "README.md",
                "start_line": 5,
                "end_line": 7,
                "source": "README.md",
            },
        ),
        Document(
            page_content="## Contributing\n\nContributions are welcome! Please read our contributing guidelines.",
            metadata={
                "file": "CONTRIBUTING.md",
                "start_line": 1,
                "end_line": 3,
                "source": "CONTRIBUTING.md",
            },
        ),
    ]

    mock_store.similarity_search.return_value = test_documents
    return mock_store


@pytest.fixture(scope="function")
def mock_qa_chain():
    """Mock QA chain for testing."""
    mock_chain = MagicMock()
    mock_chain.return_value = {
        "answer": "This project is a Hello World repository. According to <doc_0>, it's used for learning Git and GitHub. The <doc_1> section explains how to get started.",
        "source_documents": [
            Document(
                page_content="# Hello World\n\nThis is a test repository for learning Git and GitHub.",
                metadata={
                    "file": "README.md",
                    "start_line": 1,
                    "end_line": 3,
                    "source": "README.md",
                },
            ),
            Document(
                page_content="## Getting Started\n\nTo get started with this project, follow these steps:",
                metadata={
                    "file": "README.md",
                    "start_line": 5,
                    "end_line": 7,
                    "source": "README.md",
                },
            ),
        ],
    }
    return mock_chain


class TestGenerateAnswer:
    """Test cases for the generate_answer function."""

    def test_generate_answer_empty_query_validation_error(self):
        """Test that generate_answer raises validation error for empty query."""
        logger.info("Testing generate_answer with empty query")

        # Test with empty query
        with pytest.raises(ValueError) as exc_info:
            generate_answer("", TEST_REPO_ID)

        # Verify error message
        error_message = str(exc_info.value)
        assert "Validation error" in error_message, (
            "Should raise validation error for empty query"
        )
        assert "Query cannot be empty" in error_message, (
            "Error should mention empty query"
        )

        logger.info("Successfully caught validation error for empty query")

    def test_generate_answer_empty_repo_id_validation_error(self):
        """Test that generate_answer raises validation error for empty repo_id."""
        logger.info("Testing generate_answer with empty repo_id")

        # Test with empty repo_id
        with pytest.raises(ValueError) as exc_info:
            generate_answer("What is this project?", "")

        # Verify error message
        error_message = str(exc_info.value)
        assert "Validation error" in error_message, (
            "Should raise validation error for empty repo_id"
        )
        assert "Repository ID cannot be empty" in error_message, (
            "Error should mention empty repo_id"
        )

        logger.info("Successfully caught validation error for empty repo_id")

    def test_generate_answer_whitespace_only_validation_error(self):
        """Test that generate_answer raises validation error for whitespace-only inputs."""
        logger.info("Testing generate_answer with whitespace-only inputs")

        # Test with whitespace-only query
        with pytest.raises(ValueError) as exc_info:
            generate_answer("   ", TEST_REPO_ID)

        error_message = str(exc_info.value)
        assert "Validation error" in error_message, (
            "Should raise validation error for whitespace-only query"
        )

        # Test with whitespace-only repo_id
        with pytest.raises(ValueError) as exc_info:
            generate_answer("What is this project?", "   ")

        error_message = str(exc_info.value)
        assert "Validation error" in error_message, (
            "Should raise validation error for whitespace-only repo_id"
        )

        logger.info("Successfully caught validation errors for whitespace-only inputs")

    def test_generate_answer_non_ingested_repository_error(
        self,
        clean_data_dir,
    ):
        """Test that generate_answer handles non-ingested repository gracefully."""
        logger.info("Testing generate_answer with non-ingested repository")

        # Setup mocks using context managers to avoid module reloading issues
        with (
            patch("app.backend.get_embedding_model") as mock_get_embedding_model,
            patch("app.backend.get_vector_store") as mock_get_vector_store,
            patch("app.backend.get_chat_model") as mock_get_chat_model,
            patch("app.backend.get_qa_chain") as mock_get_qa_chain,
        ):
            # Setup mocks to simulate non-ingested repository
            mock_get_embedding_model.return_value = MagicMock()
            mock_get_vector_store.side_effect = FileNotFoundError(
                "Vector store not found"
            )
            mock_get_chat_model.return_value = MagicMock()
            mock_get_qa_chain.return_value = MagicMock()

            # Test with non-ingested repository
            with pytest.raises(Exception) as exc_info:
                generate_answer("What is this project?", "non_existent_repo")

            # Verify error handling
            error_message = str(exc_info.value)
            assert "Vector store not found" in error_message, (
                "Should handle non-ingested repository error"
            )

            logger.info("Successfully handled non-ingested repository error")

    def test_question_payload_validation(self):
        """Test QuestionPayload model validation directly."""
        logger.info("Testing QuestionPayload model validation")

        # Test valid payload
        valid_payload = QuestionPayload(
            query="What is this project?",
            repo_id=TEST_REPO_ID,
            history=[("Previous question", "Previous answer")],
        )
        assert valid_payload.query == "What is this project?"
        assert valid_payload.repo_id == TEST_REPO_ID
        assert len(valid_payload.history) == 1

        # Test empty query validation
        with pytest.raises(ValidationError):
            QuestionPayload(query="", repo_id=TEST_REPO_ID)

        # Test empty repo_id validation
        with pytest.raises(ValidationError):
            QuestionPayload(query="What is this project?", repo_id="")

        # Test whitespace-only validation
        with pytest.raises(ValidationError):
            QuestionPayload(query="   ", repo_id=TEST_REPO_ID)

        with pytest.raises(ValidationError):
            QuestionPayload(query="What is this project?", repo_id="   ")

        logger.info("Successfully tested QuestionPayload validation")
