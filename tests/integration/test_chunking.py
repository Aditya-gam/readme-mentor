"""Unit tests for text chunking functionality."""

import logging
from pathlib import Path
from unittest.mock import Mock, patch

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestChunkTextWithLineMapping:
    """Test the _chunk_text_with_line_mapping function."""

    @patch("app.embeddings.ingest.LineOffsetMapper")
    @patch("app.embeddings.ingest.RecursiveCharacterTextSplitter")
    def test_chunk_text_with_line_mapping(self, mock_splitter_class, mock_mapper_class):
        """Test chunking text with line mapping."""
        # Setup mocks
        mock_mapper = Mock()
        mock_mapper_class.return_value = mock_mapper
        mock_mapper.get_line_range.return_value = (1, 5)

        mock_splitter = Mock()
        mock_splitter_class.return_value = mock_splitter
        mock_splitter.split_text.return_value = ["chunk1", "chunk2"]

        # Test data
        text = "This is a test document\nwith multiple lines\nfor chunking"
        file_path = Path("test.md")

        # Call function
        from app.embeddings.ingest import (
            _chunk_text_with_line_mapping,
        )

        result = _chunk_text_with_line_mapping(
            text, file_path, chunk_size=10, chunk_overlap=2
        )

        # Verify mocks were called correctly
        mock_mapper_class.assert_called_once_with(text)
        mock_splitter_class.assert_called_once_with(
            chunk_size=10,
            chunk_overlap=2,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )
        mock_splitter.split_text.assert_called_once_with(text)

        # Verify results
        assert len(result) == 2
        assert all(hasattr(doc, "page_content") for doc in result)

        # Check first document
        doc1 = result[0]
        assert doc1.page_content == "chunk1"
        assert doc1.metadata["file"] == str(file_path)
        assert doc1.metadata["start_line"] == 1
        assert doc1.metadata["end_line"] == 5
        assert "chunk_start" in doc1.metadata
        assert "chunk_end" in doc1.metadata

    @patch("app.embeddings.ingest.LineOffsetMapper")
    @patch("app.embeddings.ingest.RecursiveCharacterTextSplitter")
    def test_chunk_text_with_defaults(self, mock_splitter_class, mock_mapper_class):
        """Test chunking text with default parameters."""
        # Setup mocks
        mock_mapper = Mock()
        mock_mapper_class.return_value = mock_mapper
        mock_mapper.get_line_range.return_value = (1, 3)

        mock_splitter = Mock()
        mock_splitter_class.return_value = mock_splitter
        mock_splitter.split_text.return_value = ["single chunk"]

        # Test data
        text = "Simple text"
        file_path = Path("simple.md")

        # Call function with defaults
        from app.embeddings.ingest import (
            DEFAULT_CHUNK_OVERLAP,
            DEFAULT_CHUNK_SIZE,
            _chunk_text_with_line_mapping,
        )

        result = _chunk_text_with_line_mapping(text, file_path)

        # Verify default parameters were used
        mock_splitter_class.assert_called_once_with(
            chunk_size=DEFAULT_CHUNK_SIZE,
            chunk_overlap=DEFAULT_CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )

        assert len(result) == 1
        assert result[0].page_content == "single chunk"


class TestProcessFileForChunking:
    """Test the _process_file_for_chunking function."""

    @patch("app.embeddings.ingest.clean_markdown_file")
    @patch("app.embeddings.ingest._chunk_text_with_line_mapping")
    def test_process_file_success(self, mock_chunk, mock_clean):
        """Test successful file processing."""
        # Setup mocks
        mock_cleaned_doc = Mock()
        mock_cleaned_doc.text = "cleaned content"
        mock_clean.return_value = mock_cleaned_doc

        mock_documents = [Mock(page_content="chunk1"), Mock(page_content="chunk2")]
        mock_chunk.return_value = mock_documents

        # Test data
        file_path = Path("test.md")

        # Call function
        from app.embeddings.ingest import _process_file_for_chunking

        result = _process_file_for_chunking(file_path)

        # Verify mocks were called
        mock_clean.assert_called_once_with(file_path, include_code=False)
        mock_chunk.assert_called_once_with("cleaned content", file_path)

        # Verify result
        assert result == mock_documents

    @patch("app.embeddings.ingest.clean_markdown_file")
    def test_process_file_exception(self, mock_clean):
        """Test file processing with exception."""
        # Setup mock to raise exception
        mock_clean.side_effect = Exception("File not found")

        # Test data
        file_path = Path("nonexistent.md")

        # Call function
        from app.embeddings.ingest import _process_file_for_chunking

        result = _process_file_for_chunking(file_path)

        # Verify result is empty list on exception
        assert result == []
