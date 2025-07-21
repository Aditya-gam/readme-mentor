"""Unit tests for text chunking functionality."""

from pathlib import Path
from unittest.mock import patch

from langchain_core.documents import Document

from app.embeddings.ingest import (
    _chunk_text_with_line_mapping,
    _process_file_for_chunking,
)


class TestChunkTextWithLineMapping:
    """Test the _chunk_text_with_line_mapping function."""

    def test_chunk_text_with_line_mapping(self):
        """Test chunking text with line mapping."""
        # Test data
        text = "This is a test document\nwith multiple lines\nfor chunking"
        file_path = Path("test.md")

        # Call function
        result = _chunk_text_with_line_mapping(
            text, file_path, chunk_size=10, chunk_overlap=2
        )

        # Verify results
        assert len(result) > 0  # Should have at least one chunk
        assert all(isinstance(doc, Document) for doc in result)

        # Check first document
        doc1 = result[0]
        assert doc1.metadata["file"] == str(file_path)
        assert "start_line" in doc1.metadata
        assert "end_line" in doc1.metadata
        assert "chunk_start" in doc1.metadata
        assert "chunk_end" in doc1.metadata

    def test_chunk_text_with_defaults(self):
        """Test chunking text with default parameters."""
        # Test data
        text = "Simple text"
        file_path = Path("simple.md")

        # Call function with defaults
        result = _chunk_text_with_line_mapping(text, file_path)

        # Verify results
        assert len(result) == 1
        assert result[0].page_content == text
        assert result[0].metadata["file"] == str(file_path)
        assert "start_line" in result[0].metadata
        assert "end_line" in result[0].metadata


class TestProcessFileForChunking:
    """Test the _process_file_for_chunking function."""

    def test_process_file_success(self):
        """Test successful file processing."""
        # Create a temporary test file
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("# Test Document\n\nThis is a test markdown file.")
            temp_file = f.name

        try:
            file_path = Path(temp_file)

            # Call function
            result = _process_file_for_chunking(file_path)

            # Verify result
            assert len(result) > 0
            assert all(isinstance(doc, Document) for doc in result)
            assert all(doc.metadata["file"] == str(file_path) for doc in result)

        finally:
            # Clean up
            os.unlink(temp_file)

    @patch("app.embeddings.ingest.clean_markdown_file")
    def test_process_file_exception(self, mock_clean):
        """Test file processing with exception."""
        # Setup mock to raise exception
        mock_clean.side_effect = Exception("File not found")

        # Test data
        file_path = Path("nonexistent.md")

        # Call function
        result = _process_file_for_chunking(file_path)

        # Verify result is empty list on exception
        assert result == []
