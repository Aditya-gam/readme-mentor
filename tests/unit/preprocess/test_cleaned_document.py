"""Unit tests for CleanedDocument functionality."""

import pytest

from app.preprocess.markdown_cleaner import CleanedDocument


class TestCleanedDocument:
    """Test CleanedDocument functionality."""

    def test_cleaned_document_creation(self):
        """Test creating a CleanedDocument instance."""
        cleaned_text = "Cleaned text\nwith multiple\nlines"
        metadata = {"test": "value"}

        doc = CleanedDocument(cleaned_text, None, [], metadata)

        assert doc.text == cleaned_text
        assert doc.path is None
        assert doc.metadata == metadata

    def test_get_line_range(self):
        """Test getting line range for a text segment."""
        cleaned_text = "Line 1\nLine 2\nLine 3\nLine 4"
        metadata = {"test": "value"}

        doc = CleanedDocument(cleaned_text, None, [], metadata)

        # Test line range for a segment spanning multiple lines
        start_line, end_line = doc.get_line_range(0, 15)  # Covers first two lines
        assert start_line == 0
        assert end_line == 2

    def test_get_line_content(self):
        """Test getting content for a specific line."""
        cleaned_text = "Line 1\nLine 2\nLine 3\nLine 4"
        metadata = {"test": "value"}

        doc = CleanedDocument(cleaned_text, None, [], metadata)

        # Test getting content for line 1 (0-indexed)
        content = doc.get_line_content(1)
        assert content == "Line 2"

    def test_empty_text(self):
        """Test CleanedDocument with empty text."""
        metadata = {"test": "value"}
        doc = CleanedDocument("", None, [], metadata)

        # For empty text, we can't get a line range with start_char == end_char
        # This should raise a ValueError
        with pytest.raises(ValueError, match="start_char must be less than end_char"):
            doc.get_line_range(0, 0)

        content = doc.get_line_content(0)
        assert content == ""

    def test_text_without_newlines(self):
        """Test CleanedDocument with text without newlines."""
        cleaned_text = "Single line text"
        metadata = {"test": "value"}

        doc = CleanedDocument(cleaned_text, None, [], metadata)

        start_line, end_line = doc.get_line_range(0, 15)
        assert start_line == 0
        assert end_line == 0

        content = doc.get_line_content(0)
        assert content == "Single line text"

    def test_get_line_content_with_empty_lines(self):
        """Test getting line content with empty lines."""
        cleaned_text = "Line 1\n\nLine 3"
        metadata = {"test": "value"}

        doc = CleanedDocument(cleaned_text, None, [], metadata)

        # Test getting content for specific lines
        content = doc.get_line_content(0)
        assert content == "Line 1"
        content = doc.get_line_content(1)
        assert content == ""
        content = doc.get_line_content(2)
        assert content == "Line 3"
