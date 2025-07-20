"""Unit tests for markdown to plain text conversion functionality."""

from app.preprocess.markdown_cleaner import markdown_to_plain_text


class TestMarkdownToPlainText:
    """Test markdown to plain text conversion functionality."""

    def test_convert_headers(self):
        """Test converting markdown headers to plain text."""
        content = """# Header 1
## Header 2
### Header 3
#### Header 4"""
        result = markdown_to_plain_text(content)
        expected = "Header 1\nHeader 2\nHeader 3\nHeader 4\n"
        assert result == expected

    def test_convert_links(self):
        """Test converting markdown links to plain text."""
        content = "Check out [GitHub](https://github.com) for more info."
        result = markdown_to_plain_text(content)
        expected = "Check out GitHub for more info.\n"
        assert result == expected

    def test_convert_bold_and_italic(self):
        """Test converting bold and italic text to plain text."""
        content = "This is **bold** and this is *italic* text."
        result = markdown_to_plain_text(content)
        expected = "This is bold and this is italic text.\n"
        assert result == expected

    def test_convert_lists(self):
        """Test converting markdown lists to plain text."""
        content = """- Item 1
- Item 2
  - Subitem 2.1
  - Subitem 2.2
- Item 3

1. Numbered item 1
2. Numbered item 2"""
        result = markdown_to_plain_text(content)
        expected = "\nItem 1\nItem 2\n\nSubitem 2.1\nSubitem 2.2\n\n\nItem 3\n\n\nNumbered item 1\nNumbered item 2\n\n"
        assert result == expected

    def test_handle_line_breaks(self):
        """Test handling line breaks in markdown."""
        content = "Line 1\nLine 2\n\nLine 3"
        result = markdown_to_plain_text(content)
        expected = "Line 1\nLine 2\nLine 3\n"
        assert result == expected
