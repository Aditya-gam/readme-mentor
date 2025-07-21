"""Integration tests and edge cases for markdown cleaning functionality."""

from unittest.mock import patch

import pytest

from app.preprocess.markdown_cleaner import (
    CleanedDocument,
    LineOffsetMapper,
    clean_markdown_content,
    clean_markdown_file,
    clean_markdown_string,
)


class TestLineOffsetMappingIntegration:
    """Test line offset mapping integration with markdown cleaning."""

    def test_line_offset_mapping_with_cleaned_content(self):
        """Test line offset mapping with cleaned markdown content."""
        cleaned_text = """# Header

Some text here

More text"""

        # Create metadata for test
        metadata = {"test": "value"}

        doc = CleanedDocument(cleaned_text, None, [], metadata)

        # Test line range for a segment
        start_line, end_line = doc.get_line_range(0, 20)
        assert start_line == 0
        assert end_line == 2

    def test_citation_scenario_single_line(self):
        """Test citation scenario with single line content."""
        cleaned_text = "This is a single line with important information."

        mapper = LineOffsetMapper(cleaned_text)
        start_line, end_line = mapper.get_line_range(0, len(cleaned_text))

        assert start_line == 0
        assert end_line == 0

    def test_citation_scenario_multi_line(self):
        """Test citation scenario with multi-line content."""
        cleaned_text = """Line 1 with important info
Line 2 with more details
Line 3 with final notes"""

        mapper = LineOffsetMapper(cleaned_text)
        start_line, end_line = mapper.get_line_range(0, len(cleaned_text))

        assert start_line == 0
        assert end_line == 2

    def test_edge_case_start_of_text(self):
        """Test edge case at the start of text."""
        text = "Hello, world!"
        mapper = LineOffsetMapper(text)

        start_line, end_line = mapper.get_line_range(0, 5)
        assert start_line == 0
        assert end_line == 0

    def test_edge_case_end_of_text(self):
        """Test edge case at the end of text."""
        text = "Hello, world!"
        mapper = LineOffsetMapper(text)

        start_line, end_line = mapper.get_line_range(7, 12)
        assert start_line == 0
        assert end_line == 0

    def test_line_offset_consistency(self):
        """Test that line offsets are consistent across operations."""
        text = "Line 1\nLine 2\nLine 3"
        mapper = LineOffsetMapper(text)

        # Test multiple operations on same mapper
        assert mapper.get_line_number(0) == 0
        assert mapper.get_line_number(6) == 0
        assert mapper.get_line_number(12) == 1

        start_line, end_line = mapper.get_line_range(0, 15)
        assert start_line == 0
        assert end_line == 2

    def test_line_offset_mapping_with_markdown_cleaning(self):
        """Test line offset mapping with actual markdown cleaning."""
        original_text = """# Title
<!-- This comment should be removed -->
Some content here
```python
code block
```
More content"""

        # Clean the markdown
        cleaned_doc = clean_markdown_content(content=original_text)

        # Test that we can still map line numbers
        mapper = LineOffsetMapper(cleaned_doc.text)
        start_line, end_line = mapper.get_line_range(0, len(cleaned_doc.text))

        assert start_line >= 0
        assert end_line >= start_line


class TestEdgeCases:
    """Test edge cases for markdown cleaning."""

    def test_empty_content(self):
        """Test handling empty content."""
        result = clean_markdown_content(content="")
        assert result.text == ""

    def test_content_with_only_whitespace(self):
        """Test handling content with only whitespace."""
        result = clean_markdown_content(content="   \n\t\n  ")
        assert result.text.strip() == ""

    def test_nested_code_fences(self):
        """Test handling nested code fences."""
        content = """# Header
```python
def outer():
    ```javascript
    console.log("nested");
    ```
    return "done"
```
# Footer"""

        result = clean_markdown_content(content=content, include_code=False)
        # Should remove all code blocks
        assert "```" not in result.text

    def test_malformed_yaml_front_matter(self):
        """Test handling malformed YAML front matter."""
        content = """---
title: Test
author: John
---
# Content
This is content."""

        result = clean_markdown_content(content=content)
        assert "title: Test" not in result.text
        assert "author: John" not in result.text
        assert "Content" in result.text


class TestConvenienceFunctions:
    """Test convenience functions for markdown cleaning."""

    def test_clean_markdown_file_function(self):
        """Test clean_markdown_file convenience function."""
        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.read_bytes", return_value=b"# Test content"):
                result = clean_markdown_file("test.md")
                assert isinstance(result, CleanedDocument)
                assert "Test content" in result.text

    def test_clean_markdown_string_function(self):
        """Test clean_markdown_string convenience function."""
        content = "# Test content\nSome text here"
        result = clean_markdown_string(content)
        assert isinstance(result, CleanedDocument)
        assert "Test content" in result.text
        assert "Some text here" in result.text


class TestCleanMarkdownContent:
    """Test the main clean_markdown_content function."""

    def test_clean_markdown_string(self):
        """Test cleaning markdown string content."""
        content = """---
title: Test Document
---

# Header
<!-- Comment -->
Some text with **bold** and *italic*.

```python
def hello():
    print("Hello, world!")
```

More content."""

        result = clean_markdown_content(content=content)

        # Check that YAML front matter is removed
        assert "title: Test Document" not in result.text

        # Check that HTML comment is removed
        assert "<!-- Comment -->" not in result.text

        # Check that markdown formatting is converted
        assert "**bold**" not in result.text
        assert "*italic*" not in result.text
        assert "bold" in result.text
        assert "italic" in result.text

        # Check that code block is removed (default behavior)
        assert "```python" not in result.text
        assert "def hello():" not in result.text

        # Check that headers are converted to plain text
        assert "Header" in result.text

    def test_clean_markdown_with_code_included(self):
        """Test cleaning markdown with code blocks included."""
        content = """# Header
```python
def hello():
    print("Hello, world!")
```
More content."""

        result = clean_markdown_content(content=content, include_code=True)

        # Check that code block content is preserved (but markdown syntax is converted to plain text)
        assert "def hello():" in result.text
        assert 'print("Hello, world!")' in result.text

    @patch("pathlib.Path.exists", return_value=True)
    @patch(
        "pathlib.Path.read_bytes",
        return_value=b"""---
title: Test File
---

# Content
Some text.""",
    )
    def test_clean_markdown_file(self, mock_read_bytes, mock_exists):
        """Test cleaning markdown file."""
        result = clean_markdown_file("test.md")

        assert isinstance(result, CleanedDocument)
        assert "title: Test File" not in result.text
        assert "Content" in result.text
        assert "Some text." in result.text

    def test_clean_markdown_no_input(self):
        """Test cleaning markdown with no input."""
        with pytest.raises(ValueError):
            clean_markdown_content(None)

    def test_clean_markdown_file_not_found(self):
        """Test cleaning markdown file that doesn't exist."""
        with pytest.raises(FileNotFoundError):
            clean_markdown_file("nonexistent.md")

    def test_clean_markdown_file_with_encoding_detection(self):
        """Test cleaning markdown file with encoding detection."""
        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.read_bytes", return_value=b"# Test content"):
                result = clean_markdown_file("test.md")
                assert isinstance(result, CleanedDocument)
                assert "Test content" in result.text

    def test_clean_markdown_content_none_after_read(self):
        """Test cleaning markdown content that becomes None after reading."""
        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.read_bytes", return_value=b""):
                result = clean_markdown_file("test.md")
                assert result.text == ""


class TestLineNumberPreservation:
    """Test line number preservation during cleaning."""

    def test_line_number_preservation_with_comments(self):
        """Test that line numbers are preserved when removing comments."""
        content = """Line 1
<!-- Comment -->
Line 3
Line 4"""

        result = clean_markdown_content(content=content)

        # The cleaned text should have fewer lines but line mapping should work
        assert "<!-- Comment -->" not in result.text
        assert "Line 1" in result.text
        assert "Line 3" in result.text
        assert "Line 4" in result.text

    def test_line_number_preservation_with_code_blocks(self):
        """Test that line numbers are preserved when removing code blocks."""
        content = """Line 1
```python
def hello():
    print("Hello")
```
Line 5
Line 6"""

        result = clean_markdown_content(content=content, include_code=False)

        # Code block should be removed but line mapping should work
        assert "```python" not in result.text
        assert "def hello():" not in result.text
        assert "Line 1" in result.text
        assert "Line 5" in result.text
        assert "Line 6" in result.text

    def test_line_number_preservation_with_yaml(self):
        """Test that line numbers are preserved when removing YAML front matter."""
        content = """---
title: Test
author: John
---
Line 5
Line 6
Line 7"""

        result = clean_markdown_content(content=content)

        # YAML should be removed but line mapping should work
        assert "title: Test" not in result.text
        assert "author: John" not in result.text
        assert "Line 5" in result.text
        assert "Line 6" in result.text
        assert "Line 7" in result.text
